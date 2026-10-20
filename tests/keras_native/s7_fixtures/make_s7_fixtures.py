r"""Write the S7 converter fixtures: two real Lightning runs at test sizes and the hybrid's outputs from its checkpoint.

    KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu PYTHONPATH=<worktree>/src <hepattn>/.venv/bin/python \\
        tests/keras_native/s7_fixtures/make_s7_fixtures.py

Run in the hybrid torch env (``hepattn/.venv``: comet_ml, lightning, the hybrid). Both runs go through the production
entry points, so the files are what a collaborator's run directory holds:

- ``org_float/``: ``main.py fit`` (``MPflow``, the org float torch model) on org ``base.yaml`` + ``linformer.yaml``,
  printed by LightningCLI (``--print_config``, the form ``SaveConfig`` saves) and shrunk to ``SIZES``;
- ``hybrid/``: ``main_hgq.py fit`` (``MPflowHGQ``, ``KerasMaskFormer``) on the ``polaris/03_make_config.py`` config at
  ``SIZES`` (the S6b tiny config with 2 + 2 layers, so value residuals and an intermediate decoder layer exist).

Each run trains one epoch of dummy CLIC events on the CPU (fp32, the hybrid with its BetaPID / PeriodicEBOPs callbacks),
and hepattn's ``Checkpoint`` writes ``ckpts/epoch=000-val_loss=....ckpt``. Kept: the ``config.yaml`` LightningCLI saved
in the run directory, and the checkpoint as ``model.ckpt``. Test-only changes to the configs: sizes, CPU / fp32,
dummy data, no logger, the scipy matcher without workers, and only the callbacks a CPU run needs (``SaveConfig``,
``Checkpoint``, and the hybrid's beta / EBOPs callbacks).

``hybrid/outputs.npz``: 3 dummy events (``in::<key>``) and the hybrid's eval-mode outputs (``out::<path>``) after
loading ``model.ckpt`` into a model built from the saved ``config.yaml`` (the ``evaluate_hgq`` path: build, materialize,
register the keras parameters, strict load). The native converter gate compares against these.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLIC = REPO / "src/hepattn/experiments/clic"
sys.path.insert(0, str(REPO / "polaris"))
from jax_common import generate_config, shrink_config  # noqa: E402

SIZES = {"dim": 16, "heads": 2, "enc_layers": 2, "dec_layers": 2, "linformer_k": 8, "nodes": 64, "queries": 16}
SCALE = "src/hepattn/experiments/clic/configs/clic_var_transform.yaml"
NUM_EVENTS = 3
HYBRID_ENV = {  # tiny_polaris.yaml's generator settings (s6b_fixtures/make_tiny_config.py)
    "HGQ_PRECISION": "32",
    "HGQ_DEVICES": "1",
    "HGQ_ACCUM": "2",
    "HGQ_EBOPS_EVERY": "2",
    "HGQ_PID_WARMUP_EPOCHS": "0",
    "HGQ_TARGET_EBOPS": "8e7",
    "HGQ_INIT_BETA": "5e-8",
    "HGQ_MAX_BETA": "5e-6",
    "HGQ_MIN_BETA": "5e-13",
    "DATA_DIR": "/nonexistent/clic",
    "CKPT_DIR": "/nonexistent/ckpts",
}
KEEP_HYBRID_CALLBACKS = ("BetaPID", "EBOPsMonitor", "PeriodicEBOPs")


def env() -> dict:
    return {**os.environ, "KERAS_BACKEND": "torch", "KERAS_TORCH_DEVICE": "cpu", "PYTHONPATH": str(REPO / "src")}


def cli(module: str, configs: list[str], *extra: str, check: bool = True) -> subprocess.CompletedProcess:
    args = [sys.executable, "-m", module, "fit"]
    for config in configs:
        args += ["--config", config]
    proc = subprocess.run([*args, *extra], cwd=REPO, env=env(), capture_output=True, text=True, check=False)
    if check and proc.returncode:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-8000:])
        proc.check_returncode()
    return proc


def shrink_normalised(cfg: dict) -> dict:
    """``shrink_config`` for a printed config, whose encoder / decoder are ``class_path`` / ``init_args`` entries."""
    init = cfg["model"]["model"]["init_args"]
    wrappers = {key: init[key] for key in ("encoder", "decoder")}
    for key, entry in wrappers.items():
        init[key] = entry["init_args"]
    out = shrink_config(cfg, **SIZES)
    out_init = out["model"]["model"]["init_args"]
    out_init["encoder"]["dim"] = SIZES["dim"]  # the printed encoder has its own dim (the generated config has none)
    for key, entry in wrappers.items():
        out_init[key] = {**entry, "init_args": out_init[key]}
    return out


def leaves(node, path: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}[{i}]")
    elif isinstance(node, int) and not isinstance(node, bool):
        yield path, node


def tiny_run_settings(cfg: dict, run_dir: Path, callbacks: list) -> dict:
    cfg["seed_everything"] = 42
    cfg["data"].update(batch_size=4, num_workers=0, num_train=16, num_val=4, num_test=4, scale_dict_path=SCALE, dummy_data=True, pin_memory=False)
    t = cfg["trainer"]
    t.update(
        accelerator="cpu",
        devices=1,
        precision="32",
        max_epochs=1,
        max_steps=-1,
        logger=False,
        callbacks=callbacks,
        default_root_dir=str(run_dir),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    # a larger LR than the configs' (as tiny_polaris.yaml), so that 4 steps move the weights and quantizers visibly
    cfg["model"]["lrs_config"].update(initial=2e-4, max=2e-3, end=2e-5, pct_start=0.3)
    init = cfg["model"]["model"]["init_args"]
    init["matcher"] = {
        "class_path": "hepattn.models.matcher.Matcher",
        "init_args": {"default_solver": "scipy", "adaptive_solver": False, "parallel_solver": False},
    }
    for task in init["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task["init_args"]:
            task["init_args"]["scale_dict_path"] = SCALE
    return cfg


def checkpoint_callback() -> dict:
    return {"class_path": "hepattn.callbacks.Checkpoint", "init_args": {"monitor": "val/loss", "log_model": False}}


def collect(root: Path, out: Path) -> None:
    runs = [p for p in root.iterdir() if p.is_dir()]  # hepattn's CLI runs in <default_root_dir>/<name>_<timestamp>
    assert len(runs) == 1, f"expected one run directory in {root}, got {runs}"
    run_dir = runs[0]
    ckpts = sorted((run_dir / "ckpts").glob("epoch=*.ckpt"))
    assert len(ckpts) == 1, f"expected one epoch checkpoint in {run_dir}/ckpts, got {ckpts}"
    config = run_dir / "config.yaml"
    assert config.is_file(), f"LightningCLI wrote no {config}: {sorted(p.name for p in run_dir.iterdir())}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config, out / "config.yaml")
    shutil.copy(ckpts[0], out / "model.ckpt")
    (out / "source.json").write_text(json.dumps({"run_dir": run_dir.name, "checkpoint": ckpts[0].name}, indent=1) + "\n")


def make_org_float(tmp: Path) -> None:
    printed = cli(
        "hepattn.experiments.clic.main",
        ["src/hepattn/experiments/clic/configs/base.yaml", "src/hepattn/experiments/clic/configs/linformer.yaml"],
        "--print_config",
    )
    cfg = shrink_normalised(yaml.safe_load(printed.stdout))
    run_dir = tmp / "org_run"
    cfg = tiny_run_settings(cfg, run_dir, [{"class_path": "hepattn.callbacks.SaveConfig"}, checkpoint_callback()])
    leftover = sorted({path for path, value in leaves(cfg["model"]) if value in {256, 518, 168, 160, 150}})
    assert not leftover, f"sizes the shrink missed: {leftover}"
    path = tmp / "org_tiny.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cli("hepattn.experiments.clic.main", [str(path)])
    collect(run_dir, HERE / "org_float")


def make_hybrid(tmp: Path) -> None:
    cfg = yaml.safe_load(generate_config(tmp / "polaris_hgq.yaml", **HYBRID_ENV).read_text())
    cfg = shrink_config(cfg, **SIZES)
    keep = [cb for cb in cfg["trainer"]["callbacks"] if cb["class_path"].rsplit(".", 1)[-1] in KEEP_HYBRID_CALLBACKS]
    assert len(keep) == len(KEEP_HYBRID_CALLBACKS), keep
    run_dir = tmp / "hybrid_run"
    cfg = tiny_run_settings(cfg, run_dir, [{"class_path": "hepattn.callbacks.SaveConfig"}, checkpoint_callback(), *keep])
    cfg["trainer"]["accumulate_grad_batches"] = 1
    path = tmp / "hybrid_tiny.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cli("hepattn.experiments.clic.main_hgq", [str(path)])
    collect(run_dir, HERE / "hybrid")


def flatten(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


@torch.no_grad()
def hybrid_outputs() -> None:
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from hepattn.experiments.clic.evaluate_hgq import load_state  # noqa: PLC0415
    from hepattn.experiments.clic.pflow_data import CLICDataset  # noqa: PLC0415
    from hepattn.experiments.clic.port_to_keras import build_keras_maskformer_from_config  # noqa: PLC0415

    cfg = yaml.safe_load((HERE / "hybrid/config.yaml").read_text())
    d = cfg["data"]
    kwargs = {k: d[k] for k in ("inputs", "targets", "num_objects", "max_nodes", "incidence_cutval") if k in d}
    ds = CLICDataset(filepath="unused", scale_dict_path=str(REPO / SCALE), num_events=NUM_EVENTS, dummy_data=True, **kwargs)
    inputs, _ = next(iter(DataLoader(ds, batch_size=NUM_EVENTS)))
    assert (~inputs["node_valid"]).any(-1).all(), "every event needs a padded node"
    model_cfg = cfg["model"]["model"]
    model = build_keras_maskformer_from_config(model_cfg, quant=model_cfg["init_args"].get("quant")).eval()
    model.materialize(inputs)
    model.register_keras_parameters()
    load_state(model, HERE / "hybrid/model.ckpt")
    outputs = flatten(model(inputs))
    arrays = {f"in::{k}": v.numpy() for k, v in inputs.items()}
    arrays |= {f"out::{k}": v.detach().numpy() for k, v in outputs.items()}
    np.savez(HERE / "hybrid/outputs.npz", **arrays)
    print("hybrid outputs:", len(outputs), "arrays")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        make_org_float(Path(tmp))
        make_hybrid(Path(tmp))
    hybrid_outputs()


if __name__ == "__main__":
    main()
