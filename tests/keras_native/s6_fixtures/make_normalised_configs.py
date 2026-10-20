"""Write the jsonargparse-normalised run configs that ``test_config.py`` loads (S6a).

A saved run ``config.yaml`` (``SaveConfig``) is what LightningCLI prints with ``--print_config``: defaults expanded,
activations and the decoder in ``class_path`` / ``init_args`` form. This script prints it for three configs and
writes them next to itself:

- ``org_normalised.yaml``: ``main.py fit --config base.yaml --config linformer.yaml`` (org torch model);
- ``polaris_normalised.yaml``: ``main_hgq.py fit --config <polaris/03_make_config.py output>`` (the hybrid run config);
- ``pflow_hgq_normalised.yaml``: ``main_hgq.py fit`` (its default config, ``pflow_hgq.yaml``);
- ``org_eval_normalised.yaml``: ``main.py fit --config base.yaml --config linformer.yaml --config eval.yaml``;
- ``org_partial_normalised.yaml``: the same with ``overlay_partial_dicts.yaml`` (next to this script) as the third file;
- ``lightningcli_probes.yaml``: base + linformer + the overlays in ``PROBES``: LightningCLI's return code, its error
  line, and the part of the printed config the probe is about.

The last three pin how LightningCLI merges several ``--config`` files (``config.merge_configs``).

The generated Polaris config holds absolute paths of this checkout; they are replaced by ``/polaris/home/hepattn``
so the fixture is machine-independent and looks like a config saved on another host.

Run in the hybrid torch env (``hepattn/.venv``), from anywhere:
    KERAS_BACKEND=torch PYTHONPATH=<worktree>/src <hepattn>/.venv/bin/python tests/keras_native/s6_fixtures/make_normalised_configs.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLIC = REPO / "src/hepattn/experiments/clic"
FAKE_REPO = "/polaris/home/hepattn"


ENCODER = "hepattn.models.Encoder"
CHECKPOINT = "hepattn.callbacks.Checkpoint"
INPUT_NETS = ["model", "model", "init_args", "input_nets"]
# overlays on base.yaml + linformer.yaml, and the printed key path each probe is about (None: LightningCLI rejects it)
PROBES = {
    # keys next to class_path: at the top-level class, a nested class and a callbacks item
    "short_model": ([{"model": {"model": {"class_path": "hepattn.models.MaskFormer", "input_sort_field": "eta"}}}], None),
    "short_encoder": ([{"model": {"model": {"init_args": {"encoder": {"class_path": ENCODER, "num_layers": 2}}}}}], None),
    "short_callback": ([{"trainer": {"callbacks": [{"class_path": CHECKPOINT, "monitor": "val/loss", "every_n_epochs": 2}]}}], None),
    # controls: the same entries with init_args
    "explicit_encoder": (
        [{"model": {"model": {"init_args": {"encoder": {"class_path": ENCODER, "init_args": {"num_layers": 2}}}}}}],
        ["model", "model", "init_args", "encoder"],
    ),
    "explicit_callback": (
        [{"trainer": {"callbacks": [{"class_path": CHECKPOINT, "init_args": {"monitor": "val/loss", "every_n_epochs": 2}}]}}],
        ["trainer", "callbacks"],
    ),
    # a ModuleList item without class_path: merged into base's item (class_path and posenc kept)
    "modulelist_bare_item": (
        [
            {
                "model": {
                    "model": {
                        "init_args": {
                            "input_nets": {
                                "init_args": {
                                    "modules": [
                                        {
                                            "init_args": {
                                                "input_name": "node",
                                                "fields": ["features"],
                                                "net": {"class_path": "hepattn.models.Dense", "init_args": {"input_size": 10, "output_size": 256}},
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        ],
        INPUT_NETS,
    ),
    # the torch decoder's posenc is dict[str, float]: the second overlay replaces the first whole
    "decoder_posenc": (
        [
            {"model": {"model": {"init_args": {"decoder": {"posenc": {"alpha": 1.0, "base": 2.0}}}}}},
            {"model": {"model": {"init_args": {"decoder": {"posenc": {"alpha": 3.0}}}}}},
        ],
        ["model", "model", "init_args", "decoder"],
    ),
}


def run_cli(module: str, configs: list[str]) -> subprocess.CompletedProcess:
    args = [sys.executable, "-m", module, "fit"]
    for config in configs:
        args += ["--config", config]
    env = {**os.environ, "KERAS_BACKEND": "torch", "PYTHONPATH": str(REPO / "src")}
    return subprocess.run([*args, "--print_config"], cwd=CLIC, env=env, capture_output=True, text=True, check=False)


def print_config(module: str, configs: list[str]) -> str:
    proc = run_cli(module, configs)
    proc.check_returncode()
    return proc.stdout


def probes(org: list[str]) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (overlays, pick) in PROBES.items():
            files = []
            for i, overlay in enumerate(overlays):
                files.append(str(Path(tmp) / f"{name}_{i}.yaml"))
                Path(files[-1]).write_text(yaml.safe_dump(overlay))
            proc = run_cli("hepattn.experiments.clic.main", [*org, *files])
            errors = [line.strip() for line in proc.stderr.splitlines() if "is not expected" in line or "Not a valid subclass" in line]
            entry = {"overlays": overlays, "returncode": proc.returncode, "error": errors[0] if errors else None}
            if pick is not None and proc.returncode == 0:
                node = yaml.safe_load(proc.stdout)
                for key in pick:
                    node = node[key]
                entry["printed"] = {"path": pick, "value": node}
            assert (proc.returncode == 0) == (pick is not None), (name, proc.returncode, proc.stderr[-2000:])
            out[name] = entry
    return out


def main() -> None:
    org = ["configs/base.yaml", "configs/linformer.yaml"]
    (HERE / "org_normalised.yaml").write_text(print_config("hepattn.experiments.clic.main", org))
    (HERE / "org_eval_normalised.yaml").write_text(print_config("hepattn.experiments.clic.main", [*org, "configs/eval.yaml"]))
    partial = print_config("hepattn.experiments.clic.main", [*org, str(HERE / "overlay_partial_dicts.yaml")])
    (HERE / "org_partial_normalised.yaml").write_text(partial)
    header = "# generated by make_normalised_configs.py (LightningCLI --print_config on base + linformer + the overlays)\n"
    (HERE / "lightningcli_probes.yaml").write_text(header + yaml.safe_dump(probes(org), sort_keys=False))
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "polaris_hgq.yaml"
        env = {
            **os.environ,
            "HGQ_CONFIG_OUT": str(generated),
            "DATA_DIR": "/eagle/hgcal-maskformer-fpga/clic_data",
            "CKPT_DIR": f"{FAKE_REPO}/hgq_ckpts",
        }
        subprocess.run([sys.executable, str(REPO / "polaris/03_make_config.py")], env=env, check=True, capture_output=True)
        text = print_config("hepattn.experiments.clic.main_hgq", [str(generated)])
    (HERE / "polaris_normalised.yaml").write_text(text.replace(str(REPO), FAKE_REPO))
    (HERE / "pflow_hgq_normalised.yaml").write_text(print_config("hepattn.experiments.clic.main_hgq", []))
    names = (
        "org_normalised.yaml",
        "org_eval_normalised.yaml",
        "org_partial_normalised.yaml",
        "lightningcli_probes.yaml",
        "polaris_normalised.yaml",
        "pflow_hgq_normalised.yaml",
    )
    for name in names:
        assert str(REPO) not in (HERE / name).read_text(), name
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
