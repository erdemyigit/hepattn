r"""QAT warm start of the keras_native (JAX) driver from a float Lightning checkpoint: init, smoke, pilot arms, report.

Stages (``22_warmstart.pbs`` runs them on Polaris, ``WS_STAGE``):

``init``    The QAT config (``03_make_config.py`` with ``HGQ_SOURCE_CONFIG`` = the float run's saved config, fp32, one
            device, ``--weight-b0``, the PID setpoints; mini-epochs of ``--epoch-steps`` optimizer steps with a
            validation each, as G4), the converted checkpoint (``convert_to_native --target-config``) and the parity
            gate on the CPU (``21_warmstart_check.py parity``, with one mutant that must fail), then ``manifest.json``.
            The optimizer is not in the model hash, so every arm warm-starts from the same ``init.npz``.
``run``     One arm (``ARMS``: optimizer, peak LR, weight decay, quantizer LR scale) with ``main_jax fit --init-from``
            (``--resume last`` when the arm has a checkpoint); ``--max-steps`` caps it (the smoke). ``DONE`` at K.
``report``  Per arm: the first and last logged train rows, every validation (loss, EBOPs, beta, bits), ms per
            micro-batch. ``--out`` JSON.

    python polaris/22_warmstart.py init --root R --tag T --source-config RUN/config.yaml --ckpt RUN/x.ckpt --data-dir D \
        --model-ebops 4.8e10 --target-ebops 2.9e10 --init-beta 8.5e-11
    python polaris/22_warmstart.py run --root R --tag T --arm adamw [--max-steps 60 --run-name smoke]
    python polaris/22_warmstart.py report --root R --tag T --out logs/ws_T.json
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("g4", REPO / "polaris/19_g4_compare.py")
g4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g4)

# optimizer recipes; the model and the converted init are shared (optimizer and lrs_config are not in the model hash)
ARMS = {
    "adamw": {"optimizer": "AdamW", "max": 5e-5, "weight_decay": 1e-4},  # the G4-validated recipe
    "adamw-q20": {"optimizer": "AdamW", "max": 5e-5, "weight_decay": 1e-4, "quantizer_lr_scale": 20.0},
    "lion-6e5": {"optimizer": "Lion", "max": 6.2e-5, "weight_decay": 0.1},  # the small float reference run's recipe at 1/10 of its peak LR
    "lion-6e4": {"optimizer": "Lion", "max": 6.2e-4, "weight_decay": 0.1},  # the small float reference run's recipe
}
COLUMNS = ("train/loss", "train/ebops", "val/loss", "val/ebops", "beta", "val/bits_f_mean", "val/bits_b_mean", "time/ms_per_micro_batch")


def make_config(args, init_dir: Path) -> Path:
    import yaml  # noqa: PLC0415

    env = {
        "HGQ_SOURCE_CONFIG": str(Path(args.source_config).resolve()),
        "HGQ_MODEL_EBOPS": str(args.model_ebops),
        "HGQ_TARGET_EBOPS": str(args.target_ebops),
        "HGQ_INIT_BETA": str(args.init_beta),
        "HGQ_WEIGHT_B0": str(args.weight_b0),
        "HGQ_PRECISION": "32",
        "HGQ_DEVICES": "1",
        "HGQ_BATCH": str(args.batch),
        "HGQ_ACCUM": str(args.accum),
        "HGQ_PID_WARMUP_EPOCHS": str(args.pid_warmup_epochs),
        "DATA_DIR": args.data_dir,
        "CKPT_DIR": str(init_dir / "unused_ckpts"),
    }
    path = g4.generate_config(init_dir / "generated.yaml", **env)
    cfg = yaml.safe_load(path.read_text())
    s = {
        "k": args.k,
        "epoch_steps": args.epoch_steps,
        "accum": args.accum,
        "log_every": args.log_every,
        "workers": args.workers,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "ckpt_every": args.ckpt_every,
        "pid_warmup_epochs": args.pid_warmup_epochs,
    }
    cfg = g4.g4_overrides(cfg, s, init_dir / "unused_ckpts")
    out = init_dir / "config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def cmd_init(args) -> int:
    root = Path(args.root) / args.tag
    init_dir = root / "init"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        raise SystemExit(f"{manifest_path} is complete; a new init needs a new --tag")
    init_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, init_dir)
    check_onecycle(config, args.k)
    log = init_dir / "init.log"
    cpu = g4.env_with_src(KERAS_BACKEND="jax", JAX_PLATFORMS="cpu")
    convert = [args.jax_python, "-m", "hepattn.experiments.clic.convert_to_native", "--config", args.source_config, "--ckpt", args.ckpt]
    convert += ["--target-config", config, "--out", init_dir / "init.npz", "--report", init_dir / "convert.json", "--overwrite"]
    if g4.run_logged(convert, cpu, log) != 0:
        raise SystemExit(f"conversion failed: {log}")
    parity = [args.jax_python, REPO / "polaris/21_warmstart_check.py", "parity", "--config", args.source_config, "--ckpt", args.ckpt]
    parity += ["--val", f"{args.data_dir}/val_clic_fix.root", "--num-events", "200", "--parity-events", "32", "--threads", "8"]
    parity += ["--qat-config", config, "--mutant", "tensor:register_tokens:1e-3", "--out", init_dir / "parity.json"]
    code = g4.run_logged(parity, cpu, log)
    report = json.loads((init_dir / "parity.json").read_text())
    conv = json.loads((init_dir / "convert.json").read_text())
    manifest = {
        "complete": code == 0,
        "git": g4.git_head(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(config),
        "native_init": str(init_dir / "init.npz"),
        "source": {"config": args.source_config, "ckpt": args.ckpt, "ckpt_sha256": conv["checkpoint"]["sha256"]},
        "settings": {k: v for k, v in vars(args).items() if k not in {"func", "command"}},
        "config_hash": conv["config_hash"],
        "renorm_lane": conv.get("renorm_lane"),
        "parity_ok": report["ok"],
        "parity_worst": {t: e["base"]["gate"] for t, e in report["targets"].items()},
        "files_sha256": {p.name: g4.sha256(p) for p in (config, init_dir / "init.npz", init_dir / "parity.json")},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(json.dumps({"complete": manifest["complete"], "parity_ok": manifest["parity_ok"], "config_hash": manifest["config_hash"]}))
    return 0 if manifest["complete"] else 1


def check_onecycle(config: Path, total_steps: int) -> None:
    """OneCycle needs pct_start x total steps > 1 (``optim.OneCycleLR``, as torch); fail before any job time is spent.

    Raises:
        SystemExit: If the run is too short for the config's ``pct_start``.
    """
    import yaml  # noqa: PLC0415

    pct = float(yaml.safe_load(config.read_text())["model"]["lrs_config"]["pct_start"])
    if pct * total_steps <= 1:
        raise SystemExit(f"pct_start {pct} x {total_steps} steps <= 1: OneCycle needs more steps (> {1 / pct:.0f})")


def arm_config(manifest: dict, arm: str, arm_dir: Path) -> Path:
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(Path(manifest["config"]).read_text())
    recipe = ARMS[arm]
    cfg["model"]["optimizer"] = recipe["optimizer"]
    lrs = cfg["model"]["lrs_config"]
    lrs["max"], lrs["weight_decay"] = recipe["max"], recipe["weight_decay"]
    if "quantizer_lr_scale" in recipe:
        lrs["quantizer_lr_scale"] = recipe["quantizer_lr_scale"]
    out = arm_dir / "arm_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def cmd_run(args) -> int:
    root = Path(args.root) / args.tag
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit("the init is not complete (manifest.json): run the init stage first")
    arm_dir = root / (args.run_name or args.arm)
    arm_dir.mkdir(parents=True, exist_ok=True)
    k = args.max_steps or manifest["settings"]["k"]
    if (arm_dir / "DONE").is_file():
        print(json.dumps({"arm": args.arm, "done": True}))
        return 0
    config = arm_config(manifest, args.arm, arm_dir)
    check_onecycle(config, k)
    cmd = [args.jax_python, "-m", "hepattn.experiments.clic.main_jax", "fit", "--config", config, "--run-dir", arm_dir]
    cmd += ["--seed", "42", "--matmul-precision", "highest", "--compile-cache", args.compile_cache]
    cmd += ["--ckpt-every-n-steps", str(manifest["settings"]["ckpt_every"])]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.num_val:
        cmd += ["--num-val", str(args.num_val)]
    last = arm_dir / "ckpts" / "last.json"
    cmd += ["--resume", "last"] if last.is_file() else ["--init-from", manifest["native_init"]]
    t0 = time.time()
    code = g4.run_logged(cmd, g4.env_with_src(KERAS_BACKEND="jax"), arm_dir / "logs" / f"{os.environ.get('PBS_JOBID', 'local')}.log")
    end = g4.progress_step(arm_dir)
    if code == 0 and end >= k:
        (arm_dir / "DONE").write_text(f"{end}\n")
    segment = {"job": os.environ.get("PBS_JOBID", "local"), "end_step": end, "wall_s": time.time() - t0, "returncode": code}
    with (arm_dir / "segments.jsonl").open("a") as f:
        f.write(json.dumps(segment) + "\n")
    print(json.dumps({"arm": args.arm, **segment}))
    return code


def pick(row: dict) -> dict:
    return {c: row[c] for c in COLUMNS if c in row and not (isinstance(row[c], float) and math.isnan(row[c]))}


def cmd_report(args) -> int:
    root = Path(args.root) / args.tag
    out = {"tag": args.tag, "arms": {}}
    for arm_dir in sorted(p for p in root.iterdir() if (p / "steps.csv").is_file()):
        steps, metrics = g4.read_rows(arm_dir / "steps.csv"), g4.read_rows(arm_dir / "metrics.csv")
        ms = [r["time/ms_per_micro_batch"] for r in steps if not math.isnan(r.get("time/ms_per_micro_batch", math.nan))]
        out["arms"][arm_dir.name] = {
            "steps_logged": len(steps),
            "first_train": {"step": steps[0].get("step"), **pick(steps[0])} if steps else None,
            "last_train": {"step": steps[-1].get("step"), **pick(steps[-1])} if steps else None,
            "validations": [{"step": r.get("step"), "epoch": r.get("epoch"), **pick(r)} for r in metrics],
            "median_ms_per_micro_batch": statistics.median(ms) if ms else None,
            "nonfinite_train_loss": sum(1 for r in steps if not math.isfinite(r.get("train/loss", 0.0))),
            "done": (arm_dir / "DONE").is_file(),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for name, a in out["arms"].items():
        last_val = a["validations"][-1] if a["validations"] else {}
        print(
            f"{name:10s} steps {a['last_train']['step'] if a['last_train'] else 0:>6} first train/loss "
            f"{(a['first_train'] or {}).get('train/loss', float('nan')):.4f} last val/loss {last_val.get('val/loss', float('nan')):.4f} "
            f"val/ebops {last_val.get('val/ebops', float('nan')):.3e} ms/micro-batch {a['median_ms_per_micro_batch']}"
            f" nonfinite {a['nonfinite_train_loss']}"
        )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="22_warmstart", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--tag", required=True)
        if name in {"init", "run"}:
            p.add_argument("--jax-python", default=sys.executable)
    init = sub.choices["init"]
    init.add_argument("--source-config", required=True, help="the float run's saved config.yaml")
    init.add_argument("--ckpt", required=True, help="its Lightning checkpoint")
    init.add_argument("--data-dir", required=True)
    init.add_argument("--model-ebops", type=float, required=True, help="EBOPs of the QAT model at the converted weights")
    init.add_argument("--target-ebops", type=float, required=True)
    init.add_argument("--init-beta", type=float, required=True)
    init.add_argument("--weight-b0", type=int, default=12)
    init.add_argument("--batch", type=int, default=128)
    init.add_argument("--accum", type=int, default=1)
    init.add_argument("--k", type=int, default=2000, help="optimizer steps per arm")
    init.add_argument("--epoch-steps", type=int, default=250, help="optimizer steps per mini-epoch (one validation each)")
    init.add_argument("--pid-warmup-epochs", type=int, default=2)
    init.add_argument("--log-every", type=int, default=10)
    init.add_argument("--ckpt-every", type=int, default=200)
    init.add_argument("--workers", type=int, default=4)
    init.add_argument("--num-train", type=int, default=400000)
    init.add_argument("--num-val", type=int, default=5000)
    run = sub.choices["run"]
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--run-name", help="run directory name (default: the arm)")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--num-val", type=int)
    run.add_argument("--compile-cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    sub.choices["report"].add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return {"init": cmd_init, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
