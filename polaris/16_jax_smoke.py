"""Polaris smoke of the native JAX driver (S6b): a short real-data training on one A100, fp32, then a resume.

1. Generate the config (``03_make_config.py`` with ``HGQ_PRECISION=32 HGQ_DEVICES=1``; ``--tiny``: test sizes, dummy
   events, CPU).
2. Run A: ``main_jax fit`` for ``--epochs`` epochs of ``--micro-per-epoch`` micro-batches (``limit_train_batches``),
   validation each epoch, a step checkpoint every ``--ckpt-every`` optimizer steps (all kept), the persistent
   compilation cache.
3. Run B: a new process resumes from A's step-``--resume-step`` checkpoint (inside an epoch) in its own run directory
   and trains to the same end.
4. Checks (``RESULT`` / ``--out`` JSON, written after every stage): the logged train loss falls (mean of the last
   ``--window`` logged steps below the first), the val loss falls from the first to the last epoch, the quantizer
   variables move (first vs final checkpoint), ``metrics.csv`` / ``steps.csv`` exist, B's final state equals A's within
   ``--rtol`` (bitwise on CPU; GPU determinism is not assumed, S6a open issue 3: the bitwise count is reported), B's
   logged losses after the resume equal A's within the same tolerance, and B's compiles hit the cache.

    python polaris/16_jax_smoke.py --run-root logs/jax_smoke --data-dir $DATA_ROOT --out logs/jax_smoke.json
    python polaris/16_jax_smoke.py --tiny --run-root /tmp/jax_smoke --out /tmp/jax_smoke.json     # local check
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jax_common import REPO, generate_config, shrink_config

TINY = {"dim": 16, "heads": 2, "enc_layers": 1, "dec_layers": 1, "linformer_k": 8, "nodes": 64, "queries": 16}
QUANTIZER_SUFFIXES = {"b", "f", "i"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", required=True)
    p.add_argument("--config", default="", help="generated config (empty: run 03_make_config.py)")
    p.add_argument("--data-dir", default="")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-per-epoch", type=int, default=400, help="limit_train_batches (micro-batches per epoch)")
    p.add_argument("--num-val", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--resume-step", type=int, default=150)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--rtol", type=float, default=1e-3, help="resume tolerance on the GPU (max |a - b| / max |a| per array)")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--compile-cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def make_config(args, root: Path) -> Path:
    if args.config:
        return Path(args.config)
    # CKPT_DIR is not used by the native driver (it writes under --run-dir); set so nothing lands in the repo
    env = {"HGQ_PRECISION": "32", "HGQ_DEVICES": "1", "HGQ_BATCH": str(args.batch), "HGQ_ACCUM": str(args.accum), "CKPT_DIR": str(root / "unused_ckpts")}
    if args.data_dir:
        env["DATA_DIR"] = args.data_dir
    path = generate_config(root / "config.yaml", **env)
    if args.tiny:
        cfg = shrink_config(yaml.safe_load(path.read_text()), **TINY)
        cfg["data"]["num_workers"] = 0
        cfg["trainer"]["accelerator"] = "cpu"
        # a 40x larger LR, so that a few dummy-data steps lower the loss (as tests/keras_native/s6b_fixtures)
        cfg["model"]["lrs_config"].update(initial=2e-4, max=2e-3, end=2e-5, pct_start=0.3)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def fit(args, config: Path, run_dir: Path, *extra) -> dict:
    cmd = [
        args.python,
        "-m",
        "hepattn.experiments.clic.main_jax",
        "fit",
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--max-epochs",
        str(args.epochs),
        "--limit-train-batches",
        str(args.micro_per_epoch),
        "--num-train",
        str(args.micro_per_epoch * args.batch),
        "--num-val",
        str(args.num_val),
        "--log-every-n-steps",
        str(args.log_every),
        "--ckpt-every-n-steps",
        str(args.ckpt_every),
        "--ckpt-save-top-k",
        "-1",
        "--compile-cache",
        args.compile_cache,
        *extra,
    ]
    if args.tiny:
        cmd.append("--dummy-data")
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    env = {**os.environ, "KERAS_BACKEND": "jax", "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    print("### " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, check=False)
    (run_dir.parent / f"{run_dir.name}.log").write_text(cp.stdout + "\n" + cp.stderr)
    line = next((ln for ln in reversed(cp.stdout.splitlines()) if ln.startswith("RESULT ")), None)
    if cp.returncode != 0 or line is None:
        raise RuntimeError(f"{run_dir.name} failed:\n" + "\n".join(cp.stderr.splitlines()[-30:]))
    res = json.loads(line[len("RESULT ") :])
    res["wall_s"] = time.perf_counter() - t0
    return res


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: float(v) if v else math.nan for k, v in r.items()} for r in csv.DictReader(f)]


def arrays(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k != "__meta__"}


def compare(a: dict, b: dict) -> dict:
    """Bitwise count and the worst relative difference (max |a - b| / max(max |a|, tiny)) over the arrays."""
    assert set(a) == set(b), sorted(set(a) ^ set(b))[:5]
    bitwise, worst, worst_key = 0, 0.0, None
    for key in a:
        x, y = a[key], b[key]
        if np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            bitwise += 1
            continue
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        rel = float(np.nanmax(np.abs(xf - yf)) / max(float(np.nanmax(np.abs(xf))), 1e-30))
        if rel > worst:
            worst, worst_key = rel, key
    return {"arrays": len(a), "bitwise_equal": bitwise, "max_rel": worst, "max_rel_key": worst_key}


def main(args) -> int:
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.compile_cache:
        args.compile_cache = str(root / "jax_cache")
    if args.tiny:
        args.batch, args.accum = 4, 4
        args.micro_per_epoch, args.num_val, args.log_every, args.ckpt_every, args.resume_step = 16, 10, 1, 2, 6
    out = Path(args.out) if args.out else root / "smoke.json"
    result: dict = {"args": vars(args)}

    def save() -> None:
        out.write_text(json.dumps(result, indent=1, default=str))

    config = make_config(args, root)
    result["config"] = str(config)
    run_a, run_b = root / "run_a", root / "run_b"
    result["run_a"] = fit(args, config, run_a)
    save()
    resume_ckpt = run_a / "ckpts" / f"step-{args.resume_step:09d}.npz"
    result["run_b"] = fit(args, config, run_b, "--resume", str(resume_ckpt))
    save()

    steps_a, epochs_a = rows(run_a / "steps.csv"), rows(run_a / "metrics.csv")
    losses = [r["train/loss"] for r in steps_a]
    final = f"step-{result['run_a']['global_step']:09d}.npz"
    first = min((run_a / "ckpts").glob("step-*.npz"))
    moved = [k for k, v in compare_keys(arrays(first), arrays(run_a / "ckpts" / final))]
    state = compare(arrays(run_a / "ckpts" / final), arrays(run_b / "ckpts" / final))
    by_step = {r["step"]: r for r in steps_a}
    resumed_rows = rows(run_b / "steps.csv")
    loss_rel = max(
        (abs(r["train/loss"] - by_step[r["step"]]["train/loss"]) / abs(by_step[r["step"]]["train/loss"]) for r in resumed_rows), default=math.nan
    )
    checks = {
        "loss_falls": float(np.mean(losses[-args.window :])) < float(np.mean(losses[: args.window])),
        "val_loss_falls": epochs_a[-1]["val/loss"] < epochs_a[0]["val/loss"],
        "quantizers_move": len(moved) > 0,
        "metrics_csv": len(epochs_a) == args.epochs and len(steps_a) > 0,
        "resume_state_within_rtol": state["max_rel"] <= args.rtol,
        "resume_losses_within_rtol": loss_rel <= args.rtol,
        "resume_hits_cache": result["run_b"]["cache_events"]["cache_hits"] > 0,
    }
    result.update(
        train_loss_first=losses[: args.window],
        train_loss_last=losses[-args.window :],
        val_loss_by_epoch=[r["val/loss"] for r in epochs_a],
        quantizer_arrays_moved=len(moved),
        bits_by_epoch={s: [r.get(f"val/bits_{s}_mean") for r in epochs_a] for s in QUANTIZER_SUFFIXES},
        ebops_by_epoch=[r.get("val/ebops") for r in epochs_a],
        beta_by_epoch=[r.get("train/quant_beta") for r in epochs_a],
        resume_state=state,
        resume_loss_max_rel=loss_rel,
        checks=checks,
        passed=all(checks.values()),
    )
    save()
    print(json.dumps({"checks": checks, "resume_state": state, "resume_loss_max_rel": loss_rel}, indent=1), flush=True)
    print(f"SMOKE-{'PASS' if result['passed'] else 'FAIL'} {out}", flush=True)
    return 0 if result["passed"] else 1


def compare_keys(a: dict, b: dict):
    """The trainable quantizer variables (``tv/.../b|f|i``) that differ between ``a`` and ``b``.

    Yields:
        (key, array of ``a``) per differing variable.
    """
    for key, x in a.items():
        if key.startswith("tv/") and key.rsplit("/", 1)[-1] in QUANTIZER_SUFFIXES and not np.array_equal(x, b[key]):
            yield key, x


if __name__ == "__main__":
    sys.exit(main(parse_args()))
