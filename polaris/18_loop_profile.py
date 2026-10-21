"""Where does the native driver's time per micro-batch go? Short profiled fits in variants that isolate the phases.

The S6b smoke (16_jax_smoke) ran at ~0.47-0.49 s per micro-batch while the compiled step alone (G3, 17_g3_step) is
161.6 ms at batch 32. Each variant below is one ``main_jax fit`` process (or one loader process) with
``--profile-phases`` (``hepattn.keras_native.profile``); docs/hgq/port_log/PROF.md has the phase definitions.

Variants (``--variants``, default ``DEFAULT_VARIANTS``; each fit is one epoch of ``--micro`` micro-batches):

- ``smoke``: the smoke's settings: step checkpoint every 50 optimizer steps (all kept), log every 10, validation at
  the epoch end, ``XLA_FLAGS`` + ``--det-flags`` (16_jax_smoke.pbs sets ``--xla_gpu_deterministic_ops=true``;
  17_g3_step.pbs does not);
- ``smoke_nodet``: ``smoke`` without the determinism flag;
- ``prod``: the production settings (step checkpoint every 1000 optimizer steps, newest kept; log every 200;
  validation). At the default ``--micro 320`` (80 optimizer steps) neither period is reached, so this is ``pure`` +
  validation + the epoch checkpoint, not the production cadence;
- ``pure``: no validation, no step checkpoint, no log (the bare loop; the train-end ``last`` checkpoint remains);
- ``pure_det``: ``pure`` with the determinism flag (``pure`` vs ``pure_det`` decides H7: both load warm programs);
- ``pure_log`` / ``pure_noprof``: ``pure`` with a log row every ``--pert-log-every`` optimizer steps, with and without
  the profiler (perturbation control). ``steps.csv``'s ``time/ms_per_micro_batch`` is the wall time between two log
  rows over the micro-batches between them (synced at both ends, the log row itself outside). The windows after the
  first hold no compile (both train programs compile in the first two optimizer steps) and no one-time cost, so
  ``perturbation`` = median window of ``pure_log`` / median window of ``pure_noprof`` is the profiler's cost on the
  steady loop;
- ``pure_sync``: ``pure`` with ``--profile-sync`` (every step blocks: its device time inside the loop);
- ``pure_prefetch``: ``pure`` with ``--prefetch 2`` (batches read ahead in a thread);
- ``pure_trace``: ``pure`` with a ``jax.profiler`` trace of micro-batches ``--trace-window`` (GPU timeline; the trace
  start / stop is phase ``trace``, so its s/micro is not comparable with the other variants);
- ``loader<N>``: the train loader alone (``ClicSource`` + ``HostBatches`` with ``device_put``), ``N`` workers,
  ``--loader-batches`` batches after the first (worker start).

Results (``--out``, written after every variant): per variant the RESULT summary, the profile summary (phase table)
and the derived ``s_per_micro`` (wall / micro-batches), ``s_per_micro_excl_compile``, ``s_per_micro_steady`` (without
the compiles and the first train / validation batch: profile.py) and ``overhead_vs_g3_ms`` (steady rate minus
``--g3-step-ms``). A variant starts only if the time since the job start (``$PROF_JOB_START``, set by the pbs; else
this script's start) plus the longest variant so far stays within ``--budget-min``; the rest are recorded as skipped.
After each fit its checkpoints are deleted (1.1 GB each at production dims; ``--keep-ckpts`` keeps them); the save
timings are in the profile.

    python polaris/18_loop_profile.py --run-root $WORK_ROOT/loop_profile --data-dir $DATA_ROOT --out logs/loop_profile.json
    python polaris/18_loop_profile.py --tiny --run-root /tmp/lp --out /tmp/lp.json             # local check (CPU, dummy)
    python polaris/18_loop_profile.py --dummy-data --micro 16 ...                               # production dims, dummy
    python polaris/18_loop_profile.py smoke-timing logs/jax_smoke_<jobid>.json                  # H0 from a smoke JSON
    python polaris/18_loop_profile.py report logs/loop_profile_<a>.json [logs/loop_profile_<b>.json]  # table, merged
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jax_common import REPO, generate_config, shrink_config

TINY = {"dim": 16, "heads": 2, "enc_layers": 1, "dec_layers": 1, "linformer_k": 8, "nodes": 64, "queries": 16}
NEVER = 10**8
PERT = -1  # log period placeholder: --pert-log-every
MIN_WINDOWS = 3  # log windows after the first that the perturbation needs in each run
# H7 pairs first (determinism flag on / off), then the perturbation pair, so a walltime cut leaves both comparisons
DEFAULT_VARIANTS = "smoke,smoke_nodet,pure,pure_det,pure_log,pure_noprof,loader8,prod,pure_sync,pure_prefetch,pure_trace,loader4,loader16"
FIT_VARIANTS = {
    # name: (step ckpt every, log every, validate, determinism flag, extra driver args, profiled)
    "smoke": (50, 10, True, True, [], True),
    "smoke_nodet": (50, 10, True, False, [], True),
    "prod": (1000, 200, True, False, [], True),
    "pure": (NEVER, NEVER, False, False, [], True),
    "pure_det": (NEVER, NEVER, False, True, [], True),
    "pure_log": (NEVER, PERT, False, False, [], True),
    "pure_noprof": (NEVER, PERT, False, False, [], False),
    "pure_sync": (NEVER, NEVER, False, False, ["--profile-sync"], True),
    "pure_prefetch": (NEVER, NEVER, False, False, ["--prefetch", "2"], True),
    "pure_trace": (NEVER, NEVER, False, False, [], True),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")
    st = sub.add_parser("smoke-timing", help="one-time costs vs steady micro-batch time of a 16_jax_smoke result (no GPU)")
    st.add_argument("smoke_json", help="logs/jax_smoke_<jobid>.json")
    st.add_argument("--run-root", default="", help="the smoke's run root (default: its --run-root from the JSON)")
    rp = sub.add_parser("report", help="table and perturbation of one or more result JSONs (partial ones too), merged")
    rp.add_argument("results", nargs="+", help="logs/loop_profile_<jobid>.json; a later file's variant replaces an earlier one's")
    lo = sub.add_parser("loader", help="(internal) time the train loader alone in this process")
    lo.add_argument("--config", required=True)
    lo.add_argument("--workers", type=int, required=True)
    lo.add_argument("--batches", type=int, required=True)
    lo.add_argument("--num-train", type=int, required=True)
    lo.add_argument("--dummy-data", action="store_true")
    lo.add_argument("--out", required=True)
    p.add_argument("--run-root", default="")
    p.add_argument("--config", default="", help="generated config (empty: run 03_make_config.py)")
    p.add_argument("--data-dir", default="")
    p.add_argument("--train-path", default="", help="override the config's train file (local runs)")
    p.add_argument("--valid-path", default="")
    p.add_argument("--tiny", action="store_true", help="test dims, dummy events, CPU (local check of every variant)")
    p.add_argument("--dummy-data", action="store_true", help="CLIC dummy events at the config's dims")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--micro", type=int, default=320, help="micro-batches per fit (one epoch)")
    p.add_argument("--num-val", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=None, help="loader workers of the fits (default: the config's)")
    p.add_argument("--loader-batches", type=int, default=100)
    p.add_argument("--variants", default=os.environ.get("PROF_VARIANTS") or DEFAULT_VARIANTS)
    p.add_argument("--det-flags", default="--xla_gpu_deterministic_ops=true", help="XLA flags of the *_det / smoke variants")
    p.add_argument("--trace-window", default="40:48")
    p.add_argument("--smoke-ckpt-every", type=int, default=None, help="override the smoke variants' checkpoint period")
    p.add_argument("--smoke-log-every", type=int, default=None)
    p.add_argument("--pert-log-every", type=int, default=8, help="log period (optimizer steps) of pure_log / pure_noprof")
    p.add_argument("--g3-step-ms", type=float, default=161.6, help="G3 step median (job 7763879, batch 32)")
    p.add_argument("--budget-min", type=float, default=55.0, help="minutes since the job start that the next variant must end within")
    start = os.environ.get("PROF_JOB_START")
    p.add_argument("--job-start", type=float, default=float(start) if start else None, help="epoch seconds (default: now)")
    p.add_argument("--keep-ckpts", action="store_true", help="keep each fit's checkpoints (default: delete after the fit)")
    p.add_argument("--compile-cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def make_config(args, root: Path) -> Path:
    if args.config:
        return Path(args.config)
    env = {"HGQ_PRECISION": "32", "HGQ_DEVICES": "1", "HGQ_BATCH": str(args.batch), "HGQ_ACCUM": str(args.accum), "CKPT_DIR": str(root / "unused")}
    if args.data_dir:
        env["DATA_DIR"] = args.data_dir
    path = generate_config(root / "config.yaml", **env)
    cfg = yaml.safe_load(path.read_text())
    if args.tiny:
        cfg = shrink_config(cfg, **TINY)
        cfg["data"]["num_workers"] = 0
        cfg["trainer"]["accelerator"] = "cpu"
    for key in ("train_path", "valid_path"):
        if getattr(args, key):
            cfg["data"][key] = getattr(args, key)
    # OneCycleLR needs pct_start * total_steps > 1; a short fit gets a longer warm-up (the LR does not change the timing)
    lrs, steps = cfg["model"]["lrs_config"], args.micro // args.accum
    if lrs.get("pct_start", 0.3) * steps <= 1:
        lrs["pct_start"] = min(0.9, 1.5 / steps)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def driver_env(xla_flags: str) -> dict:
    env = {**os.environ, "KERAS_BACKEND": "jax", "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    env["XLA_FLAGS"] = xla_flags
    return env


def run_fit(args, config: Path, root: Path, name: str) -> dict:
    ckpt_every, log_every, validate, det, extra, profiled = FIT_VARIANTS[name]
    if name.startswith("smoke"):
        ckpt_every = args.smoke_ckpt_every or ckpt_every
        log_every = args.smoke_log_every or log_every
    if log_every == PERT:
        log_every = args.pert_log_every
    run_dir = root / name
    base_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags = f"{base_flags} {args.det_flags}".strip() if det else base_flags
    cmd = [
        args.python, "-m", "hepattn.experiments.clic.main_jax", "fit", "--config", str(config), "--run-dir", str(run_dir),
        "--max-epochs", "1", "--limit-train-batches", str(args.micro), "--num-train", str(args.micro * args.batch),
        "--num-val", str(args.num_val), "--log-every-n-steps", str(log_every), "--ckpt-every-n-steps", str(ckpt_every),
        "--ckpt-save-top-k", "-1" if name.startswith("smoke") else "1",
        "--check-val-every-n-epoch", "1" if validate else str(NEVER), "--compile-cache", args.compile_cache, *extra,
    ]  # fmt: skip
    if profiled:
        cmd += ["--profile-phases", str(run_dir / "phases.csv")]
    if name == "pure_trace":
        cmd += ["--jax-trace", str(root / "jax_trace"), "--jax-trace-window", args.trace_window]
    if args.tiny or args.dummy_data:
        cmd.append("--dummy-data")
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    print(f"### {name}: XLA_FLAGS={xla_flags!r} " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, env=driver_env(xla_flags), cwd=REPO, capture_output=True, text=True, check=False)
    (root / f"{name}.log").write_text(cp.stdout + "\n" + cp.stderr)
    line = next((ln for ln in reversed(cp.stdout.splitlines()) if ln.startswith("RESULT ")), None)
    if cp.returncode != 0 or line is None:
        return {"error": "\n".join(cp.stderr.splitlines()[-30:]), "returncode": cp.returncode, "wall_s": time.perf_counter() - t0}
    res = {"result": json.loads(line[len("RESULT ") :]), "wall_s": time.perf_counter() - t0, "xla_flags": xla_flags}
    phases = run_dir / "phases.csv.json"
    if phases.is_file():
        res["profile"] = json.loads(phases.read_text())
    windows = log_windows(run_dir / "steps.csv")
    if windows:
        res["log_windows"] = windows
    if not args.keep_ckpts:
        res["ckpt_bytes_deleted"] = delete_ckpts(run_dir / "ckpts")
    return res


def delete_ckpts(directory: Path) -> int:
    """Delete a fit's checkpoint directory; returns the bytes it held."""
    if not directory.is_dir():
        return 0
    size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    shutil.rmtree(directory)
    return size


def log_windows(steps_csv: Path) -> dict | None:
    """``time/ms_per_micro_batch`` of a ``steps.csv``: the first window (compiles, first batch) and the ones after it."""
    if not steps_csv.is_file():
        return None
    with steps_csv.open(newline="") as f:
        ms = [float(r["time/ms_per_micro_batch"]) for r in csv.DictReader(f) if r.get("time/ms_per_micro_batch")]
    if not ms:
        return None
    rest = ms[1:]
    return {
        "n": len(ms),
        "first_ms": ms[0],
        "median_ms": statistics.median(ms),
        "median_ms_after_first": statistics.median(rest) if rest else math.nan,
        "mean_ms_after_first": statistics.fmean(rest) if rest else math.nan,
        "max_ms_after_first": max(rest) if rest else math.nan,
    }


def derive(res: dict, g3_step_ms: float) -> dict:
    """S per micro-batch (with and without the compile phases) and the overhead over the G3 step."""
    summary = res["result"]
    micro = summary["micro_step"]
    prof = res.get("profile")
    if prof is not None:
        per = prof["s_per_micro_batch_steady"]
        out = {
            "s_per_micro": prof["s_per_micro_batch"],
            "s_per_micro_excl_compile": prof["s_per_micro_batch_excl_compile"],
            "s_per_micro_steady": per,
            "one_time_s": prof["one_time_s"],
            "groups_ms_per_micro": {},
        }
        for group, seconds in prof["groups_s"].items():
            out["groups_ms_per_micro"][group] = 1e3 * seconds / micro
        out["unattributed_frac"] = prof["unattributed_frac"]
        out["steady"] = prof["steady"]
        out["match_callback"] = prof["concurrent"].get("match_callback")
    else:  # no profiler: only the fit wall time (later compiles / cache loads included); see perturbation()
        per = None
        out = {"s_per_micro": summary["fit_s"] / micro, "s_per_micro_excl_compile": None, "s_per_micro_steady": None}
    out["overhead_vs_g3_ms"] = None if per is None else 1e3 * per - g3_step_ms
    out["fit_s"] = summary["fit_s"]
    out["cache_misses"] = summary.get("cache_events", {}).get("cache_misses")
    return out


def loader_only(args) -> int:
    """Time ``--batches`` train batches (after the first) of the CLIC loader with ``--workers`` workers, on the device."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import jax  # noqa: PLC0415

    from hepattn.experiments.clic.main_jax import ClicSource  # noqa: PLC0415
    from hepattn.keras_native import profile  # noqa: PLC0415
    from hepattn.keras_native.cli import apply_overrides  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    rc = apply_overrides(load_config([args.config]), argparse.Namespace(num_workers=args.workers, num_train=args.num_train))
    t0 = time.perf_counter()
    source = ClicSource(rc, "train", seed=42, dummy_data=True if args.dummy_data else None)
    dataset_s = time.perf_counter() - t0
    prof = profile.PhaseProfiler()
    with prof.active():
        prof.start()
        it = iter(source.epoch(0))
        with prof.phase("first"):
            batch = next(it)
            jax.block_until_ready(batch.inputs)
        per_batch = []
        for _ in range(min(args.batches, source.num_batches - 1)):
            t = time.perf_counter()
            with prof.phase("data"):
                batch = next(it)
            prof.wait((batch.inputs, batch.targets), "device_put_wait")
            per_batch.append(time.perf_counter() - t)
        prof.stop()
    summary = prof.summary()
    out = {
        "workers": args.workers,
        "dataset_s": dataset_s,
        "first_batch_s": summary["phases"]["first"]["total_s"],
        "batches": len(per_batch),
        "median_ms": 1e3 * statistics.median(per_batch),
        "mean_ms": 1e3 * statistics.fmean(per_batch),
        "p90_ms": 1e3 * sorted(per_batch)[int(0.9 * (len(per_batch) - 1))],
        "phases_ms_per_batch": {
            k: 1e3 * v["excl_s"] / len(per_batch) for k, v in summary["phases"].items() if k != "first" and not k.startswith("first/")
        },
        "device": str(jax.devices()[0]),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("LOADER " + json.dumps(out), flush=True)
    return 0


def run_loader(args, config: Path, root: Path, workers: int) -> dict:
    out = root / f"loader{workers}.json"
    cmd = [args.python, str(Path(__file__).resolve()), "loader", "--config", str(config), "--workers", str(workers)]
    cmd += ["--batches", str(args.loader_batches), "--num-train", str((args.loader_batches + 1) * args.batch), "--out", str(out)]
    if args.tiny or args.dummy_data:
        cmd.append("--dummy-data")
    print("### loader " + " ".join(cmd), flush=True)
    cp = subprocess.run(cmd, env=driver_env(os.environ.get("XLA_FLAGS", "")), cwd=REPO, capture_output=True, text=True, check=False)
    (root / f"loader{workers}.log").write_text(cp.stdout + "\n" + cp.stderr)
    if cp.returncode != 0 or not out.is_file():
        return {"error": "\n".join(cp.stderr.splitlines()[-30:]), "returncode": cp.returncode}
    return json.loads(out.read_text())


def main(args, clock=time.time) -> int:
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.compile_cache:
        args.compile_cache = str(root / "jax_cache")
    if args.tiny:
        args.batch, args.accum = 4, 4
        args.micro, args.num_val, args.loader_batches = min(args.micro, 16), min(args.num_val, 10), min(args.loader_batches, 8)
        args.smoke_ckpt_every, args.smoke_log_every = args.smoke_ckpt_every or 2, args.smoke_log_every or 1
        args.trace_window = "4:8"
        args.pert_log_every = 1  # 4 optimizer steps: 3 log windows after the first
    out = Path(args.out) if args.out else root / "loop_profile.json"
    result: dict = {"args": vars(args), "variants": {}, "skipped": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save() -> None:
        out.write_text(json.dumps(result, indent=1, default=str))

    job_start = clock() if args.job_start is None else args.job_start
    config = make_config(args, root)
    result["config"] = str(config)
    save()
    longest = 0.0
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if not fits_budget(clock() - job_start, longest, args.budget_min):
            result["skipped"].append(name)
            save()
            continue
        t = clock()
        if name.startswith("loader"):
            res = run_loader(args, config, root, int(name[len("loader") :]))
        elif name in FIT_VARIANTS:
            res = run_fit(args, config, root, name)
            if "result" in res:
                res["derived"] = derive(res, args.g3_step_ms)
        else:
            raise SystemExit(f"unknown variant {name!r} (fits: {sorted(FIT_VARIANTS)}, loader<N>)")
        res["elapsed_s"] = clock() - t
        longest = max(longest, res["elapsed_s"])
        result["variants"][name] = res
        save()
        brief = res.get("derived") or {k: res.get(k) for k in ("median_ms", "mean_ms", "first_batch_s", "error")}
        print(f"=== {name}: " + json.dumps(brief, default=str)[:2000], flush=True)
    result["perturbation"] = perturbation(result)
    result["table"] = table(result)
    save()
    print(result["table"], flush=True)
    print(f"LOOP-PROFILE-DONE {out}", flush=True)
    return 0


def fits_budget(elapsed_s: float, longest_s: float, budget_min: float) -> bool:
    """Start the next variant only if it ends within the budget, assuming it takes as long as the longest so far."""
    return elapsed_s + longest_s <= 60 * budget_min


def perturbation(result: dict) -> dict | None:
    """The profiler's cost on the steady loop: median log window after the first of ``pure_log`` over ``pure_noprof``.

    ``valid`` if each run has at least ``MIN_WINDOWS`` windows after the first. ``fit_s_ratio`` (whole fits: compile,
    cache loads and loader start included in both) is kept for reference; it is not the perturbation.
    """
    runs = result["variants"]
    if not all("result" in runs.get(name, {}) for name in ("pure_log", "pure_noprof")):
        return None
    on, off = runs["pure_log"], runs["pure_noprof"]
    w_on, w_off = on.get("log_windows") or {}, off.get("log_windows") or {}
    n = [max(0, w.get("n", 0) - 1) for w in (w_on, w_off)]
    valid = min(n) >= MIN_WINDOWS
    return {
        "pure_log_ms": w_on.get("median_ms_after_first"),
        "pure_noprof_ms": w_off.get("median_ms_after_first"),
        "ratio": w_on["median_ms_after_first"] / w_off["median_ms_after_first"] if valid else None,
        "windows_after_first": n,
        "valid": valid,
        "fit_s_ratio": on["result"]["fit_s"] / off["result"]["fit_s"],
    }


def smoke_timing(smoke_json: str | Path, run_root: str | Path = "") -> dict:
    """Split a 16_jax_smoke run's time per micro-batch into one-time costs and the steady loop (hypothesis H0).

    Per run (A; B resumes at ``resume_step``): ``wall_s`` (the fit process, from 16_jax_smoke), ``fit_s``, ``build_s``
    and ``first_step_s`` (the driver's RESULT), the micro-batches the process trained, and the per-micro-batch rates
    with and without the one-time costs (process start, imports, data sets, model build, first-step compile or cache
    load). ``log_windows``: ``time/ms_per_micro_batch`` of ``steps.csv`` (wall time between log rows over the
    micro-batches between them; the first window holds the first step, later windows hold the step checkpoints and
    the epoch ends, not the log rows themselves).
    """
    j = json.loads(Path(smoke_json).read_text())
    args = j["args"]
    root = Path(run_root or args["run_root"])
    out = {}
    for run in ("run_a", "run_b"):
        res = j.get(run)
        if res is None:
            continue
        micro = res["micro_step"] - (args["resume_step"] * args["accum"] if run == "run_b" else 0)
        wall, fit, first = res["wall_s"], res["fit_s"], res["first_step_s"]
        row = {
            "micro_batches": micro,
            "wall_s": wall,
            "fit_s": fit,
            "build_s": res.get("build_s"),
            "first_step_s": first,
            "outside_fit_s": wall - fit,
            "s_per_micro_wall": wall / micro,
            "s_per_micro_fit": fit / micro,
            "s_per_micro_fit_excl_first_step": (fit - first) / max(1, micro - 1),
            "one_time_s_per_micro": (wall - fit + first) / micro,
            "cache_events": res.get("cache_events"),
        }
        windows = log_windows(root / run / "steps.csv")
        if windows:
            row["log_windows"] = windows
        out[run] = row
    return out


def table(result: dict) -> str:
    lines = [f"{'variant':16s} {'s/micro':>8s} {'excl.compile':>12s} {'steady':>8s} {'over G3 ms':>10s} {'misses':>6s}  top groups (ms/micro)"]
    for name, res in result["variants"].items():
        d = res.get("derived")
        if d is None:
            if "median_ms" in res:
                lines.append(f"{name:16s} loader: {res['median_ms']:.1f} ms/batch median, first batch {res['first_batch_s']:.1f} s")
            else:
                lines.append(f"{name:16s} ERROR {str(res.get('error'))[-200:]}")
            continue
        groups = sorted(d.get("groups_ms_per_micro", {}).items(), key=lambda kv: -kv[1])[:6]
        top = ", ".join(f"{g} {ms:.1f}" for g, ms in groups)
        misses = str(d.get("cache_misses"))
        if d["s_per_micro_steady"] is None:
            lines.append(f"{name:16s} {d['s_per_micro']:8.3f} {'-':>12s} {'-':>8s} {'-':>10s} {misses:>6s}  no profiler: fit_s {d['fit_s']:.1f}")
            continue
        lines.append(
            f"{name:16s} {d['s_per_micro']:8.3f} {d['s_per_micro_excl_compile']:12.3f} {d['s_per_micro_steady']:8.3f} "
            f"{d['overhead_vs_g3_ms']:10.1f} {misses:>6s}  {top}"
        )
    if result.get("skipped"):
        lines.append(f"skipped (budget): {', '.join(result['skipped'])}")
    p = result.get("perturbation")
    if p:
        if p["valid"]:
            lines.append(
                f"profiler perturbation (steady log windows): pure_log {p['pure_log_ms']:.1f} / pure_noprof {p['pure_noprof_ms']:.1f} ms "
                f"per micro-batch = {p['ratio']:.3f} (whole-fit fit_s ratio {p['fit_s_ratio']:.3f}, not the perturbation)"
            )
        else:
            lines.append(f"profiler perturbation: NOT VALID (log windows after the first: {p['windows_after_first']}, need {MIN_WINDOWS} each)")
    return "\n".join(lines)


def report(paths: list[str]) -> dict:
    """Merge the variants of result JSONs (partial ones too: a job killed at the walltime), then table and perturbation."""
    merged: dict = {"variants": {}, "skipped": [], "sources": {}}
    for path in paths:
        r = json.loads(Path(path).read_text())
        for name, res in r.get("variants", {}).items():
            merged["variants"][name] = res
            merged["sources"][name] = str(path)
        merged["skipped"] += [n for n in r.get("skipped", []) if n not in merged["skipped"]]
    merged["skipped"] = [n for n in merged["skipped"] if n not in merged["variants"]]
    merged["perturbation"] = perturbation(merged)
    merged["table"] = table(merged)
    return merged


if __name__ == "__main__":
    a = parse_args()
    if a.command == "smoke-timing":
        print(json.dumps(smoke_timing(a.smoke_json, a.run_root), indent=1))
        sys.exit(0)
    if a.command == "report":
        rep = report(a.results)
        print(rep["table"])
        print(json.dumps(rep["perturbation"], indent=1))
        sys.exit(0)
    sys.exit(loader_only(a) if a.command == "loader" else main(a))
