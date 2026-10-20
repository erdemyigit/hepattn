"""Gate G3 (KERAS_JAX_PLAN.md section 4, S5): the full native train step against the hybrid's in-run step.

Both arms build the org model from the config ``polaris/03_make_config.py`` generates with ``HGQ_PRECISION=32
HGQ_DEVICES=1`` (fp32, one GPU, accumulation 4, PeriodicEBOPs every 8, BetaPID). Each job runs in its own process
(OOM at batch 32 is recorded, not fatal), in ``--jobs`` order, and the JSON is saved after every job. The default order
puts the gate first (``jax-prod:32``, then the hybrid denominators, then ``jax-match`` and ``jax-prod:16``); the
diagnostics ``jax-g1`` and ``jax-precision`` (about 10 more compiled programs at ~2 min each on an A100) go in a second
submission (``17_g3_step.pbs``: ``G3_JOBS``), so the hybrid arms are not cut off by the 1 h walltime.

Jobs (``--job``):
  jax-prod       ``TrainStep(matmul_precision="highest", donate=True)`` with the config's accumulation and EBOPs period:
                 forward, host matching (pure_callback), losses, optimizer, as ``keras_native.loop`` runs it. Times every
                 micro-batch (device synchronised, data already on the device: the hybrid ``StepTimer`` also excludes
                 loading). Reports the EBOPs share of the compiled step (EBOPs-on vs -off micro-batches) and the host
                 matching callback time.
  jax-g1         The G1 gap (S5 open issue 5) at one precision (JAX default, the Keras trainer sets none) and EBOPs on
                 every step: ``TrainStep(None, donate=True)``, ``TrainStep(None, donate=False)``, and Keras'
                 compiled trainer (``model.compile(jit_compile=True)`` + ``train_on_batch``, ``compute_loss`` = the same
                 matched loss + ``model.losses``) on the same model, optimizer settings and batches. If the donating
                 step is slower than ``train_on_batch``, both are traced with ``jax.profiler`` (``--profile-dir``).
  jax-precision  "highest" vs JAX "default" (TF32 on an A100): forward output differences and prediction flips at the
                 initial state, and the loss trajectories of both from the same state over the same batches.
  jax-match      The matching share: the production step (the config's accumulation and EBOPs period) with the host
                 matching, and with an identity assignment and no callback (the costs are still computed and kept
                 live). Shares of the EBOPs-on and -off micro-batches and of the production mix (one on in ``every``).
  hybrid         ``polaris/13_step_time_fit.py`` (Lightning fit of ``main_hgq`` with ``StepTimer``) on the same
                 generated config: the hybrid's in-run micro-batch time, and its peak CUDA memory (printed by the child
                 process at exit).

G3 passes when hybrid median / jax-prod median >= 1.8 at batch 32 and the production matching share is <= 15% of the
step. If the hybrid has no batch 32 result (G1 and G2 saw it OOM there), the verdict decides at batch 16, records the
substitution, and also reports the per-event comparison jax b32 vs hybrid b16 (ms / event).

Local check at tiny sizes (CPU, dummy events): ``python polaris/17_g3_step.py --sweep --tiny --out g3_local.json``.
"""
# ruff: noqa: PLC0415, SLF001
# PLC0415: jax / keras / torch import only inside a job, after the backend is fixed.
# SLF001: TrainStep internals (_precision) for the matching-share variant.

import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jax_common import REPO, generate_config, median_ms, quantiles_ms, shrink_config

JOBS = ("jax-prod", "jax-g1", "jax-precision", "jax-match", "hybrid")
GATE_JOBS = "jax-prod:32,hybrid:32,hybrid:16,jax-match:32,jax-prod:16"
DIAGNOSTIC_JOBS = "jax-g1:32,jax-precision:32"
DEFAULT_JOBS = f"{GATE_JOBS},{DIAGNOSTIC_JOBS}"
TINY = {"dim": 16, "heads": 2, "enc_layers": 1, "dec_layers": 1, "linformer_k": 8, "nodes": 64, "queries": 16}
G3_RATIO, MATCH_SHARE = 1.8, 0.15


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", choices=JOBS)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--config", default="", help="generated config (empty: run 03_make_config.py)")
    p.add_argument("--data-file", default="", help="CLIC ROOT file for the JAX jobs (empty: dummy events)")
    p.add_argument("--data-dir", default="", help="DATA_DIR for the generated config (hybrid arm)")
    p.add_argument("--tiny", action="store_true", help="tiny sizes, dummy events, CPU (local check of both arms)")
    p.add_argument("--warmup", type=int, default=8, help="JAX micro-batches before timing (covers both EBOPs programs)")
    p.add_argument("--iters", type=int, default=128, help="timed JAX micro-batches")
    p.add_argument("--distinct", type=int, default=4, help="distinct batches cycled by the JAX jobs")
    p.add_argument("--precision-steps", type=int, default=20)
    p.add_argument("--hybrid-batches", type=int, default=121, help="hybrid micro-batches (StepTimer drops STEP_WARMUP)")
    p.add_argument("--hybrid-warmup", type=int, default=25)
    p.add_argument("--profile-dir", default="")
    p.add_argument("--seed", type=int, default=0)
    # sweep
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--jobs", default=DEFAULT_JOBS)
    p.add_argument("--jax-python", default=sys.executable)
    p.add_argument("--torch-python", default=sys.executable)
    p.add_argument("--out", default="")
    p.add_argument("--timeout", type=int, default=2400)
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------------- config
def make_config(args, out_dir: Path, batch: int) -> Path:
    """The generated config at ``batch`` (fp32, one device); at tiny sizes for ``--tiny``."""
    import yaml

    # the hybrid arm's ModelCheckpoint / MetricsCsvWriter write under CKPT_DIR: keep them next to the results
    env = {"HGQ_PRECISION": "32", "HGQ_DEVICES": "1", "HGQ_BATCH": str(batch), "CKPT_DIR": str(out_dir / f"hybrid_ckpts_b{batch}")}
    if args.data_dir:
        env["DATA_DIR"] = args.data_dir
    path = generate_config(out_dir / f"g3_b{batch}.yaml", **env)
    if args.tiny:
        cfg = shrink_config(yaml.safe_load(path.read_text()), **TINY)
        cfg["data"]["num_workers"] = 0
        cfg["trainer"]["accelerator"] = "cpu"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --------------------------------------------------------------------------------------------- sweep
def run_sweep(args) -> None:
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp()) / "g3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    def save() -> None:
        out.write_text(json.dumps({"args": vars(args), "results": results, "verdict": verdict(results)}, indent=1, default=str))

    plan = [(job, int(b)) for job, b in (item.split(":") for item in args.jobs.split(","))]
    for job, batch in plan:
        config = Path(args.config) if args.config else make_config(args, out.parent, batch)
        if args.config:  # one given config: rewrite its batch size
            import yaml

            cfg = yaml.safe_load(config.read_text())
            cfg["data"]["batch_size"] = batch
            config = out.parent / f"g3_b{batch}_given.yaml"
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        py = args.torch_python if job == "hybrid" else args.jax_python
        backend = "torch" if job == "hybrid" else "jax"
        env = {**os.environ, "KERAS_BACKEND": backend, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        passthrough = [
            "--data-file",
            args.data_file,
            "--warmup",
            args.warmup,
            "--iters",
            args.iters,
            "--distinct",
            args.distinct,
            "--precision-steps",
            args.precision_steps,
            "--hybrid-batches",
            args.hybrid_batches,
            "--hybrid-warmup",
            args.hybrid_warmup,
            "--profile-dir",
            args.profile_dir,
            "--seed",
            args.seed,
        ]
        cmd = [py, str(Path(__file__).resolve()), "--job", job, "--batch", str(batch), "--config", str(config), *map(str, passthrough)]
        if args.tiny:
            cmd.append("--tiny")
        print(f"### {job} batch={batch}", flush=True)
        t0 = time.perf_counter()
        try:
            cp = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            results.append({"job": job, "batch": batch, "error": f"timeout {args.timeout}s"})
            save()
            continue
        line = next((ln for ln in reversed(cp.stdout.splitlines()) if ln.startswith("RESULT ")), None)
        if cp.returncode != 0 or line is None:
            tail = (cp.stderr or cp.stdout).strip().splitlines()[-20:]
            oom = any(re.search(r"out of memory|RESOURCE_EXHAUSTED|OutOfMemory", ln, re.IGNORECASE) for ln in tail)
            print("\n".join(tail), flush=True)
            results.append({"job": job, "batch": batch, "error": "\n".join(tail), "oom": oom})
        else:
            res = json.loads(line[len("RESULT ") :])
            res["wall_s"] = time.perf_counter() - t0
            print(json.dumps(res, indent=1, default=str), flush=True)
            results.append(res)
        save()
    print(json.dumps(verdict(results), indent=1), flush=True)
    print(f"G3 results: {out}", flush=True)


def _get(results, job, batch):
    return next((r for r in results if r.get("job") == job and r.get("batch") == batch and "error" not in r), None)


def verdict(results) -> dict:
    """The G3 ratios and gates from the results so far (None where a job is missing)."""
    out: dict = {"ratio_threshold": G3_RATIO, "matching_share_threshold": MATCH_SHARE}
    for batch in (32, 16):
        prod, hyb = _get(results, "jax-prod", batch), _get(results, "hybrid", batch)
        if prod and hyb:
            out[f"b{batch}"] = {
                "hybrid_median_ms": hyb["median_ms"],
                "jax_median_ms": prod["median_ms"],
                "ratio_median": hyb["median_ms"] / prod["median_ms"],
                "hybrid_mean_ms": hyb["mean_ms"],
                "jax_mean_ms": prod["mean_ms"],
                "ratio_mean": hyb["mean_ms"] / prod["mean_ms"],
            }
    decided = out.get("b32") or out.get("b16")
    out["decided_at"] = 32 if "b32" in out else (16 if "b16" in out else None)
    out["g3_speed_pass"] = None if decided is None else decided["ratio_median"] >= G3_RATIO
    if out["decided_at"] == 16:
        out["substitution"] = "no hybrid batch 32 result: decided at batch 16 (the plan's criterion is at batch 32)"
    prod32, hyb32, hyb16 = _get(results, "jax-prod", 32), _get(results, "hybrid", 32), _get(results, "hybrid", 16)
    if prod32 and hyb16 and not hyb32:
        # the throughput question when the hybrid cannot run batch 32: each arm at its largest batch, per event
        out["per_event_jax_b32_vs_hybrid_b16"] = {
            "jax_ms_per_event": prod32["median_ms"] / 32,
            "hybrid_ms_per_event": hyb16["median_ms"] / 16,
            "ratio": (hyb16["median_ms"] / 16) / (prod32["median_ms"] / 32),
        }
    match = _get(results, "jax-match", 32) or _get(results, "jax-match", 16)
    out["matching_share"] = None if match is None else match["matching_share"]
    out["matching_share_ebops_on_off"] = None if match is None else (match.get("matching_share_ebops_on"), match.get("matching_share_ebops_off"))
    out["matching_pass"] = None if match is None else match["matching_share"] <= MATCH_SHARE
    g1 = _get(results, "jax-g1", 32)
    if g1:
        out["g1"] = {k: g1[k] for k in ("none_donate_ms", "none_nodonate_ms", "keras_trainer_ms", "donate_vs_keras")}
    return out


# --------------------------------------------------------------------------------------------- JAX jobs
def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20


def peak_dev_gb():
    import jax

    dev = jax.devices()[0]
    stats = dev.memory_stats() if dev.platform == "gpu" else None
    return stats["peak_bytes_in_use"] / 2**30 if stats else None


def load_batches(args, rc):
    """``args.distinct`` batches of ``args.batch`` events (ROOT file or dummy) as device ``Batch`` objects."""
    import dataclasses

    from hepattn.experiments.clic.native_data import make_dataset, make_loader
    from hepattn.keras_native.data import HostBatches

    n = args.batch * args.distinct
    data = dataclasses.replace(rc.data, batch_size=args.batch, num_workers=0)
    if args.data_file:
        data = dataclasses.replace(data, train_path=args.data_file)
    dataset = make_dataset(data, "train", dummy_data=not args.data_file, num_events=n)
    loader = make_loader(dataset, "val", batch_size=args.batch)  # in order, no shuffling
    return list(HostBatches(loader, args.batch, pad_to_batch=False))


def build(args, rc):
    """The model (eager build on the first batch) and the config's optimizer; one per job, shared by its steps."""
    import keras

    from hepattn.keras_native.loop import build_run_optimizer

    keras.utils.set_random_seed(args.seed)
    model = rc.build_model()
    model(args.batches[0].inputs, training=False)
    return model, build_run_optimizer(rc, model, total_steps=100_000)


def make_steps(rc, model, optimizer, *, matmul_precision="highest", donate=True, loss_fn=None, host_fn=None):
    """A ``TrainStep`` (its own jitted programs) on a shared model and optimizer; each ``init_state`` is a fresh copy."""
    from hepattn.keras_native.train_step import TrainStep, maskformer_loss

    loss_fn = loss_fn or maskformer_loss(rc.task_specs, n_jobs=rc.matcher_n_jobs, host_fn=host_fn)
    return TrainStep(model, optimizer, loss_fn, matmul_precision=matmul_precision, donate=donate)


def beta_of(rc) -> float:
    """The beta the config's first epoch runs at (BetaPID init_beta, or the ramp start), > 0 for the EBOPs term."""
    if rc.quant.beta_pid:
        return rc.quant.beta_pid["init_beta"]
    if rc.quant.beta_scheduler:
        return max(rc.quant.beta_scheduler["beta_start"], 1e-12)
    return 1e-12


def time_steps(steps, state, batches, *, n_warm, n_iter, ebops_fn):
    """Synchronised wall time of each micro-batch after ``n_warm``; returns (state, times_s, ebops flags, last out)."""
    import jax

    times, flags, out = [], [], None
    for i in range(n_warm + n_iter):
        b = batches[i % len(batches)]
        on = ebops_fn(i)
        t0 = time.perf_counter()
        state, out = steps.train_step(state, (b.inputs, b.targets), ebops_on=on)
        jax.block_until_ready((out.loss, state.tv[0], state.ov[0]))
        if i >= n_warm:
            times.append(time.perf_counter() - t0)
            flags.append(on)
    return state, times, flags, out


def summary(times) -> dict:
    p10, p90 = quantiles_ms(times)
    return {"median_ms": median_ms(times), "mean_ms": statistics.mean(times) * 1e3, "p10_ms": p10, "p90_ms": p90, "n": len(times)}


def job_prod(args, rc) -> dict:
    """The production step: the config's accumulation k and EBOPs period, "highest", donation."""
    from hepattn.keras_native.matching import host_match

    calls = []

    def timed_match(costs, valid):
        t0 = time.perf_counter()
        idx = host_match(costs, valid, n_jobs=rc.matcher_n_jobs)
        calls.append(time.perf_counter() - t0)
        return idx

    steps = make_steps(rc, *build(args, rc), host_fn=timed_match)
    k, every = rc.optimizer.accumulate_grad_batches, rc.quant.ebops_every_n_steps
    state = steps.set_beta(steps.init_state(), beta_of(rc))
    t0 = time.perf_counter()
    state, times, flags, out = time_steps(steps, state, args.batches, n_warm=args.warmup, n_iter=args.iters, ebops_fn=lambda i: (i // k) % every == 0)
    on = [t for t, f in zip(times, flags, strict=True) if f]
    off = [t for t, f in zip(times, flags, strict=True) if not f]
    res = {**summary(times), "accumulate": k, "ebops_every": every, "matmul_precision": "highest", "donate": True}
    res["ebops_on"], res["ebops_off"] = summary(on) if on else None, summary(off) if off else None
    if on and off:
        res["ebops_share_of_on_step"] = 1 - median_ms(off) / median_ms(on)
        res["mean_ms_every1_vs_every8"] = (median_ms(on), (median_ms(on) + (every - 1) * median_ms(off)) / every)
    timed_calls = calls[-len(times) :]
    res["match_callback_median_ms"] = median_ms(timed_calls) if timed_calls else None
    res["match_callback_share_of_median"] = res["match_callback_median_ms"] / res["median_ms"] if timed_calls else None
    res["loss_last"] = float(out.loss)
    res["first_steps_s"] = time.perf_counter() - t0 - sum(times)
    res["compile_counts"] = steps.compile_counts()
    return res


def job_g1(args, rc) -> dict:
    """``TrainStep(None, donate)``, ``TrainStep(None, no donation)``, Keras ``train_on_batch``: EBOPs every step."""
    from keras import ops

    from hepattn.keras_native.objective import matched_loss

    model, optimizer = build(args, rc)
    res = {}
    for name, donate in (("none_donate", True), ("none_nodonate", False)):
        steps = make_steps(rc, model, optimizer, matmul_precision=None, donate=donate)
        state = steps.set_beta(steps.init_state(), beta_of(rc))
        state, times, _, out = time_steps(steps, state, args.batches, n_warm=args.warmup, n_iter=args.iters, ebops_fn=lambda _i: True)
        res[name] = {**summary(times), "loss_last": float(out.loss)}
        res[f"{name}_ms"] = res[name]["median_ms"]
        del steps, state, out

    # Keras' trainer on the same model and optimizer (last: it writes their variables)
    leaves = make_steps(rc, model, optimizer, matmul_precision=None).ebops_leaves.leaves
    for leaf in leaves:
        leaf._beta.assign(beta_of(rc))
    specs, n_jobs = rc.task_specs, rc.matcher_n_jobs

    def compute_loss(x=None, y=None, y_pred=None, sample_weight=None, training=True):  # noqa: ARG001
        task, _, _ = matched_loss(y_pred, y, specs, n_jobs=n_jobs)
        return task + ops.sum(ops.stack(model.losses))  # EBOPs terms + regularizers of this forward (stateless scope)

    model.compute_loss = compute_loss
    model.compile(optimizer=optimizer, jit_compile=True)
    times, logs = [], {}
    for i in range(args.warmup + args.iters):
        b = args.batches[i % len(args.batches)]
        t0 = time.perf_counter()
        logs = model.train_on_batch(b.inputs, b.targets, return_dict=True)
        float(logs["loss"])  # train_on_batch already synchronises (jax_state_sync); this reads the value
        if i >= args.warmup:
            times.append(time.perf_counter() - t0)
    res["keras_trainer"] = {**summary(times), "loss_last": float(logs["loss"])}
    res["keras_trainer_ms"] = res["keras_trainer"]["median_ms"]
    res["donate_vs_keras"] = res["none_donate_ms"] / res["keras_trainer_ms"]
    if args.profile_dir and res["donate_vs_keras"] > 1.0:
        res["profile"] = profile_both(args, rc, model, optimizer)
    return res


def profile_both(args, rc, model, optimizer) -> str:
    """``jax.profiler`` traces of 5 donating TrainStep steps and 5 ``train_on_batch`` calls (S5 open issue 5)."""
    import jax

    out = Path(args.profile_dir) / f"g1_b{args.batch}"
    steps = make_steps(rc, model, optimizer, matmul_precision=None, donate=True)
    state = steps.set_beta(steps.init_state(), beta_of(rc))
    state, *_ = time_steps(steps, state, args.batches, n_warm=2, n_iter=1, ebops_fn=lambda _i: True)
    with jax.profiler.trace(str(out / "trainstep")):
        time_steps(steps, state, args.batches, n_warm=0, n_iter=5, ebops_fn=lambda _i: True)
    with jax.profiler.trace(str(out / "keras_trainer")):
        for i in range(5):
            float(
                model.train_on_batch(args.batches[i % len(args.batches)].inputs, args.batches[i % len(args.batches)].targets, return_dict=True)[
                    "loss"
                ]
            )
    return str(out)


def job_precision(args, rc) -> dict:
    """Matmul precision "highest" vs "default" from one initial state: outputs, prediction flips, loss trajectories, step time."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from hepattn.keras_native.predict_writer import predict
    from hepattn.keras_native.train_step import TrainState

    k, every = rc.optimizer.accumulate_grad_batches, rc.quant.ebops_every_n_steps
    model, optimizer = build(args, rc)
    hi = make_steps(rc, model, optimizer, matmul_precision="highest")
    lo = make_steps(rc, model, optimizer, matmul_precision=None)
    host0 = [[np.array(x, copy=True) for x in group] for group in hi.set_beta(hi.init_state(), beta_of(rc))]

    def fresh() -> TrainState:
        return TrainState(*([jnp.asarray(x) for x in group] for group in host0))

    b = args.batches[0]
    o_hi, o_lo = hi.predict_step(fresh(), b.inputs), lo.predict_step(fresh(), b.inputs)
    pairs = [(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree_util.tree_leaves(o_hi), jax.tree_util.tree_leaves(o_lo), strict=True)]
    floats = [(x, y) for x, y in pairs if np.issubdtype(x.dtype, np.floating)]
    max_abs = max(float(np.max(np.abs(np.where(np.isfinite(x) & np.isfinite(y), x - y, 0.0)))) for x, y in floats)
    p_hi = [np.asarray(x) for x in jax.tree_util.tree_leaves(predict(o_hi, model.tasks))]
    p_lo = [np.asarray(x) for x in jax.tree_util.tree_leaves(predict(o_lo, model.tasks))]
    flips = sum(int(np.sum(x != y)) for x, y in zip(p_hi, p_lo, strict=True) if x.dtype == np.bool_)
    res = {"forward_max_abs_diff": max_abs, "prediction_flips": flips, "prediction_bools": sum(x.size for x in p_hi if x.dtype == np.bool_)}
    losses = {}
    for name, steps in (("highest", hi), ("default", lo)):
        state, times, losses[name] = fresh(), [], []
        for i in range(args.precision_steps):
            bb = args.batches[i % len(args.batches)]
            t0 = time.perf_counter()
            state, out = steps.train_step(state, (bb.inputs, bb.targets), ebops_on=(i // k) % every == 0)
            losses[name].append(float(out.loss))  # synchronises
            times.append(time.perf_counter() - t0)
        res[f"{name}_step"] = summary(times[k + 1 :])  # after both programs compiled (micro 0: on, micro k: off)
    rel = [abs(a - c) / abs(a) for a, c in zip(losses["highest"], losses["default"], strict=True)]
    res.update(loss_max_rel_diff=max(rel), loss_rel_diff_last=rel[-1], losses_highest=losses["highest"], losses_default=losses["default"])
    return res


def job_match(args, rc) -> dict:
    """Matching share = 1 - t(no callback) / t(host matching), both "highest", at the config's EBOPs period.

    ``matching_share`` (the gate) is of the production mix: medians of the EBOPs-on and -off micro-batches weighted
    1 : (every - 1), as ``loop`` runs them (all micro-batches of every ``every``-th optimizer step carry the term).
    """
    import jax
    import jax.numpy as jnp

    from hepattn.keras_native.objective import model_costs, model_losses, permute_outputs, total_loss

    specs = rc.task_specs

    def no_match(outputs, targets):
        costs = model_costs(outputs, targets, specs)
        live = jax.lax.stop_gradient(sum(jnp.sum(jnp.where(jnp.isfinite(c), c, 0.0)) for c in costs.values()))
        idx = {}
        for layer, c in costs.items():
            batch, num = c.shape[0], c.shape[1]
            # x * 0 is not folded for floats (NaN / inf), so the costs stay computed, as in the matched step
            idx[layer] = jnp.broadcast_to(jnp.arange(num, dtype=jnp.int32), (batch, num)) + (live * 0).astype(jnp.int32)
        losses = model_losses(permute_outputs(outputs, idx, specs), targets, specs)
        return total_loss(losses), losses

    k, every = rc.optimizer.accumulate_grad_batches, rc.quant.ebops_every_n_steps
    model, optimizer = build(args, rc)
    res, med = {"accumulate": k, "ebops_every": every}, {}
    for name, loss_fn in (("host_matching", None), ("no_callback", no_match)):
        steps = make_steps(rc, model, optimizer, loss_fn=loss_fn)
        state = steps.set_beta(steps.init_state(), beta_of(rc))
        state, times, flags, _ = time_steps(
            steps, state, args.batches, n_warm=args.warmup, n_iter=args.iters, ebops_fn=lambda i: (i // k) % every == 0
        )
        on = [t for t, f in zip(times, flags, strict=True) if f]
        off = [t for t, f in zip(times, flags, strict=True) if not f]
        res[name] = {**summary(times), "ebops_on": summary(on) if on else None, "ebops_off": summary(off) if off else None}
        med[name] = (median_ms(on) if on else None, median_ms(off) if off else None)
        del steps, state
    (m_on, m_off), (n_on, n_off) = med["host_matching"], med["no_callback"]
    res["matching_share_ebops_on"] = None if m_on is None else 1 - n_on / m_on
    res["matching_share_ebops_off"] = None if m_off is None else 1 - n_off / m_off
    if m_on is not None and m_off is not None:
        res["matching_share"] = 1 - (n_on + (every - 1) * n_off) / (m_on + (every - 1) * m_off)
    else:  # every == 1, or too few iterations for both kinds: the plain medians
        res["matching_share"] = 1 - res["no_callback"]["median_ms"] / res["host_matching"]["median_ms"]
    return res


# --------------------------------------------------------------------------------------------- hybrid job
STEP_RE = re.compile(r"\[STEP-TIME rank=0\] batches=(\d+) warmup=(\d+) median=([\d.]+)s mean=([\d.]+)s p10=([\d.]+)s first=([\d.]+)s")
PEAK_RE = re.compile(r"^\[G3-PEAK-CUDA-BYTES\] (-?\d+)$")
# runs 13_step_time_fit.py as __main__ and prints the child's own peak CUDA memory at exit (-1 without CUDA)
HYBRID_CHILD = (
    "import atexit, runpy, sys, torch\n"
    "atexit.register(lambda: print(f'[G3-PEAK-CUDA-BYTES] {torch.cuda.max_memory_allocated() if torch.cuda.is_available() else -1}', flush=True))\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def job_hybrid(args) -> dict:
    """``13_step_time_fit.py fit`` on the generated config (one device, fp32, the config's accumulation and EBOPs period).

    Raises:
        SystemExit: If the run fails or prints no ``STEP-TIME`` line.
    """
    cmd = [
        sys.executable,
        "-c",
        HYBRID_CHILD,
        str(REPO / "polaris/13_step_time_fit.py"),
        "fit",
        "--config",
        args.config,
        "--trainer.devices",
        "1",
        "--trainer.limit_train_batches",
        str(args.hybrid_batches),
        "--trainer.limit_val_batches",
        "1",
        "--trainer.max_epochs",
        "1",
        "--data.num_train",
        str(args.hybrid_batches * args.batch),
        "--data.num_val",
        str(2 * args.batch),
        "--trainer.default_root_dir",
        str(Path(tempfile.mkdtemp()) / "g3_hybrid"),
    ]
    if args.tiny:
        cmd += ["--data.dummy_data", "true", "--trainer.accelerator", "cpu", "--data.num_workers", "0"]
    env = {**os.environ, "STEP_WARMUP": str(args.hybrid_warmup), "KERAS_BACKEND": "torch"}
    cp = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    match = next((STEP_RE.search(ln) for ln in cp.stdout.splitlines() if STEP_RE.search(ln)), None)
    if cp.returncode != 0 or match is None:
        raise SystemExit("hybrid run failed:\n" + "\n".join((cp.stderr or cp.stdout).splitlines()[-30:]))
    batches, warmup, median, mean, p10, first = match.groups()
    res = {
        "batches": int(batches),
        "warmup": int(warmup),
        "median_ms": float(median) * 1e3,
        "mean_ms": float(mean) * 1e3,
        "p10_ms": float(p10) * 1e3,
        "first_s": float(first),
    }
    peak = next((int(m.group(1)) for ln in reversed(cp.stdout.splitlines()) if (m := PEAK_RE.match(ln))), None)
    res["peak_dev_gb"] = None if peak is None or peak < 0 else peak / 2**30
    res["peak_line_seen"] = peak is not None  # the child printed its line (-1 without CUDA)
    return res


# --------------------------------------------------------------------------------------------- one job
def run_job(args) -> None:
    info = {"job": args.job, "batch": args.batch, "precision": "fp32", "config": args.config, "tiny": args.tiny}
    if args.job == "hybrid":
        info.update(job_hybrid(args))
        print("RESULT " + json.dumps(info, default=str), flush=True)
        return
    if os.environ.get("KERAS_BACKEND") != "jax":
        raise SystemExit(f"--job {args.job} needs KERAS_BACKEND=jax")
    import jax
    import keras

    import hepattn
    from hepattn.keras_native.config import load_config

    rc = load_config(args.config)
    if rc.trainer.get("precision") not in {None, 32, "32", "32-true"}:
        raise SystemExit(f"config precision {rc.trainer.get('precision')!r}: G3 compares fp32")
    args.batches = load_batches(args, rc)
    info.update(
        jax=jax.__version__,
        keras=keras.__version__,
        device=str(jax.devices()[0]),
        hepattn_file=hepattn.__file__,
        data=args.data_file or "dummy",
        accumulate=rc.optimizer.accumulate_grad_batches,
        ebops_every=rc.quant.ebops_every_n_steps,
    )
    job = {"jax-prod": job_prod, "jax-g1": job_g1, "jax-precision": job_precision, "jax-match": job_match}[args.job]
    info.update(job(args, rc))
    info.update(peak_rss_gb=peak_rss_gb(), peak_dev_gb=peak_dev_gb())
    print("RESULT " + json.dumps(info, default=str), flush=True)


if __name__ == "__main__":
    a = parse_args()
    if a.sweep:
        run_sweep(a)
    elif a.job:
        run_job(a)
    else:
        raise SystemExit("pass --job or --sweep")
