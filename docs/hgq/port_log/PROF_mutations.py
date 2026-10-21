"""PROF mutation runner: each mutant must make one of its tests fail (tests/keras_native/test_profile.py,
tests/polaris/test_loop_profile_script.py).

Same scheme as ``S7_mutations.py``: the worktree is copied (rsync, without .git and the venvs) to ``<scratch>/prof_mut``;
each mutation replaces one snippet (which must occur exactly once) in the copy, runs the named tests there with ``-x``
(``PYTHONPATH=<copy>/src``, so the test conftest checks that hepattn resolves to the copy), and restores the file. The
worktree is never written. The baseline (``p00``) runs the unmutated copy and must pass.

    python docs/hgq/port_log/PROF_mutations.py <scratch> [OUT.json] [--only p01,p02]

Python: ``$PROF_JAX_PYTHON``, else ``<repo>/.venv-jax``, else ``../hepattn-keras-jax/.venv-jax`` (JAX backend).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROF = "src/hepattn/keras_native/profile.py"
LOOP = "src/hepattn/keras_native/loop.py"
DATA = "src/hepattn/keras_native/data.py"
CKPT = "src/hepattn/keras_native/checkpoint.py"
CLI = "src/hepattn/keras_native/cli.py"
TP = "tests/keras_native/test_profile.py"
LP = "polaris/18_loop_profile.py"
TL = "tests/polaris/test_loop_profile_script.py"


def python() -> Path:
    for p in (os.environ.get("PROF_JAX_PYTHON"), REPO / ".venv-jax/bin/python", REPO.parent / "hepattn-keras-jax/.venv-jax/bin/python"):
        if p and Path(p).is_file():
            return Path(p)
    raise SystemExit("no JAX python: set PROF_JAX_PYTHON")


MUTATIONS = [
    # (name, file, old, new, tests, what the mutant breaks)
    ("p00", None, "", "", [TP], "baseline: the unmutated copy passes"),
    ("p00b", None, "", "", [TL], "baseline: the unmutated copy passes the script tests"),
    # profiler-off must add nothing: NULL.wait blocks -> the off run gains syncs
    (
        "p01",
        PROF,
        '    def wait(self, tree, name: str = "device_wait"):\n        return tree',
        '    def wait(self, tree, name: str = "device_wait"):\n        return jax.block_until_ready(tree)',
        [TP, "-k", "syncs_added"],
        "NULL.wait syncs (profiler off adds a block per epoch end)",
    ),
    # a profiler that perturbs the run (the matching callback sees reordered costs) -> async state != off state
    (
        "p02",
        PROF,
        "                return fn(*args, **kwargs)",
        "                return fn(args[0][::-1], *args[1:], **kwargs)",
        [TP, "-k", "do_not_change_the_run"],
        "profiled matching callback changes the matching",
    ),
    # prefetch order (LIFO) -> order test and the bitwise run comparison
    (
        "p03",
        DATA,
        "    items: queue.Queue = queue.Queue(maxsize=depth)",
        "    items: queue.Queue = queue.LifoQueue(maxsize=depth)",
        [TP, "-k", "prefetch_order"],
        "prefetch reorders items",
    ),
    # prefetch swallows the producer's exception (ends the sequence instead)
    ("p04", DATA, "            put(_Raised(error))", "            put(_END)", [TP, "-k", "prefetch_order"], "prefetch drops the error"),
    # prefetch close does not wait for the thread
    (
        "p05",
        DATA,
        "        while thread.is_alive():  # the thread",
        "        while False and thread.is_alive():  # the thread",
        [TP, "-k", "prefetch_order"],
        "prefetch close leaves the thread running",
    ),
    # exclusive time: children not subtracted
    ("p06", PROF, "            stack[-1].child += dur", "            pass", [TP, "-k", "fake_clock"], "exclusive time = duration"),
    # summary attributes total instead of exclusive time -> the phases over-count the wall
    (
        "p07",
        PROF,
        "            excl = sum(r[4] for r in rows)",
        "            excl = sum(r[3] for r in rows)",
        [TP, "-k", "fake_clock or sum_to_the_wall"],
        "summary sums durations",
    ),
    # current() ignores the thread -> other threads would record phases
    (
        "p08",
        PROF,
        "    if prof is NULL or threading.get_ident() != prof.thread:",
        "    if prof is NULL:",
        [TP, "-k", "null_profiler"],
        "other threads record phases",
    ),
    # async mode blocks on every step (block -> wait)
    (
        "p09",
        LOOP,
        '            prof.block(out.loss, "train/device")',
        '            prof.wait(out.loss, "train/device")',
        [TP, "-k", "syncs_added or cadence"],
        "async mode syncs every step",
    ),
    # compile relabelling lost -> compile time counted as steady train/step
    ("p10", LOOP, "            self.trainer.prof.mark_compile()", "            pass", [TP, "-k", "cadence"], "+compile label lost"),
    # the step checkpoint phase not opened (its sub-phases become top level)
    (
        "p11",
        LOOP,
        '                            with prof.phase("train/ckpt"):',
        '                            with profile.NULL.phase("train/ckpt"):',
        [TP, "-k", "cadence"],
        "train/ckpt phase missing",
    ),
    # the data phase not opened: the loader wait is attributed elsewhere
    (
        "p12",
        LOOP,
        '                with self.prof.phase("data"):',
        '                with profile.NULL.phase("data"):',
        [TP, "-k", "cadence"],
        "data phase missing",
    ),
    # the async drain wait dropped (the epoch-end block moves out of its phase)
    (
        "p13",
        LOOP,
        '                prof.wait(self.state, "drain")',
        "                pass",
        [TP, "-k", "syncs_added"],
        "epoch_end/drain wait missing",
    ),
    # checkpoint npz write not timed
    (
        "p14",
        CKPT,
        '    with prof.phase("npz_write"), _atomic(path) as f:',
        "    with _atomic(path) as f:",
        [TP, "-k", "cadence"],
        "ckpt npz_write phase missing",
    ),
    # sync mode does not wait for device transfers
    (
        "p15",
        DATA,
        '            prof.block((inputs, targets), "device_put_wait")',
        "            pass",
        [TP, "-k", "syncs_added"],
        "sync mode misses transfer waits",
    ),
    # CLI: --profile-phases ignored / --prefetch ignored
    ("p16", CLI, "if args.profile_phases else None", "if False else None", [TP, "-k", "fit_command"], "CLI ignores --profile-phases"),
    ("p17", CLI, "        prefetch=args.prefetch,", "        prefetch=0,", [TP, "-k", "fit_command"], "CLI ignores --prefetch"),
    # compile phases counted in their steady-state groups (the attribution table would mix compiles into them)
    (
        "p18",
        PROF,
        '            groups[COMPILE_GROUP if is_compile(path) else group_of(path)] += p["excl_s"]',
        '            groups[group_of(path)] += p["excl_s"]',
        [TP, "-k", "sum_to_the_wall"],
        "compile phases not grouped",
    ),
    # 18_loop_profile arithmetic: H0 split of a smoke result, the unprofiled variant, the perturbation ratio
    (
        "p19",
        LP,
        '        micro = res["micro_step"] - (args["resume_step"] * args["accum"] if run == "run_b" else 0)',
        '        micro = res["micro_step"]',
        [TL],
        "run B counted from step 0",
    ),
    (
        "p20",
        LP,
        '            "one_time_s_per_micro": (wall - fit + first) / micro,',
        '            "one_time_s_per_micro": (wall - fit) / micro,',
        [TL],
        "first-step compile not a one-time cost",
    ),
    (
        "p21",
        LP,
        "        per = steady = None\n",
        '        per, steady = (summary["fit_s"] - summary["first_step_s"]) / max(1, micro - 1), None\n',  # review 2: new snippet
        [TL],
        "unprofiled variant gets a compile-including 'excl' rate",
    ),
    (
        "p22",
        LP,
        '"ratio": w_on["median_ms_after_first"] / w_off["median_ms_after_first"] if valid else None',
        '"ratio": w_off["median_ms_after_first"] / w_on["median_ms_after_first"] if valid else None',
        [TL],
        "perturbation ratio inverted",
    ),
    ("p23", LP, '        "valid": valid,', '        "valid": True,', [TL], "perturbation valid with too few log windows"),
    # p21-p23 were rewritten in the review fix (2026-10-20): the perturbation is now the ratio of steady log windows
    ("p00c", None, "", "", [TP, "-k", "fake_clock or cli_profiler"], "baseline: the fast profile tests pass"),
    # review fix 1 (2026-10-20): the derived numbers of summary(), checked on a fake clock (test_derived_numbers_*)
    ("m01", PROF, '"forward", "tail", ', "", [TP, "-k", "fake_clock"], "validation / log forward and tail left out of busy"),
    ("m02", PROF, "if e[6] > first]", "if e[6] >= first]", [TP, "-k", "fake_clock"], "steady medians include the first optimizer step"),
    (
        "m03",
        PROF,
        'out["device_idle_frac_est"] = idle(steady_micro * step_ref_s, steady_wall)',
        'out["device_idle_frac_est"] = idle(step_ref_s, steady_wall)',  # review 2: idle() helper
        [TP, "-k", "fake_clock"],
        "idle estimate without the micro-batch factor",
    ),
    (
        "m05",
        PROF,
        '"s_per_micro_batch_excl_compile"] = (wall - compile_s) / micro',
        '"s_per_micro_batch_excl_compile"] = wall / micro',
        [TP, "-k", "fake_clock"],
        "compile not subtracted",
    ),
    (
        "m06",
        PROF,
        'return path.endswith(COMPILE_SUFFIX) or path == "train/first_step_wait"',
        "return path.endswith(COMPILE_SUFFIX)",
        [TP, "-k", "fake_clock"],
        "first-step wait not a compile",
    ),
    (
        "m07",
        PROF,
        'out["device_idle_frac"] = idle(busy, steady_wall)',
        'out["device_idle_frac"] = idle(busy, wall)',  # review 2: idle() helper
        [TP, "-k", "fake_clock"],
        "idle fraction over the whole wall (one-time costs in it)",
    ),
    (
        "m08",
        PROF,
        "steady_wall = wall - compile_s - first_train - first_val",
        "steady_wall = wall - compile_s - first_train",
        [TP, "-k", "fake_clock"],
        "validation loader start counted as steady",
    ),
    (
        "m09",
        PROF,
        'steady_micro = micro - len(by_path.get("train/first_step_wait", ()))',
        "steady_micro = micro",
        [TP, "-k", "fake_clock"],
        "steady count includes the first micro-batch",
    ),
    (
        "m10",
        PROF,
        'if path.rsplit("/", 1)[-1] in DEVICE_PHASES',
        'if path.removesuffix(COMPILE_SUFFIX).rsplit("/", 1)[-1] in DEVICE_PHASES',
        [TP, "-k", "fake_clock"],
        "compile phases counted as busy",
    ),
    ("m11", PROF, '"host_copy", ', "", [TP, "-k", "fake_clock"], "checkpoint host copy left out of busy"),
    ("m12", PROF, '"metrics_device", ', "", [TP, "-k", "fake_clock"], "log-row device wait left out of busy"),
    (
        "c01",
        CLI,
        "profile.PhaseProfiler(sync=args.profile_sync)",
        "profile.PhaseProfiler(sync=False)",
        [TP, "-k", "cli_profiler"],
        "--profile-sync ignored",
    ),
    # 18_loop_profile.py: variant commands, budget, perturbation, report, checkpoint cleanup
    (
        "k01",
        LP,
        'xla_flags = f"{base_flags} {args.det_flags}".strip() if det else base_flags',
        "xla_flags = base_flags",
        [TL],
        "determinism flag never set (H7 a no-op)",
    ),
    ("k02", LP, '"smoke_nodet": (50, 10, True, False, [], True)', '"smoke_nodet": (50, 10, True, True, [], True)', [TL], "smoke_nodet gets the flag"),
    ("k03", LP, "if not fits_budget(clock() - job_start, longest, args.budget_min):", "if False:", [TL], "budget check off"),
    (
        "k04",
        LP,
        '"pure_noprof": (NEVER, PERT, False, False, [], False)',
        '"pure_noprof": (NEVER, PERT, False, False, [], True)',
        [TL],
        "pure_noprof profiled",
    ),
    (
        "s01",
        LP,
        "return elapsed_s + longest_s <= 60 * budget_min",
        "return elapsed_s <= 60 * budget_min",
        [TL],
        "budget ignores the next variant's length",
    ),
    (
        "s02",
        LP,
        "job_start = clock() if args.job_start is None else args.job_start",
        "job_start = clock()",
        [TL],
        "budget counts from the script start",
    ),
    (
        "s03",
        LP,
        '"ratio": w_on["median_ms_after_first"] / w_off["median_ms_after_first"] if valid else None',
        '"ratio": on["result"]["fit_s"] / off["result"]["fit_s"] if valid else None',
        [TL],
        "perturbation from whole fits again",
    ),
    ("s04", LP, "valid = min(n) >= MIN_WINDOWS", "valid = min(n) >= 1", [TL], "perturbation valid with too few windows"),
    (
        "s05",
        LP,
        'per, steady = prof["s_per_micro_batch_loop"], prof["s_per_micro_batch_steady"]',
        'per, steady = prof["s_per_micro_batch_excl_compile"], prof["s_per_micro_batch_steady"]',
        [TL],
        "overhead includes the first batches (review 2: the snippet follows the loop-rate change)",
    ),
    ("s06", LP, "    if not args.keep_ckpts:\n", "    if False:\n", [TL], "checkpoints not deleted"),
    ("s07", LP, "        log_every = args.pert_log_every\n", "        pass\n", [TL], "perturbation pair without log rows"),
    (
        "s08",
        LP,
        '            merged["variants"][name] = res\n',
        '            merged["variants"].setdefault(name, res)\n',
        [TL],
        "report keeps the earlier file's variant",
    ),
    (
        "s09",
        LP,
        '    merged["skipped"] = [n for n in merged["skipped"] if n not in merged["variants"]]\n',
        "",
        [TL],
        "report lists a variant that ran later as skipped",
    ),
    ("s10", LP, 'misses = str(d.get("cache_misses"))', 'misses = "-"', [TL], "table without the cache misses"),
    # review 2 (2026-10-21): the loop window (profile.is_outside_loop) and the overheads / idle fractions that use it
    ("p00d", None, "", "", [TP, "-k", "fake_clock or cli_profiler"], "baseline: the fast profile tests pass (review 2)"),
    (
        "v01",
        PROF,
        'return path.split("/", 1)[0] in OUTSIDE_LOOP and path != LOOP_DRAIN',
        'return path.split("/", 1)[0] in OUTSIDE_LOOP and path != LOOP_DRAIN and "ckpt" not in path',
        [TP, "-k", "fake_clock"],
        "the epoch-end last checkpoint stays in the loop window",
    ),
    (
        "v02",
        PROF,
        'return path.split("/", 1)[0] in OUTSIDE_LOOP and path != LOOP_DRAIN',
        'return path.split("/", 1)[0] in OUTSIDE_LOOP',
        [TP, "-k", "fake_clock"],
        "the epoch drain (queued steps) leaves the loop window",
    ),
    (
        "v03",
        PROF,
        "if not is_compile(path) and is_outside_loop(path))",
        "if is_outside_loop(path))",
        [TP, "-k", "fake_clock"],
        "outside-loop compiles subtracted twice",
    ),
    (
        "v04",
        PROF,
        "loop_wall = wall - compile_s - first_train - outside",
        "loop_wall = wall - compile_s - first_train - first_val - outside",
        [TP, "-k", "fake_clock"],
        "the first validation batch subtracted twice",
    ),
    (
        "v05",
        PROF,
        "busy_loop = sum(s for path, s in device.items() if not is_outside_loop(path))",
        "busy_loop = busy",
        [TP, "-k", "fake_clock"],
        "loop busy time includes validation / last-checkpoint device phases",
    ),
    (
        "v06",
        PROF,
        "idle(busy_loop, loop_wall)",
        "idle(busy_loop, steady_wall)",
        [TP, "-k", "fake_clock"],
        "loop idle fraction over the steady window",
    ),
    (
        "v07",
        PROF,
        'out["s_per_micro_batch_loop"] = loop_wall / steady_micro',
        'out["s_per_micro_batch_loop"] = steady_wall / steady_micro',
        [TP, "-k", "fake_clock"],
        "loop rate = steady rate",
    ),
    (
        "v08",
        PROF,
        "idle(steady_micro * step_ref_s, loop_wall)",
        "idle(steady_micro * step_ref_s, steady_wall)",
        [TP, "-k", "fake_clock or fit_command"],
        "loop idle estimate over the steady window",
    ),
    ("v09", PROF, '"setup", "epoch_start", ', '"epoch_start", ', [TP, "-k", "fake_clock"], "setup counted in the loop window"),
    ("v10", PROF, '"epoch_end", "train_end", ', '"epoch_end", ', [TP, "-k", "fake_clock"], "the train end counted in the loop window"),
    (
        "v11",
        LP,
        'per, steady = prof["s_per_micro_batch_loop"], prof["s_per_micro_batch_steady"]',
        'per, steady = prof["s_per_micro_batch_steady"], prof["s_per_micro_batch_steady"]',
        [TL],
        "overhead vs G3 from the steady rate (outside-loop work in it)",
    ),
    (
        "v12",
        LP,
        "{d['s_per_micro_loop']:8.3f} {d['overhead_vs_g3_ms']",
        "{d['s_per_micro_steady']:8.3f} {d['overhead_vs_g3_ms']",
        [TL],
        "table loop column shows the steady rate",
    ),
    (
        "v13",
        LP,
        'for key in ("device_idle_frac_loop", "device_idle_frac"):',
        'for key in ("device_idle_frac",):',
        [TL],
        "derive drops the loop idle fraction",
    ),
    (
        "v14",
        LOOP,
        "                self.trace.before(self.micro_step)",
        "                self.trace.before(self.micro_step + 1)",
        [TP, "-k", "wrote_a_trace"],
        "the driver starts the trace one micro-batch late",
    ),
    (
        "v15",
        LOOP,
        "                self.trace.after(self.micro_step, self.state)",
        "                self.trace.after(self.micro_step + 1, self.state)",
        [TP, "-k", "wrote_a_trace"],
        "the driver stops the trace one micro-batch early",
    ),
    # the second reviewer's surviving mutants (scratch rev3/mut1.json r01-r04, rev/rmut.out r01, r04-r08, r10, r11)
    (
        "q01",
        LOOP,
        "self.prof.at(self.micro_step + 1, self.global_step, self.epoch)",
        "self.prof.at(self.micro_step, self.global_step, self.epoch)",
        [TP, "-k", "data_position"],
        "micro-batch position off by one",
    ),
    (
        "q02",
        LOOP,
        "self.prof.at(self.micro_step + 1, self.global_step, self.epoch)",
        "self.prof.at(self.micro_step + 1, self.global_step + 1, self.epoch)",
        [TP, "-k", "data_position"],
        "optimizer-step position shifted (steady medians include the first step)",
    ),
    (
        "q03",
        PROF,
        "if not self.running and not self.done and micro_step >= self.first:",
        "if not self.running and not self.done and micro_step >= self.first + 1:",
        [TP, "-k", "wrote_a_trace"],
        "trace window starts one micro-batch late",
    ),
    (
        "q04",
        CLI,
        "report = profiler.summary(step_ref_s=step_ref)",
        "report = profiler.summary(step_ref_s=None)",
        [TP, "-k", "fit_command"],
        "--profile-step-ref-ms ignored",
    ),
    (
        "q05",
        PROF,
        'first_train = by_path["data"][0][3]',
        'first_train = by_path["data"][0][4]',
        [TP, "-k", "fake_clock"],
        "first train batch by exclusive time (loader start leaks)",
    ),
    ("q06", PROF, '"host_copy", "drain", "wait",', '"host_copy",', [TP, "-k", "fake_clock"], "drains left out of busy"),
    (
        "q07",
        PROF,
        '"metrics_device", "bits", "forward", "tail", "meta",',
        '"metrics_device", "forward", "meta",',
        [TP, "-k", "fake_clock"],
        "bits / tail left out of busy",
    ),
    ("q08", LP, "rest = ms[1:]", "rest = ms[2:]", [TL], "log windows drop the second window too"),
    (
        "q09",
        LP,
        'longest = max(longest, res["elapsed_s"])',
        'longest = max(longest, res["elapsed_s"]) if name in FIT_VARIANTS else longest',
        [TL],
        "budget ignores loader variants",
    ),
    ("q10", LP, "log_every = args.pert_log_every", "log_every = 8", [TL], "--pert-log-every ignored"),
    (
        "q11",
        LP,
        'env=driver_env(os.environ.get("XLA_FLAGS", "")), cwd=REPO',
        'env=driver_env("--xla_gpu_deterministic_ops=true"), cwd=REPO',
        [TL],
        "loader runs with the determinism flag",
    ),
    ("q12", LP, "ckpt_every = args.smoke_ckpt_every or ckpt_every", "ckpt_every = ckpt_every", [TL], "--smoke-ckpt-every ignored"),
]


def run(copy: Path, tests: list[str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"}
    env["PYTHONPATH"] = str(copy / "src")
    cmd = [str(python()), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning", *tests]
    return subprocess.run(cmd, cwd=copy, env=env, capture_output=True, text=True, check=False)


def main() -> None:
    scratch = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    copy = scratch / "prof_mut"
    copy.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync") or "rsync"
    excludes = [x for d in (".git", ".venv-jax", ".venv", "__pycache__", "/logs", "/data") for x in ("--exclude", d)]
    subprocess.run([rsync, "-a", "--delete", *excludes, f"{REPO}/", f"{copy}/"], check=True)
    results = []
    for name, rel, old, new, tests, what in MUTATIONS:
        if only and name not in only:
            continue
        start = time.time()
        if rel is None:
            proc = run(copy, tests)
        else:
            path = copy / rel
            original = path.read_text()
            assert original.count(old) == 1, f"{name}: snippet found {original.count(old)} times in {rel}"
            path.write_text(original.replace(old, new))
            try:
                proc = run(copy, tests)
            finally:
                path.write_text(original)
        tail = [line for line in proc.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line][-1:]
        failed_tests = [line.split(" - ")[0] for line in proc.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
        results.append({
            "mutant": name,
            "file": rel,
            "what": what,
            "tests": tests,
            "killed": proc.returncode != 0,
            "failed": failed_tests,
            "summary": tail[0] if tail else proc.stdout[-300:],
            "seconds": round(time.time() - start),
        })
        print(json.dumps(results[-1]), flush=True)
        if out_path:
            out_path.write_text(json.dumps(results, indent=1))
    survivors = [r["mutant"] for r in results if r["killed"] == (r["file"] is None)]  # a baseline must pass
    print(f"{len(results)} mutants, survivors: {survivors}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
