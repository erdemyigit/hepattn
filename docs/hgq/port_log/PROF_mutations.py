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
        "        per = None\n",
        '        per = (summary["fit_s"] - summary["first_step_s"]) / max(1, micro - 1)\n',
        [TL],
        "unprofiled variant gets a compile-including 'excl' rate",
    ),
    (
        "p22",
        LP,
        '        "ratio": on["fit_s"] / off["fit_s"],',
        '        "ratio": off["fit_s"] / on["fit_s"],',
        [TL],
        "perturbation ratio inverted",
    ),
    ("p23", LP, '        "valid": misses == [0, 0],', '        "valid": True,', [TL], "perturbation valid despite a cold compile"),
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
