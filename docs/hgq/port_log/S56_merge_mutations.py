"""Mutation run for the S5 x S6a cross-lane resume test (S56-merge.md).

Each mutant is applied to a fresh copy of ``src`` and ``tests`` (never to the worktree) and runs
``tests/keras_native/test_resume_cross_lane.py`` there. Usage, from the worktree root::

    KERAS_BACKEND=jax .venv-jax/bin/python docs/hgq/port_log/S56_merge_mutations.py <scratch dir> [name ...]
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEST = "tests/keras_native/test_resume_cross_lane.py"
LOAD_ANCHOR = '            array = arrays[f"{group}/{key}"]\n'

MUTANTS = {
    "XL0_none": [],
    # the accumulated gradient of the micro-batches before the checkpoint is lost on load
    "XL1_accumulators_zeroed": [
        (
            "src/hepattn/keras_native/checkpoint.py",
            LOAD_ANCHOR,
            LOAD_ANCHOR + '            if "accumulat" in key:\n                array = np.zeros_like(array)\n',
        )
    ],
    # beta (an ntv leaf) is not restored
    "XL2_beta_zeroed": [
        (
            "src/hepattn/keras_native/checkpoint.py",
            LOAD_ANCHOR,
            LOAD_ANCHOR + '            if group == "ntv" and key.endswith("beta"):\n                array = np.zeros_like(array)\n',
        )
    ],
    # the optimizer step counter (schedule position, accumulation phase) is off by one
    "XL3_iterations_plus_one": [
        (
            "src/hepattn/keras_native/checkpoint.py",
            LOAD_ANCHOR,
            LOAD_ANCHOR + '            if key == "iteration":\n                array = array + 1\n',
        )
    ],
    # the PID's prev_error is not restored (matters with d > 0)
    "XL4_pid_prev_error_dropped": [("src/hepattn/keras_native/beta_control.py", '        self.prev_error = float(state["prev_error"])\n', "")],
}


def apply(copy: Path, edits) -> None:
    for rel, old, new in edits:
        path = copy / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: anchor found {text.count(old)} times")
        path.write_text(text.replace(old, new))


def main(scratch: str, names: list[str]) -> int:
    results = {}
    for name in names or list(MUTANTS):
        copy = Path(scratch) / name
        shutil.rmtree(copy, ignore_errors=True)
        for part in ("src", "tests"):
            shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
        shutil.copy(ROOT / "pyproject.toml", copy / "pyproject.toml")
        apply(copy, MUTANTS[name])
        env = {**os.environ, "KERAS_BACKEND": "jax", "PYTHONPATH": str(copy / "src")}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", TEST, "-q", "-p", "no:cacheprovider", "-rf"],
            cwd=copy,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (copy / "pytest.out").write_text(proc.stdout + proc.stderr)
        summary = [line for line in proc.stdout.splitlines() if line.startswith(("FAILED", "ERROR")) or " passed" in line or " failed" in line]
        results[name] = summary
        print(name, *summary, sep="\n  ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
