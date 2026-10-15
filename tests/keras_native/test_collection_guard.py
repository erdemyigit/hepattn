"""The tests/conftest.py guard: a whole-repo pytest run still collects when tests/keras_native cannot run.

Each case is a fresh ``pytest --collect-only`` subprocess from the repo root, because the keras backend is
fixed per process. The CI case is run with this interpreter and, when present, with the hybrid env
(``HEPATTN_HYBRID_PYTHON``, default ``../hepattn/.venv/bin/python`` next to this worktree).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HYBRID_PYTHON = Path(os.environ.get("HEPATTN_HYBRID_PYTHON", REPO_ROOT.parent / "hepattn" / ".venv" / "bin" / "python"))
# The CI unit job: pyproject [tool.pixi.tasks.test-unit-ci], without the coverage flags.
CI_ARGS = ["--ignore=tests/experiments/", "-m", "not gpu and not requiresdata"]
GUARD_LINE = "tests/keras_native ignored: they need the jax backend"


def collect(args: list[str], backend: str | None, python: str = sys.executable) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND" and not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    if backend is not None:
        env["KERAS_BACKEND"] = backend
    cmd = [python, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q", "-W", "ignore", *args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
    return proc.returncode, proc.stdout + proc.stderr


def node_ids(out: str, prefix: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(prefix)]


@pytest.mark.parametrize("interpreter", ["this", "hybrid"])
def test_ci_command_collects_and_ignores_native(interpreter):
    python = sys.executable
    if interpreter == "hybrid":
        if not HYBRID_PYTHON.exists():
            pytest.skip(f"hybrid env not found at {HYBRID_PYTHON}")
        python = str(HYBRID_PYTHON)
    rc, out = collect(CI_ARGS, backend=None, python=python)
    assert rc == 0, out[-3000:]
    assert len(node_ids(out, "tests/keras/")) > 50, out[-3000:]
    assert node_ids(out, "tests/keras_native/") == []
    assert GUARD_LINE in out


@pytest.mark.parametrize("target", ["tests/keras_native", "tests/keras_native/test_helpers.py"])
def test_explicit_native_target_under_torch_fails_loudly(target):
    rc, out = collect([target], backend="torch")
    assert rc != 0
    assert "tests/keras_native need KERAS_BACKEND=jax, found 'torch'" in out, out[-3000:]


def test_native_collected_by_a_whole_suite_run_under_jax():
    # tests/keras is ignored here because hepattn.keras raises under jax by design.
    rc, out = collect(["--ignore=tests/keras", *CI_ARGS], backend="jax")
    assert rc == 0, out[-3000:]
    assert len(node_ids(out, "tests/keras_native/")) > 40, out[-3000:]
    assert GUARD_LINE not in out
