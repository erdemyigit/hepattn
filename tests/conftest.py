"""Collection guard for ``tests/keras_native``.

The keras backend is process-global, and ``tests/keras_native/conftest.py`` raises unless it is jax. In a
whole-repo run (the CI unit job, ``pytest -m 'not gpu and not requiresdata'``) ``tests/keras`` puts keras on
torch, so without this guard collection stops and nothing runs.

Rule: ignore ``tests/keras_native`` unless the process is on the jax backend; the terminal summary says so.
Naming ``tests/keras_native`` (or a file in it) on the command line still fails loudly under another backend:
pytest loads the conftest of each command-line path at startup, before this hook runs, and that conftest
raises (measured: the hook's answer does not change that outcome).
"""

import os
import sys
from pathlib import Path

import pytest

NATIVE_DIR = Path(__file__).resolve().parent / "keras_native"
_IGNORED = pytest.StashKey[str]()


def _backend_is_jax() -> bool:
    keras = sys.modules.get("keras")
    if keras is not None:
        return keras.backend.backend() == "jax"
    return os.environ.get("KERAS_BACKEND") == "jax"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path.resolve() != NATIVE_DIR or _backend_is_jax():
        return None
    keras = sys.modules.get("keras")
    state = f"keras imported as {keras.backend.backend()!r}" if keras is not None else "keras not imported yet"
    config.stash[_IGNORED] = (
        f"tests/keras_native ignored: they need the jax backend (KERAS_BACKEND={os.environ.get('KERAS_BACKEND')!r}, "
        f"{state}). "
        "Run them in their own process: pytest tests/keras_native (see docs/hgq/KERAS_NATIVE.md)."
    )
    return True


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    reason = config.stash.get(_IGNORED, None)
    if reason:
        terminalreporter.write_sep("-", "keras_native collection guard")
        terminalreporter.write_line(reason)
