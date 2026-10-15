"""Shared setup for the hepattn.keras_native tests: JAX backend, and hepattn from this worktree.

The keras backend is process-global and fixed at the first ``import keras``, so this conftest selects
JAX before anything imports keras. Run these tests in their own pytest process, never together with
``tests/keras`` (the hybrid needs the torch backend).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_requested = os.environ.setdefault("KERAS_BACKEND", "jax")
if _requested != "jax":
    raise RuntimeError(f"tests/keras_native need KERAS_BACKEND=jax, found {_requested!r}. Unset it or set it to 'jax'.")

import keras  # noqa: E402

# Catches a keras imported earlier in this process (e.g. by a plugin) under another backend.
if keras.backend.backend() != "jax":
    raise RuntimeError(f"tests/keras_native need the jax keras backend, got {keras.backend.backend()!r}.")

import hepattn  # noqa: E402

# A hepattn installed from another tree (or another worktree on PYTHONPATH) would silently test the wrong code.
if not Path(hepattn.__file__).resolve().is_relative_to(REPO_ROOT / "src"):
    raise RuntimeError(f"hepattn resolves to {hepattn.__file__}, not to {REPO_ROOT / 'src'}. Run with PYTHONPATH={REPO_ROOT / 'src'}.")
