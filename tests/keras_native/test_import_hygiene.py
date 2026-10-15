"""S1 gate: no module of hepattn.keras_native imports torch.

Each check runs in a fresh interpreter, because this pytest process already holds torch (the float
parity tests import the torch reference models). The scan imports every module that
``pkgutil.walk_packages`` finds, so modules added by later stages are covered without editing this file.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN = textwrap.dedent(
    """
    import importlib, json, pkgutil, sys
    pkg = importlib.import_module(sys.argv[1])
    names = [pkg.__name__]
    if hasattr(pkg, "__path__"):
        for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
            importlib.import_module(info.name)
            names.append(info.name)
    import keras
    torch_mods = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
    print(json.dumps({"modules": names, "torch": torch_mods, "backend": keras.backend.backend()}))
    """
)


def scan(package: str, backend: str, extra_path: Path | None = None) -> subprocess.CompletedProcess:
    paths = [str(REPO_ROOT / "src")] + ([str(extra_path)] if extra_path else [])
    env = {**os.environ, "KERAS_BACKEND": backend, "PYTHONPATH": os.pathsep.join(paths)}
    return subprocess.run([sys.executable, "-c", SCAN, package], env=env, capture_output=True, text=True, timeout=300, check=False)


def result(proc: subprocess.CompletedProcess) -> dict:
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_keras_native_imports_no_torch():
    out = result(scan("hepattn.keras_native", "jax"))
    assert out["backend"] == "jax"
    # The scan must have reached the submodules, not only the package __init__.
    assert "hepattn.keras_native.factory" in out["modules"], out["modules"]
    assert out["torch"] == [], f"torch imported by hepattn.keras_native: {out['torch']}"


def test_control_hybrid_package_imports_torch():
    """Control: the same scan of hepattn.keras (torch backend) reports torch."""
    out = result(scan("hepattn.keras", "torch"))
    assert out["backend"] == "torch"
    assert "torch" in out["torch"]


def test_control_hybrid_package_raises_under_jax():
    proc = scan("hepattn.keras", "jax")
    assert proc.returncode != 0
    assert "hepattn.keras requires KERAS_BACKEND='torch'" in proc.stderr


def test_control_scan_finds_torch_in_a_submodule(tmp_path):
    """Control for the scanner: a clean __init__ with a torch-importing submodule must be caught."""
    pkg = tmp_path / "fake_native"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sub").mkdir()
    (pkg / "sub" / "__init__.py").write_text("")
    (pkg / "sub" / "uses_torch.py").write_text("import torch\n")
    out = result(scan("fake_native", "jax", extra_path=tmp_path))
    assert "fake_native.sub.uses_torch" in out["modules"]
    assert "torch" in out["torch"]
