"""S1 gate: no module of hepattn.keras_native imports torch.

Two checks. (1) Runtime: a fresh interpreter (this pytest process already holds torch for the float parity
tests) imports every module that ``pkgutil.walk_packages`` finds, then inspects ``sys.modules``.
(2) Static: every ``.py`` file is parsed and any import of a torch-stack package, or of a hepattn module
outside keras_native (those import torch), is rejected at any nesting depth. (2) catches what (1) cannot:
an ``import torch`` inside a function body, which only runs at call time. Both cover new modules
without editing this file.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_SRC = REPO_ROOT / "src" / "hepattn" / "keras_native"
NATIVE_PKG = "hepattn.keras_native"
# Top-level packages that are, or import, torch.
FORBIDDEN_ROOTS = {"torch", "torchvision", "torchaudio", "torchmetrics", "lightning", "pytorch_lightning", "lion_pytorch", "torchjd"}

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


def _forbidden(name: str) -> bool:
    root = name.split(".", 1)[0]
    if root in FORBIDDEN_ROOTS:
        return True
    # Everything in hepattn outside keras_native may import torch (hepattn.keras, .models, .utils.scaling, ...).
    return root == "hepattn" and name not in {"hepattn", NATIVE_PKG} and not name.startswith(NATIVE_PKG + ".")


def _resolve_relative(module: str | None, level: int, file_module: str, is_init: bool) -> str:
    package = file_module.split(".") if is_init else file_module.split(".")[:-1]
    base = package[: len(package) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def forbidden_imports(src_dir: Path, package: str) -> tuple[list[str], list[str]]:
    """Return (scanned files, violations as 'file:line: module') for every .py file under src_dir."""
    scanned, bad = [], []
    for path in sorted(src_dir.rglob("*.py")):
        rel = path.relative_to(src_dir)
        is_init = path.name == "__init__.py"
        parts = [package, *rel.parent.parts] + ([] if is_init else [path.stem])
        file_module = ".".join(parts)
        scanned.append(str(rel))
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    mod = _resolve_relative(node.module, node.level, file_module, is_init)
                names = [mod] + [f"{mod}.{a.name}" for a in node.names if a.name != "*"]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                func = node.func
                fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if fname in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
                    names = [node.args[0].value]
            bad += [f"{rel}:{node.lineno}: {n}" for n in names if _forbidden(n)]
    return scanned, bad


def test_keras_native_has_no_forbidden_import_statements():
    scanned, bad = forbidden_imports(NATIVE_SRC, NATIVE_PKG)
    assert "factory.py" in scanned, scanned
    assert bad == [], f"keras_native imports torch-stack modules: {bad}"


FORBIDDEN_SNIPPETS = {
    "top_level": "import torch\n",
    "submodule": "import torch.nn as nn\n",
    "lazy_in_function": "def load(path):\n    import torch\n    return torch.load(path)\n",
    "nested_in_class": "class A:\n    def f(self):\n        if True:\n            from torch import nn\n",
    "lightning": "import lightning as L\n",
    "hybrid_package": "from hepattn.keras import keras\n",
    "hepattn_models_via_from": "from hepattn import models\n",
    "relative_to_hybrid": "from ...keras.factory import LayerFactory\n",
    "relative_parent_name": "from ... import models\n",
    "utils_scaling": "from hepattn.utils.scaling import FeatureScaler\n",
    "importlib": "import importlib\nimportlib.import_module('torch')\n",
    "dunder_import": "__import__('torch.nn')\n",
}


@pytest.mark.parametrize("case", sorted(FORBIDDEN_SNIPPETS))
def test_control_static_check_flags(case, tmp_path):
    """Control: each forbidden form, written into a copy-shaped package, is reported."""
    src = tmp_path / "keras_native"
    (src / "sub").mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "sub" / "__init__.py").write_text("")
    (src / "sub" / "mod.py").write_text(FORBIDDEN_SNIPPETS[case])
    _, bad = forbidden_imports(src, NATIVE_PKG)
    assert len(bad) >= 1, case
    assert all(b.startswith("sub/mod.py:") for b in bad), bad


def test_control_static_check_allows_native_and_backend_neutral_imports(tmp_path):
    src = tmp_path / "keras_native"
    (src / "sub").mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "sub" / "__init__.py").write_text("from .. import factory\nfrom . import mod\n")
    (src / "sub" / "mod.py").write_text(
        "import keras\nimport numpy as np\nimport hepattn\nfrom hepattn.keras_native.factory import LayerFactory\n"
        "from ..norm import get_hybrid_norm_config\nfrom hepattn import keras_native\nimport torchlike_but_not_torch\n"
    )
    scanned, bad = forbidden_imports(src, NATIVE_PKG)
    assert len(scanned) == 3
    assert bad == []
