"""Measure native-vs-hybrid MaskFormer golden differences and write ``mf_measured.json`` (JAX env).

    env -u KERAS_BACKEND PYTHONPATH=<worktree>/src <worktree>/../hepattn-keras-jax/.venv-jax/bin/python \
        tests/keras_native/golden/measure_mf_golden.py

The S3 golden test (``test_mf_golden.py``) allows 2x these values (plan: measure first, then 2x). Rows: every case and
mode with the eager forward, ``<case>/<mode>/jit`` for the jitted forward (``JIT_ROWS``), and ``qlosses.max_rel`` for the
training-mode HGQ2 loss terms of ``stateless_call`` against the hybrid's per-layer terms. Values are rounded
up to 3 significant digits; counts (mask flips, tie rows) are exact. Re-run this, and review the diff, whenever the
golden files are regenerated or the platform changes (a measured 0 gives a tolerance of 0).
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

if os.environ.setdefault("KERAS_BACKEND", "jax") != "jax":
    raise SystemExit("measure_mf_golden.py needs KERAS_BACKEND=jax")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.keras_native.mf_golden_utils import JIT_ROWS, MF_CASES, MF_MODES, build_mf_from_golden, compare_mf, qloss_terms  # noqa: E402

OUT = Path(__file__).resolve().parent / "mf_measured.json"


def round_up(value: float) -> float:
    if value == 0:
        return 0.0
    exp = math.floor(math.log10(value)) - 2
    return float(f"{math.ceil(value / 10**exp) * 10**exp:.3g}")


def main() -> None:
    measured = {}
    for case in MF_CASES:
        for mode in MF_MODES:
            model, golden = build_mf_from_golden(case, mode)
            metrics = compare_mf(model, golden, mode)
            if mode != "float":
                native, hybrid = qloss_terms(model, golden, mode)
                assert native.shape == hybrid.shape, f"{case}/{mode}: {native.size} native vs {hybrid.size} hybrid loss terms"
                metrics["qlosses.max_rel"] = float((np.abs(native - hybrid) / np.abs(hybrid)).max())
            rows = {f"{case}/{mode}": metrics}
            if (case, mode) in JIT_ROWS:
                rows[f"{case}/{mode}/jit"] = compare_mf(model, golden, mode, jit=True)
            for key, row in rows.items():
                measured[key] = {k: round_up(v) if isinstance(v, float) else v for k, v in row.items()}
                print(key, measured[key], flush=True)
    OUT.write_text(json.dumps(measured, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
