"""Parity assertion + ledger for the keras_native tests (numpy in, so torch and JAX outputs compare alike).

Every float/quantized parity check goes through ``assert_parity``, which appends the measured maxima to
``tests/keras_native/.parity_ledger.jsonl`` (git-ignored; set ``HEPATTN_PARITY_LEDGER=0`` to skip writing),
so the stage log quotes measured numbers.
"""

import json
import os
from pathlib import Path

import numpy as np

LEDGER_PATH = Path(__file__).parent / ".parity_ledger.jsonl"


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def record_measurement(record: dict) -> None:
    if os.environ.get("HEPATTN_PARITY_LEDGER", "1") != "0":
        with LEDGER_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")


def max_errors(reference, candidate, valid_mask=None) -> tuple[float, float]:
    ref, cand = to_numpy(reference).astype(np.float64), to_numpy(candidate).astype(np.float64)
    assert ref.shape == cand.shape, f"shape mismatch: {ref.shape} vs {cand.shape}"
    if valid_mask is not None:
        mask = to_numpy(valid_mask).astype(bool)
        mask = mask.reshape(mask.shape + (1,) * (ref.ndim - mask.ndim))
        mask = np.broadcast_to(mask, ref.shape)
        ref, cand = ref[mask], cand[mask]
    diff = np.abs(cand - ref)
    if diff.size == 0:
        return 0.0, 0.0
    return float(diff.max()), float((diff / np.maximum(np.abs(ref), 1e-12)).max())


def assert_parity(tag: str, config: str, reference, candidate, valid_mask=None, atol: float = 1e-6, rtol: float = 1e-5) -> tuple[float, float]:
    """Assert |candidate - reference| <= atol + rtol * |reference| on valid slots; record and return (max abs, max rel)."""
    ref, cand = to_numpy(reference).astype(np.float64), to_numpy(candidate).astype(np.float64)
    max_abs, max_rel = max_errors(ref, cand, valid_mask)
    record_measurement({"tag": tag, "config": config, "max_abs_err": max_abs, "max_rel_err": max_rel, "atol": atol, "rtol": rtol})
    if valid_mask is not None:
        mask = to_numpy(valid_mask).astype(bool)
        mask = np.broadcast_to(mask.reshape(mask.shape + (1,) * (ref.ndim - mask.ndim)), ref.shape)
        ref, cand = ref[mask], cand[mask]
    assert np.isfinite(ref).all() and np.isfinite(cand).all(), f"[{tag} / {config}] non-finite values"
    bad = np.abs(cand - ref) > atol + rtol * np.abs(ref)
    assert not bad.any(), (
        f"[{tag} / {config}] parity violated: max abs err {max_abs:.3e}, max rel err {max_rel:.3e} "
        f"(atol={atol:.1e}, rtol={rtol:.1e}), {int(bad.sum())}/{bad.size} elements out of tolerance"
    )
    return max_abs, max_rel


def make_padded_batch(batch_size: int, max_len: int, dim: int, seed: int = 0, min_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Random (B, S, D) float32 inputs and a True=valid mask with random lengths; event 0 is full length."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch_size, max_len, dim)).astype(np.float32)
    min_len = max(1, max_len // 2) if min_len is None else min_len
    lengths = rng.integers(min_len, max_len + 1, size=batch_size)
    lengths[0] = max_len
    mask = np.arange(max_len)[None, :] < lengths[:, None]
    return x, mask
