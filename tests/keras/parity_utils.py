"""Shared fixtures and the parity ledger for hepattn.keras tests.

Every parity assertion goes through assert_parity(), which records the measured
maximum absolute/relative error into PARITY_RECORDS. The ledger is dumped to
docs/hgq/PARITY.md by tests/keras/gen_parity_report.py (Stage 3) so the documented
numbers are always the measured ones.
"""

import json
import os
from pathlib import Path

import torch

PARITY_RECORDS: list[dict] = []
_LEDGER_PATH = Path(__file__).parent / ".parity_ledger.jsonl"


def record_measurement(record: dict) -> None:
    """Append a measurement record to the in-memory list and the on-disk ledger."""
    PARITY_RECORDS.append(record)
    if os.environ.get("HEPATTN_PARITY_LEDGER", "1") != "0":
        with _LEDGER_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")


def make_padded_batch(batch_size: int, max_len: int, dim: int, seed: int = 0, min_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (B, S, D) embeddings with a boolean validity mask of random per-event lengths.

    True = valid slot (hepattn convention). The first event always has max_len valid slots
    so the padded and unpadded regimes are both exercised.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, max_len, dim, generator=gen)
    if min_len is None:
        min_len = max(1, max_len // 2)
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), generator=gen)
    lengths[0] = max_len
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(-1)
    return x, mask


def assert_parity(
    tag: str,
    config: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert |candidate - reference| <= atol + rtol*|reference| on valid slots; record measured errors."""
    assert reference.shape == candidate.shape, f"[{tag}] shape mismatch: {reference.shape} vs {candidate.shape}"
    ref = reference.detach()
    cand = candidate.detach()
    if valid_mask is not None:
        while valid_mask.dim() < ref.dim():
            valid_mask = valid_mask.unsqueeze(-1)
        valid_mask = valid_mask.expand_as(ref)
        ref = ref[valid_mask]
        cand = cand[valid_mask]

    diff = (cand - ref).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    denom = ref.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max()) if diff.numel() else 0.0

    record_measurement({"tag": tag, "config": config, "max_abs_err": max_abs, "max_rel_err": max_rel, "atol": atol, "rtol": rtol})

    tol = atol + rtol * ref.abs()
    ok = diff <= tol
    assert bool(ok.all()), (
        f"[{tag} / {config}] parity violated: max abs err {max_abs:.3e}, max rel err {max_rel:.3e} (atol={atol:.1e}, rtol={rtol:.1e}), "
        f"{int((~ok).sum())}/{ok.numel()} elements out of tolerance"
    )
