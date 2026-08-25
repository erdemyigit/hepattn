"""Float parity: torch LinformerAttention vs KerasLinformerAttention.

Mirrors the full-attention parity tests: per-projection is implicit (ported weights),
then masked self- and cross-attention end-to-end, plus a mask-polarity probe. k >= kv_len
is required by the projected-space masking (as in the torch reference).
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import assert_parity, make_padded_batch  # ty: ignore [unresolved-import]

from hepattn.keras.linformer import KerasLinformerAttention
from hepattn.keras.porting import port_linformer
from hepattn.models.linformer import LinformerAttention

DIM = 256
HEADS = 16
SEQ = 256
K = 256  # >= kv_len (168) as the masking requires


def build_pair(seed: int = 0) -> tuple[LinformerAttention, KerasLinformerAttention]:
    torch.manual_seed(seed)
    tlin = LinformerAttention(DIM, seq_len=SEQ, k=K, heads=HEADS, dim_head=DIM // HEADS).eval()
    klin = KerasLinformerAttention(DIM, num_heads=HEADS, seq_len=SEQ, k=K).eval()
    port_linformer(tlin, klin)
    return tlin, klin


def test_self_attention_parity():
    tlin, klin = build_pair(seed=1)
    x, _ = make_padded_batch(2, 168, DIM, seed=2)  # encoder-like: N=168, no attn_mask
    with torch.no_grad():
        out_t = tlin(x, k=x, v=x)
    out_k = klin(x)
    assert_parity("linformer.self", "dim256h16k256", out_t, out_k, atol=2e-5, rtol=2e-4)


def test_cross_attention_parity_masked():
    tlin, klin = build_pair(seed=3)
    gen = torch.Generator().manual_seed(4)
    q, _ = make_padded_batch(2, 150, DIM, seed=5)  # decoder queries
    kv, _ = make_padded_batch(2, 168, DIM, seed=6)  # nodes
    attn_mask = torch.rand(2, 150, 168, generator=gen) > 0.3
    attn_mask[..., 0] = True  # every query attends to >=1 key
    with torch.no_grad():
        out_t = tlin(q, k=kv, v=kv, attn_mask=attn_mask)
    out_k = klin(q, k=kv, v=kv, attn_mask=attn_mask)
    assert_parity("linformer.cross_masked", "dim256h16k256", out_t, out_k, atol=2e-5, rtol=2e-4)


def test_fully_masked_query_is_key_independent():
    """A query masked to attend to nothing must output only to_out's bias, regardless of keys.

    Linformer mixes keys via the projection, so the full-attention "perturb masked keys"
    probe doesn't apply. But a FULLY-masked query row has zero attention weights -> zero
    value contribution -> output == to_out(0) == bias, independent of key content. This is
    discriminating: it fails if fully-masked rows are not zeroed (softmax NaN/uniform leak).
    """
    _, klin = build_pair(seed=7)
    q, _ = make_padded_batch(2, 150, DIM, seed=9)
    kv1, _ = make_padded_batch(2, 168, DIM, seed=10)
    kv2, _ = make_padded_batch(2, 168, DIM, seed=11)  # completely different keys
    attn_mask = torch.ones(2, 150, 168, dtype=torch.bool)
    attn_mask[:, 0, :] = False  # query 0 attends to nothing
    with torch.no_grad():
        out1 = klin(q, k=kv1, v=kv1, attn_mask=attn_mask)
        out2 = klin(q, k=kv2, v=kv2, attn_mask=attn_mask)
    assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-6), "fully-masked query output depends on keys"
    assert not torch.allclose(out1[:, 1], out2[:, 1], atol=1e-4), "unmasked query should depend on keys (fixture sanity)"
