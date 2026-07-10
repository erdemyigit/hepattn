"""Float parity: hepattn Encoder (torch backend) vs KerasEncoder, up to CLIC shape."""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import assert_parity, make_padded_batch

from hepattn.keras.encoder import KerasEncoder
from hepattn.keras.porting import port_encoder
from hepattn.models.encoder import Encoder


def build_pair(seed: int = 0, **cfg) -> tuple[Encoder, KerasEncoder]:
    torch.manual_seed(seed)
    tenc = Encoder(attn_type="torch", **cfg).eval()
    kenc = KerasEncoder(**cfg).eval()
    port_encoder(tenc, kenc)
    return tenc, kenc


def test_hybrid_norm_per_layer_parity():
    """Per-layer comparison localizes hybrid-norm wiring errors.

    A wrong hybrid-norm schedule (pre-norm at depth>0, missing dense post-norm)
    produces O(1) divergence at the specific layer.
    """
    cfg = {"num_layers": 3, "dim": 64, "hybrid_norm": True, "attn_kwargs": {"num_heads": 4}}
    tenc, kenc = build_pair(seed=10, **cfg)
    x, kv_mask = make_padded_batch(2, 20, 64, seed=11)

    h_t, h_k = x, x
    iv_t: dict = {}
    iv_k: dict = {}
    for i, (tl, kl) in enumerate(zip(tenc.layers, kenc.layers, strict=True)):
        with torch.no_grad():
            h_t = tl(h_t, kv_mask=kv_mask, initial_values=iv_t)
        h_k = kl(h_k, kv_mask=kv_mask, initial_values=iv_k)
        assert_parity("encoder.hybrid_norm", f"layer{i}", h_t, h_k, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_register_tokens_parity():
    cfg = {"num_layers": 2, "dim": 64, "num_register_tokens": 4, "attn_kwargs": {"num_heads": 4}}
    tenc, kenc = build_pair(seed=12, **cfg)
    x, kv_mask = make_padded_batch(2, 16, 64, seed=13)
    with torch.no_grad():
        out_t = tenc(x, kv_mask=kv_mask)
    out_k = kenc(x, kv_mask=kv_mask)
    assert out_k.shape == x.shape, "register tokens must be stripped from the output"
    assert_parity("encoder.register_tokens", "2L+4reg", out_t, out_k, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_clic_shaped_encoder_parity():
    """CLIC base.yaml encoder: 6 layers, dim 256, 16 heads, hybrid norm, value residual, 8 registers."""
    cfg = {
        "num_layers": 6,
        "dim": 256,
        "hybrid_norm": True,
        "value_residual": True,
        "num_register_tokens": 8,
        "attn_kwargs": {"num_heads": 16},
    }
    tenc, kenc = build_pair(seed=14, **cfg)
    x, kv_mask = make_padded_batch(2, 80, 256, seed=15)
    with torch.no_grad():
        out_t = tenc(x, kv_mask=kv_mask)
    out_k = kenc(x, kv_mask=kv_mask)
    assert_parity("encoder.clic_shaped", "6L_dim256_h16_hybrid_vres_8reg", out_t, out_k, valid_mask=kv_mask, atol=2e-6, rtol=2e-5)
