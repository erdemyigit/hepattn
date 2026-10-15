"""The hybrid encoder and decoder layers apply the org PR #8 options as torch does.

``base.yaml`` (and so the config ``polaris/03_make_config.py`` writes) sets ``query_update_order: ca_sa_dense``,
``self_attn_norm_kv: true`` and ``dense_norm_placement: legacy``. A hybrid layer that stores but ignores one of
them still passes every attribute-only test, so this file compares each layer with org torch at matched options.
From ``port_log/retarget/trial-probes/probe_pr8_hybrid.py`` (RETARGET_ORG_MAIN.md section 4); R0 measurements in
``port_log/R0.md``.

Controls (each must exceed TOL by at least 10x, or be refused by the porter):
- the other ``query_update_order`` (depth 0 and 1);
- ``self_attn_norm_kv=False`` at depth 0. At depth >= 1 it has no effect under hybrid norm (no pre-attention norm),
  which is asserted too, so that nobody moves this control to depth 1;
- ``dense_norm_placement=hybridnorm`` against torch ``legacy`` (encoder and decoder): the porter refuses the norm mismatch.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from hepattn.keras.decoder import KerasMaskFormerDecoderLayer
from hepattn.keras.encoder import KerasEncoderLayer
from hepattn.keras.porting import port_attention, port_dense, port_encoder_layer, port_residual
from hepattn.models.decoder import MaskFormerDecoderLayer
from hepattn.models.encoder import EncoderLayer

DIM = 32
ENC = {
    "dim": DIM,
    "norm": "FastLayerNorm",
    "hybrid_norm": True,
    "dense_norm_placement": "legacy",
    "attn_kwargs": {"num_heads": 16},
    "dense_kwargs": {"activation": "SwiGLU"},
}
DEC = {**ENC, "query_update_order": "ca_sa_dense", "self_attn_norm_kv": True}

# Measured at R0 (macOS arm64, torch 2.10.0, CPU): max over depth 0/1 of 7.15e-7 (decoder), 4.77e-7 (encoder); 2x.
TOL = 1.44e-6
DEPTHS = [0, 1]


def decoder_diff(tcfg: dict, kcfg: dict, depth: int) -> tuple[float, float]:
    """Max |torch - hybrid| of the decoder layer outputs (queries, valid key slots)."""
    torch.manual_seed(depth)
    tl = MaskFormerDecoderLayer(depth=depth, **tcfg).eval()
    kl = KerasMaskFormerDecoderLayer(depth=depth, **kcfg).eval()
    for part, fn in (
        ("q_ca", port_attention),
        ("q_sa", port_attention),
        ("q_dense", port_dense),
        ("kv_ca", port_attention),
        ("kv_dense", port_dense),
    ):
        port_residual(getattr(tl, part), getattr(kl, part), fn)
    g = torch.Generator().manual_seed(7)
    q = torch.randn(2, 150, DIM, generator=g)
    kv = torch.randn(2, 40, DIM, generator=g)
    attn_mask = torch.rand(2, 150, 40, generator=g) > 0.5
    attn_mask[..., 0] = True
    kv_mask = torch.ones(2, 40, dtype=torch.bool)
    kv_mask[:, 35:] = False
    with torch.no_grad():
        q_t, kv_t = tl(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)
        q_k, kv_k = kl(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)
    return (q_t - q_k).abs().max().item(), (kv_t - kv_k)[kv_mask].abs().max().item()


def encoder_diff(tcfg: dict, kcfg: dict, depth: int) -> float:
    """Max |torch - hybrid| of the encoder layer output on valid slots."""
    torch.manual_seed(depth)
    tl = EncoderLayer(depth=depth, **tcfg).eval()
    kl = KerasEncoderLayer(depth=depth, **kcfg).eval()
    port_encoder_layer(tl, kl)
    x = torch.randn(2, 40, DIM, generator=torch.Generator().manual_seed(3))
    kv_mask = torch.ones(2, 40, dtype=torch.bool)
    kv_mask[:, 35:] = False
    with torch.no_grad():
        y_t = tl(x, kv_mask=kv_mask)
        y_k = kl(x, kv_mask=kv_mask)
    return (y_t - y_k)[kv_mask].abs().max().item()


@pytest.mark.parametrize("depth", DEPTHS)
def test_decoder_layer_matches_torch_with_pr8_options(depth):
    dq, dkv = decoder_diff(DEC, DEC, depth)
    assert dq <= TOL
    assert dkv <= TOL


@pytest.mark.parametrize("depth", DEPTHS)
def test_encoder_layer_matches_torch_with_legacy_placement(depth):
    assert encoder_diff(ENC, ENC, depth) <= TOL


@pytest.mark.parametrize("depth", DEPTHS)
def test_control_wrong_query_update_order(depth):
    dq, _ = decoder_diff(DEC, {**DEC, "query_update_order": "ca_dense_sa"}, depth)
    assert dq >= 10 * TOL


def test_control_self_attn_norm_kv_matters_at_depth_0_only():
    dq0, _ = decoder_diff(DEC, {**DEC, "self_attn_norm_kv": False}, 0)
    assert dq0 >= 10 * TOL
    dq1, _ = decoder_diff(DEC, {**DEC, "self_attn_norm_kv": False}, 1)
    assert dq1 <= TOL


@pytest.mark.parametrize("depth", DEPTHS)
def test_control_wrong_dense_norm_placement_is_refused(depth):
    with pytest.raises(AssertionError, match="post_norm mismatch"):
        encoder_diff(ENC, {**ENC, "dense_norm_placement": "hybridnorm"}, depth)
    with pytest.raises(AssertionError, match="post_norm mismatch"):
        decoder_diff(DEC, {**DEC, "dense_norm_placement": "hybridnorm"}, depth)
