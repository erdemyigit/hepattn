"""``build_keras_attention`` honours ``linformer_k`` (org PR #9) and rejects every option org/main Attention lacks.

Helen's KerasLinformerAttention swallowed unknown kwargs, so a dropped ``linformer_k`` silently built k=256 (the bug
class behind the retarget, RETARGET_ORG_MAIN.md section 1.3). Since R4b the hybrid builds the masked-Linformer
KerasAttention, and ``linformer_proj_dim`` or a misspelt key raises TypeError, as native ``build_attention`` does
(plan section 5.1). k=8 is far from 256.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from hepattn.keras.attention import KerasAttention, build_keras_attention

DIM, HEADS, SEQ_LEN, K = 32, 4, 64, 8


def test_linformer_k_sets_the_projection_rank():
    assert K != 256, "a k equal to the old default cannot show that the kwarg was read"
    layer = build_keras_attention(DIM, "linformer", num_heads=HEADS, linformer_seq_len=SEQ_LEN, linformer_k=K, name="lk")
    assert type(layer) is KerasAttention and (layer.linformer_seq_len, layer.linformer_k) == (SEQ_LEN, K)
    x = torch.randn(2, 40, DIM)
    kv_mask = torch.ones(2, 40, dtype=torch.bool)
    kv_mask[0, -3:] = False
    with torch.no_grad():
        layer(x, kv_mask=kv_mask)  # builds the core: masked project over 40 keys, tail rows 40..63 for the renorm
    assert tuple(layer.core.E.kernel.shape) == (40, K) and tuple(layer.core.E.tail.shape) == (SEQ_LEN - 40, K)
    assert [tuple(p.shape) for p in layer.full_projections()] == [(SEQ_LEN, K), (SEQ_LEN, K)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"linformer_seq_len": SEQ_LEN, "linformer_proj_dim": 16}, "linformer_k"),
        ({"linformer_seq_len": SEQ_LEN, "linformer_k": K, "linformer_proj_dim": 16}, "linformer_k"),
        ({"linformer_seq_len": SEQ_LEN, "linformer_K": K}, "unknown attention option"),
        ({"linformer_seq_len": SEQ_LEN, "linformer_k": K, "attn_typ": "torch"}, "unknown attention option"),
    ],
    ids=["legacy", "both", "misspelt", "unknown"],
)
def test_options_org_attention_lacks_raise(kwargs, match):
    with pytest.raises(TypeError, match=match):
        build_keras_attention(DIM, "linformer", num_heads=HEADS, **kwargs)


def test_linformer_needs_seq_len_and_k():
    with pytest.raises(ValueError, match="linformer_seq_len and linformer_k"):
        build_keras_attention(DIM, "linformer", num_heads=HEADS, linformer_seq_len=SEQ_LEN)


def test_dense_attention_accepts_and_ignores_the_linformer_keys():
    """As org's Attention signature: the keys are valid options, only attn_type='linformer' uses them."""
    layer = build_keras_attention(DIM, "torch", num_heads=HEADS, linformer_seq_len=SEQ_LEN, linformer_k=K, name="ld")
    assert layer.attn_type == "torch" and not hasattr(layer, "core")
