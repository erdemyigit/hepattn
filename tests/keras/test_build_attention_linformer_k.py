"""``build_keras_attention`` honours ``linformer_k``, the torch Attention spelling since org PR #9.

KerasLinformerAttention swallows unknown kwargs, so a dropped ``linformer_k`` silently builds the default
k=256 (the bug class behind the retarget, RETARGET_ORG_MAIN.md section 1.3). k=8 is far from that default.
R4b replaces this branch with the masked-Linformer core and makes ``linformer_proj_dim`` raise (plan 5.1);
it rewrites the ``both`` row then.
"""

import pytest

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from hepattn.keras.attention import build_keras_attention

DEFAULT_K = 256


@pytest.mark.parametrize(
    ("kwargs", "k"),
    [
        ({"linformer_k": 8}, 8),
        ({"linformer_k": 8, "linformer_proj_dim": 16}, 8),  # both given: linformer_k wins
        ({"linformer_proj_dim": 16}, 16),  # pre-PR #9 spelling, still read until R4b
    ],
    ids=["k", "both", "legacy"],
)
def test_linformer_k_sets_the_projection_rank(kwargs, k):
    assert k != DEFAULT_K, "a k equal to the default cannot show that the kwarg was read"
    layer = build_keras_attention(32, "linformer", num_heads=4, linformer_seq_len=64, **kwargs)
    assert layer.k == k
    assert tuple(layer.proj_k.shape) == (64, k)
    assert tuple(layer.proj_v.shape) == (64, k)
