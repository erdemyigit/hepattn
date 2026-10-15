"""S2 gate: a fully masked query is independent of the keys (port of tests/keras/test_linformer_parity.py:59-73).

Linformer mixes keys through the sequence projection, so the "perturb masked keys" probe does not apply. A query
whose mask row is all False has zero attention weights, so its output is ``to_out(0)`` (the bias), whatever the
keys. Positive control: an unmasked query does depend on the keys. Negative control: with a softmax that does not
zero fully masked rows (masked scores at -1e9, the classic leak), the masked query depends on the keys, so the
check can fail.
"""

import keras
import numpy as np
import pytest

from hepattn.keras_native.attention import Attention
from hepattn.keras_native.factory import LayerFactory
from hepattn.keras_native.linformer import LinformerAttention

from .golden.cases import QUANT_SPECS
from .parity_utils import make_padded_batch

DIM, HEADS, SEQ, K = 256, 16, 256, 256


class LeakySoftmax(keras.layers.Layer):
    """Masked softmax WITHOUT the fully-masked-row zeroing (negative control)."""

    def call(self, x, attn_mask=None, training=None):
        return keras.ops.softmax(keras.ops.where(attn_mask, x, -1e9), axis=-1)


def make_layer(kind: str, mode: str):
    factory = LayerFactory(QUANT_SPECS[mode])
    keras.utils.set_random_seed(7)
    with factory.scopes():
        if kind == "linformer":
            layer = LinformerAttention(DIM, num_heads=HEADS, seq_len=SEQ, k=K, factory=factory, name="lin")
        else:
            layer = Attention(DIM, num_heads=HEADS, factory=factory, name="attn")
        layer.build((None, 150, DIM), (None, 168, DIM))
    # non-zero out-projection bias, so "output == bias" is not trivially "output == 0"
    out_proj = layer.to_out if kind == "linformer" else layer.out_proj
    out_proj.bias.assign(np.random.default_rng(0).normal(0, 0.1, DIM).astype(np.float32))
    return layer


def outputs_for_two_key_sets(layer):
    q, _ = make_padded_batch(2, 150, DIM, seed=9)
    kv1, _ = make_padded_batch(2, 168, DIM, seed=10)
    kv2, _ = make_padded_batch(2, 168, DIM, seed=11)  # completely different keys
    attn_mask = np.ones((2, 150, 168), dtype=bool)
    attn_mask[:, 0, :] = False  # query 0 attends to nothing
    out1 = keras.ops.convert_to_numpy(layer(q, k=kv1, v=kv1, attn_mask=attn_mask))
    out2 = keras.ops.convert_to_numpy(layer(q, k=kv2, v=kv2, attn_mask=attn_mask))
    return out1, out2


@pytest.mark.parametrize("mode", ["float", "high"])
@pytest.mark.parametrize("kind", ["linformer", "dense"])
def test_fully_masked_query_is_key_independent(kind, mode):
    layer = make_layer(kind, mode)
    out1, out2 = outputs_for_two_key_sets(layer)
    np.testing.assert_allclose(out1[:, 0], out2[:, 0], atol=1e-6, err_msg="fully masked query output depends on keys")
    # positive control: an unmasked query depends on the keys (fixture sanity)
    assert not np.allclose(out1[:, 1], out2[:, 1], atol=1e-4), "unmasked query should depend on keys"
    if mode == "float":
        # the masked output is exactly the out-projection bias (zero attention weights)
        bias = keras.ops.convert_to_numpy((layer.to_out if kind == "linformer" else layer.out_proj).bias)
        np.testing.assert_allclose(out1[:, 0], np.broadcast_to(bias, out1[:, 0].shape), atol=1e-6)


@pytest.mark.parametrize("kind", ["linformer", "dense"])
def test_negative_control_leaky_softmax_breaks_key_independence(kind):
    layer = make_layer(kind, "float")
    # object.__setattr__ bypasses keras' post-build state lock (setattr would raise)
    object.__setattr__(layer, "attn_softmax", LeakySoftmax())  # noqa: PLC2801
    out1, out2 = outputs_for_two_key_sets(layer)
    assert np.abs(out1[:, 0] - out2[:, 0]).max() > 1e-3, "the key-independence check would not catch a leaking softmax"


@pytest.mark.parametrize("kind", ["linformer", "dense"])
def test_kv_sort_idx_is_accepted_and_ignored(kind):
    """R0 interface stub: ``call`` takes ``kv_sort_idx`` and ignores it (bit-exact output).

    Exact equality fails for ANY use of the index: a gather of k and v without the mask (R0 review: max 0.63)
    and also a consistent gather of k, v and ``kv_mask``, which is mathematically equal for dense attention and
    differs only by summation order (max 3.6e-7). So this pins "ignored", not "gathered correctly"; R2 replaces
    it when it implements the gather. For ``linformer`` only the signature is pinned: Helen's layer discards
    ``kv_mask``.
    """
    layer = make_layer(kind, "float")
    q, _ = make_padded_batch(2, 150, DIM, seed=9)
    kv, kv_mask = make_padded_batch(2, 168, DIM, seed=10)
    if kind == "dense":  # native LinformerAttention drops kv_mask, so the padding matters only here
        assert not kv_mask.all(), "fixture needs padding, or ignoring the mask and ignoring the index look the same"
    sort_idx = np.stack([np.random.default_rng(s).permutation(168) for s in (1, 2)])
    assert (sort_idx != np.arange(168)).any(), "a non-identity index is needed"
    ref = keras.ops.convert_to_numpy(layer(q, k=kv, v=kv, kv_mask=kv_mask))
    out = keras.ops.convert_to_numpy(layer(q, k=kv, v=kv, kv_mask=kv_mask, kv_sort_idx=sort_idx))
    np.testing.assert_array_equal(out, ref)
