"""S2 gate: a fully masked query is independent of the keys (port of tests/keras/test_linformer_parity.py:59-73).

``linformer`` is ``Attention(attn_type="linformer")`` on the attend core (org/main ``masked_linformer.attend``): a query
whose mask row is all False gets a zero core output, so its output is ``out_proj(0)`` (the bias), whatever the keys.
Positive control: an unmasked query does depend on the keys. Negative controls: for dense attention a softmax that does
not zero fully masked rows (masked scores at -1e9, the classic leak); for linformer an attend core without its final
``where`` (the rescale of an all-False row is ``1 / sqrt(1e-16)``, so the leak is large). Either makes the check fail.
"""

import keras
import numpy as np
import pytest
from keras import ops

from hepattn.keras_native.attention import Attention
from hepattn.keras_native.factory import LayerFactory, apply_softmax
from hepattn.keras_native.masked_linformer import rescale

from .golden.cases import QUANT_SPECS
from .parity_utils import make_padded_batch

DIM, HEADS, SEQ, K = 256, 16, 168, 64


class LeakySoftmax(keras.layers.Layer):
    """Masked softmax WITHOUT the fully-masked-row zeroing (negative control)."""

    def call(self, x, attn_mask=None, training=None):
        return keras.ops.softmax(keras.ops.where(attn_mask, x, -1e9), axis=-1)


def attend_without_final_where(core):
    """``AttendCore.call`` with ``p = F(a)`` instead of ``where(m, F(a), 0)`` (negative control)."""

    def call(q, k, v, attn_mask, training=None):
        m = attn_mask[:, None]
        s = ops.where(m, core.qk([q * core.d**-0.5, k], training=training), 0.0)
        s = core.mul_k([core.E(s), rescale(core.kept_k, core.lut_k, attn_mask, core.E.energy(training), training)])
        a = apply_softmax(core.softmax, s, training=training)
        a = core.mul_v([a, rescale(core.kept_v, core.lut_v, attn_mask, core.F.energy(training), training)])
        return core.pv([core.F(a), v], training=training)

    return call


def make_layer(kind: str, mode: str):
    factory = LayerFactory(QUANT_SPECS[mode])
    keras.utils.set_random_seed(7)
    lin = {"attn_type": "linformer", "linformer_seq_len": SEQ, "linformer_k": K} if kind == "linformer" else {}
    with factory.scopes():
        layer = Attention(DIM, num_heads=HEADS, factory=factory, name="attn", **lin)
        layer.build((None, 150, DIM), (None, 168, DIM))
    assert (layer.core_kind == "attend") == (kind == "linformer")
    # non-zero out-projection bias, so "output == bias" is not trivially "output == 0"
    layer.out_proj.bias.assign(np.random.default_rng(0).normal(0, 0.1, DIM).astype(np.float32))
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
        bias = keras.ops.convert_to_numpy(layer.out_proj.bias)
        np.testing.assert_allclose(out1[:, 0], np.broadcast_to(bias, out1[:, 0].shape), atol=1e-6)


@pytest.mark.parametrize("kind", ["linformer", "dense"])
def test_negative_control_leak_breaks_key_independence(kind):
    layer = make_layer(kind, "float")
    # object.__setattr__ bypasses keras' post-build state lock (setattr would raise)
    if kind == "dense":
        object.__setattr__(layer, "attn_softmax", LeakySoftmax())  # noqa: PLC2801
    else:
        object.__setattr__(layer.core, "call", attend_without_final_where(layer.core))  # noqa: PLC2801
    out1, out2 = outputs_for_two_key_sets(layer)
    assert np.abs(out1[:, 0] - out2[:, 0]).max() > 1e-3, "the key-independence check would not catch the leak"


def test_dense_attention_accepts_and_ignores_kv_sort_idx():
    """Dense attention is permutation-equivariant, and org ``Attention.forward`` uses the index only for linformer.

    Exact equality fails for ANY use of the index: a gather of k and v without the mask (R0 review: max 0.63) and
    also a consistent gather of k, v and ``kv_mask``, which is mathematically equal and differs only by summation
    order (max 3.6e-7). The linformer gather is tested against org in test_masked_linformer.py.
    """
    layer = make_layer("dense", "float")
    q, _ = make_padded_batch(2, 150, DIM, seed=9)
    kv, kv_mask = make_padded_batch(2, 168, DIM, seed=10)
    assert not kv_mask.all(), "fixture needs padding, or ignoring the mask and ignoring the index look the same"
    sort_idx = np.stack([np.random.default_rng(s).permutation(168) for s in (1, 2)])
    assert (sort_idx != np.arange(168)).any(), "a non-identity index is needed"
    ref = keras.ops.convert_to_numpy(layer(q, k=kv, v=kv, kv_mask=kv_mask))
    out = keras.ops.convert_to_numpy(layer(q, k=kv, v=kv, kv_mask=kv_mask, kv_sort_idx=sort_idx))
    np.testing.assert_array_equal(out, ref)
