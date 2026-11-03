"""Rounded attention score scale (``Attention(scale_frac_bits=F)``, ``masked_linformer.attention_scale``).

Alkaid implements a constant multiply exactly as shifts and adds, but raises or hangs on a non-dyadic float64 constant
such as ``8**-0.5`` (calad0i/alkaid#8). With ``scale_frac_bits`` every core (dense, Linformer attend / project /
project_unmasked) scales by the rounded constant: checked functionally, since an attention with the rounded scale equals
the exact-scale attention whose q projection is multiplied by ``rounded / exact``. Default (None): the exact scale, so
the torch parity tests stay as they were.
"""

import math

import keras
import numpy as np
import pytest

from hepattn.keras_native.attention import Attention
from hepattn.keras_native.masked_linformer import attention_scale

DIM, HEADS, NQ, N = 16, 2, 6, 10


def test_scale_values():
    assert attention_scale(8) == 8**-0.5
    s = attention_scale(8, 12)
    assert s == 1448 / 4096 and (s * 2**12).is_integer()
    assert abs(s / 8**-0.5 - 1) < 1.1e-4  # F = 12: about 1e-4 relative
    assert attention_scale(8, 4) == 6 / 16
    with pytest.raises(ValueError, match="frac_bits"):
        attention_scale(8, 0)


def make(kind: str, frac_bits: int | None) -> Attention:
    linformer = {"attn_type": "linformer", "linformer_seq_len": N, "linformer_k": 4, "linformer_core": kind}
    kwargs = {"attn_type": "torch"} if kind == "torch" else linformer
    keras.utils.set_random_seed(0)
    a = Attention(DIM, num_heads=HEADS, scale_frac_bits=frac_bits, name="a", **kwargs)
    q_shape = (None, NQ, DIM)
    a.build(q_shape, q_shape if kind in {"project", "project_unmasked"} else (None, N, DIM))
    return a


def run(a: Attention, kind: str, q, kv, kv_mask, attn_mask):
    if kind == "torch":
        return np.asarray(a(q, kv, kv))
    if kind == "attend":
        return np.asarray(a(q, kv, kv, attn_mask=attn_mask))
    if kind == "project":
        return np.asarray(a(q, kv_mask=kv_mask))
    return np.asarray(a(q))


@pytest.mark.parametrize("kind", ["torch", "attend", "project", "project_unmasked"])
def test_every_core_uses_the_rounded_scale(kind):
    rng = np.random.default_rng(1)
    n_kv = NQ if kind in {"project", "project_unmasked"} else N
    q = rng.normal(0, 1, (3, NQ, DIM)).astype(np.float32)
    kv = q if n_kv == NQ else rng.normal(0, 1, (3, N, DIM)).astype(np.float32)
    kv_mask = np.ones((3, n_kv), bool)
    kv_mask[:, -2:] = False
    attn_mask = rng.random((3, NQ, N)) < 0.7
    attn_mask[..., 0] = True
    rounded, exact = make(kind, 4), make(kind, None)
    assert exact.scale == (DIM // HEADS) ** -0.5 and rounded.scale == 6 / 16
    if exact.core is not None:
        assert exact.core.scale == exact.scale and rounded.core.scale == rounded.scale
    exact.set_weights(rounded.get_weights())
    out_rounded = run(rounded, kind, q, kv, kv_mask, attn_mask)
    out_exact = run(exact, kind, q, kv, kv_mask, attn_mask)
    ratio = rounded.scale / exact.scale
    exact.q_proj.kernel.assign(np.asarray(exact.q_proj.kernel) * ratio)
    exact.q_proj.bias.assign(np.asarray(exact.q_proj.bias) * ratio)
    out_folded = run(exact, kind, q, kv, kv_mask, attn_mask)
    scale = float(np.abs(out_rounded).max())
    assert float(np.abs(out_folded - out_rounded).max()) <= 1e-5 * scale  # measured <= 3.2e-7 x scale
    assert float(np.abs(out_exact - out_rounded).max()) > 1e-3 * scale  # the 6% scale change: measured 0.023-0.056 x scale
    assert math.isfinite(scale)
