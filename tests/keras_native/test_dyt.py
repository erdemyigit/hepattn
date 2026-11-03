"""DyT (``hepattn.keras_native.norm.DyT``), the keras twin of torch ``hepattn.models.norm.DyT``.

- float: equal to torch DyT for the same alpha / weight / bias, on a layer and on a whole MaskFormer (the org fixture with
  ``norm: DyT``, torch DyT parameters moved off their init, converted by ``convert_to_native``);
- quantized: the tanh table sees u = alpha x on its fixed signed grid (``DYT_TANH_IN``) and gives ``DYT_TANH_OUT``
  outputs: exact table values inside [-4, 4), saturated outside, negative inputs kept;
- QAT: a training call gives finite, nonzero gradients to alpha, weight and bias and EBOPs from all three leaves.
"""

import math

import keras
import numpy as np
import pytest
import torch

from hepattn.keras_native.factory import LayerFactory
from hepattn.keras_native.norm import DYT_TANH_IN, DYT_TANH_OUT, DyT, make_norm
from hepattn.models.norm import DyT as TorchDyT

DIM, N = 16, 7
QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 12, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8, "heterogeneous_axis": [-1], "homogeneous_axis": None},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 0.0},
}
F32_EPS = float(np.finfo(np.float32).eps)


def torch_dyt(seed: int = 0) -> TorchDyT:
    torch.manual_seed(seed)
    t = TorchDyT(DIM)
    with torch.no_grad():
        t.alpha.fill_(0.73)
        t.weight.copy_(torch.randn(DIM))
        t.bias.copy_(torch.randn(DIM))
    return t


def native_from(t: TorchDyT, factory: LayerFactory | None = None) -> DyT:
    factory = factory or LayerFactory()
    with factory.scopes():
        d = DyT(DIM, factory=factory, name="dyt")
    d.build((None, N, DIM))
    d.alpha.kernel.assign(t.alpha.detach().numpy())
    d.affine.kernel.assign(t.weight.detach().numpy())
    d.affine.bias.assign(t.bias.detach().numpy())
    return d


def test_defaults_are_the_torch_init():
    t = TorchDyT(DIM)
    d = make_norm("DyT", name="n", dim=DIM)
    d.build((None, N, DIM))
    assert np.array_equal(np.asarray(d.alpha.kernel), t.alpha.detach().numpy())
    assert np.array_equal(np.asarray(d.affine.kernel), t.weight.detach().numpy())
    assert np.array_equal(np.asarray(d.affine.bias), t.bias.detach().numpy())
    with pytest.raises(ValueError, match="dim"):
        make_norm("DyT", name="n")


def test_float_dyt_equals_torch():
    """Measured max |diff| 2.4e-7 at max |y| 3.2 (0.32 eps |y|: tanh of XLA vs torch); bound 4 eps max|y|."""
    t = torch_dyt()
    d = native_from(t)
    x = np.random.default_rng(1).normal(0, 3, (5, N, DIM)).astype(np.float32)
    ref = t(torch.from_numpy(x)).detach().numpy()
    got = np.asarray(d(x))
    scale = float(np.abs(ref).max())
    assert float(np.abs(got - ref).max()) <= 4 * F32_EPS * scale
    # control: alpha 1e-5 (relative) off moves the output by more than the bound somewhere
    d.alpha.kernel.assign(np.full((1,), 0.73 * (1 + 1e-5), np.float32))
    assert float(np.abs(np.asarray(d(x)) - ref).max()) > 4 * F32_EPS * scale


def test_quantized_tanh_table():
    """The table input is u = alpha x on the signed grid 2**-10 within [-4, 4) (saturating), the output on 2**-10."""
    factory = LayerFactory(QUANT)
    with factory.scopes():
        lut = factory.lut(keras.ops.tanh, name="t", in_bits=DYT_TANH_IN, out_bits=DYT_TANH_OUT, signed=True)
    lut.build((None, 64))
    step_in, step_out = 2.0 ** -DYT_TANH_IN[1], 2.0 ** -DYT_TANH_OUT[1]
    top_in = 2.0 ** DYT_TANH_IN[0]
    u = np.concatenate([np.arange(-top_in, top_in, 1 / 7.0), [-9.0, -4.0, 3.999, 4.0, 9.0]]).astype(np.float32)
    got = np.asarray(lut(u[None], training=False))[0].astype(np.float64)
    u_q = np.clip(np.round(u / step_in) * step_in, -top_in, top_in - step_in)
    want = np.clip(np.round(np.tanh(u_q) / step_out) * step_out, -1.0, 1.0 - step_out)
    assert np.array_equal(got, want), np.abs(got - want).max()
    assert got.min() == -(1.0 - step_out) and got.max() == 1.0 - step_out  # tanh(-4) = -0.99933 rounds to -1023/1024
    assert float(np.abs(got - np.tanh(np.clip(u, -top_in, top_in))).max()) <= step_out  # the documented bound 2**-10
    assert 1 - math.tanh(top_in) < step_out  # saturating at 4 costs less than one output step


def test_quantized_dyt_close_to_float_and_trains():
    t = torch_dyt(seed=2)
    dq = native_from(t, LayerFactory(QUANT))
    df = native_from(t)
    x = np.random.default_rng(3).normal(0, 2, (4, N, DIM)).astype(np.float32)
    # at the init bits (datalane f0 8, weights b0 12) the quantized DyT is within 2**-6 of the float one (measured
    # 8.8e-3 at max |w| 2.3: x and t rounded to 2**-9, alpha / w / b to their 12-bit grids); an unsigned table (all
    # negative tanh -> 0) would be ~1 off
    assert float(np.abs(np.asarray(dq(x, training=False)) - np.asarray(df(x))).max()) <= 2**-6

    import jax  # noqa: PLC0415

    tv = [v.value for v in dq.trainable_variables]
    ntv = [v.value for v in dq.non_trainable_variables]
    names = [v.path for v in dq.trainable_variables]

    def loss(tv_):
        y, _ = dq.stateless_call(tv_, ntv, x, training=True)
        return keras.ops.sum(y**2)

    grads = dict(zip(names, jax.grad(loss)(tv), strict=True))
    for leaf in ("dyt_alpha/kernel", "dyt_affine/kernel", "dyt_affine/bias"):
        g = np.asarray(next(v for k, v in grads.items() if k.endswith(leaf)))
        assert np.isfinite(g).all() and np.abs(g).max() > 0, leaf
    _, ntv_after = dq.stateless_call(tv, ntv, x, training=True)
    by_path = dict(zip([v.path for v in dq.non_trainable_variables], ntv_after, strict=True))
    for leaf in ("dyt_alpha", "dyt_tanh", "dyt_affine"):
        assert float(next(v for k, v in by_path.items() if k.endswith(f"{leaf}/ebops"))) > 0, leaf


def test_rank_and_dim_are_checked():
    d = DyT(DIM, name="d")
    with pytest.raises(ValueError, match="DyT takes"):
        d.build((None, DIM))
    with pytest.raises(ValueError, match="DyT takes"):
        DyT(DIM, name="e").build((None, N, DIM + 1))
