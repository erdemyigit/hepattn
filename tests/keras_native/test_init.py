"""Initialisation of the weights the native layers create themselves (no porting).

Every parity test ports the weights first, so it cannot see an initializer drift; from-scratch training (S5, S8)
would. Each expected value is checked against the torch reference module built with the same config, so the
statistics below are the torch ones, not a re-derivation. Shapes are large where a sample std is compared
(relative standard error of the std < 1%, tolerance 5%).
"""

import math

import keras
import numpy as np
import pytest
import torch

from hepattn.keras_native.dense import Dense
from hepattn.keras_native.encoder import Encoder, LayerScale
from hepattn.keras_native.factory import LayerFactory
from hepattn.keras_native.input import FourierPositionEncoder, InputNet
from hepattn.models.dense import Dense as TorchDense
from hepattn.models.encoder import Encoder as TorchEncoder
from hepattn.models.encoder import LayerScale as TorchLayerScale
from hepattn.models.posenc import FourierPositionEncoder as TorchFourier

from .golden.cases import QUANT_SPECS

STD_RTOL = 0.05
FP32_SLACK = 1 + 1e-6  # a float32 sample can exceed the float64 bound by one ulp


def n(var) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(var), dtype=np.float64)


def tn(param) -> np.ndarray:
    return param.detach().cpu().numpy().astype(np.float64)


@pytest.fixture(autouse=True)
def seeded():
    keras.utils.set_random_seed(0)
    torch.manual_seed(0)


@pytest.mark.parametrize("init_value", [1e-5, 0.5])
def test_layer_scale_gamma_is_init_value(init_value):
    layer = LayerScale(64, init_value)
    layer.build((None, 10, 64))
    ref = tn(TorchLayerScale(64, init_value).gamma)
    assert layer.gamma.trainable
    np.testing.assert_array_equal(n(layer.gamma), ref)
    assert np.all(ref == np.float32(init_value))


@pytest.mark.parametrize("scale", [0.1, 2.0])
def test_fourier_b_is_frozen_normal_with_std_scale(scale):
    fields, dim = ["eta", "phi"], 32768  # 2 x 16384 samples
    layer = FourierPositionEncoder("node", dim, fields, scale=scale)
    layer.build({})
    b, ref = n(layer.B), tn(TorchFourier("node", dim, fields, scale=scale).B)
    assert not layer.B.trainable
    assert b.shape == ref.shape == (2, dim // 2)
    for sample in (b, ref):
        assert sample.std() == pytest.approx(scale, rel=STD_RTOL)
        assert abs(sample.mean()) < 0.05 * scale


def test_register_tokens_are_standard_normal():
    regs, dim = 8, 4096
    encoder = Encoder(num_layers=0, dim=dim, num_register_tokens=regs)
    encoder.build((None, 160, dim))
    ref = tn(TorchEncoder(num_layers=0, dim=dim, num_register_tokens=regs).register_tokens)
    got = n(encoder.register_tokens)
    assert encoder.register_tokens.trainable
    assert got.shape == ref.shape == (1, regs, dim)
    for sample in (got, ref):
        assert sample.std() == pytest.approx(1.0, rel=STD_RTOL)
        assert abs(sample.mean()) < 0.05


def dense_layers(module) -> list:
    """Every kernel-holding leaf; a SwiGLU Dense has the x1 and gate halves of each hidden layer."""
    return [*module.hidden, *module.gates, module.final]


def torch_linears(module) -> list:
    return [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]


def torch_pieces(module, gate: bool) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """(weight, bias, fan_in) in ``dense_layers`` order; a SwiGLU hidden Linear splits by rows into x1 and gate."""
    *hidden, final = torch_linears(module)
    if gate:
        pieces = [(lin.weight.chunk(2, 0)[half], lin.bias.chunk(2, 0)[half], lin.in_features) for half in (0, 1) for lin in hidden]
    else:
        pieces = [(lin.weight, lin.bias, lin.in_features) for lin in hidden]
    return [*pieces, (final.weight, final.bias, final.in_features)]


@pytest.mark.parametrize("mode", ["float", "high"])
@pytest.mark.parametrize("activation", ["SiLU", "SwiGLU"])
def test_dense_torch_init_matches_nn_linear(mode, activation):
    """init='torch': kernel and bias U(+-1/sqrt(fan_in)) per layer, as the torch Dense the hybrid ports from.

    SwiGLU: both halves of the packed torch Linear(in, 2h) have its fan-in, so both native halves take its bound.
    """
    cfg = {"input_size": 512, "output_size": 64, "hidden_layers": [2048], "activation": activation}
    factory = LayerFactory(QUANT_SPECS[mode])
    with factory.scopes():
        dense = Dense(**cfg, init="torch", factory=factory)
        dense.build((None, 10, 512))
    tdense = TorchDense(**{**cfg, "activation": torch.nn.SiLU() if activation == "SiLU" else "SwiGLU"})
    pieces = torch_pieces(tdense, activation == "SwiGLU")
    assert len(pieces) == (3 if activation == "SwiGLU" else 2)
    for layer, (weight, tbias, fan_in) in zip(dense_layers(dense), pieces, strict=True):
        bound = 1.0 / math.sqrt(fan_in)
        kernel, bias = n(layer.kernel), n(layer.bias)
        assert kernel.shape == tuple(reversed(weight.shape))
        for sample, ref in ((kernel, tn(weight)), (bias, tn(tbias))):
            for values in (sample, ref):
                assert np.abs(values).max() <= bound * FP32_SLACK
                assert np.abs(values).max() > 0.75 * bound
            if sample.size >= 2048:
                assert sample.std() == pytest.approx(ref.std(), rel=STD_RTOL)
                assert sample.std() == pytest.approx(bound / math.sqrt(3), rel=STD_RTOL)


@pytest.mark.parametrize("activation", ["SiLU", "SwiGLU"])
def test_dense_keras_init_is_glorot_with_zero_bias(activation):
    """init='keras' (the default; the hybrid's encoder/decoder Dense layers): glorot-uniform kernel, zero bias.

    SwiGLU: the hybrid packs both halves in one (in, 2h) kernel, so each native half takes that glorot bound.
    """
    dense = Dense(input_size=512, output_size=64, hidden_layers=[2048], activation=activation)
    dense.build((None, 10, 512))
    halves = {id(layer) for layer in (*dense.hidden, *dense.gates)} if activation == "SwiGLU" else set()
    for layer in dense_layers(dense):
        fan_in, fan_out = layer.kernel.shape
        limit = math.sqrt(6.0 / (fan_in + fan_out * (2 if id(layer) in halves else 1)))
        kernel = n(layer.kernel)
        assert np.abs(kernel).max() <= limit * FP32_SLACK
        assert kernel.std() == pytest.approx(limit / math.sqrt(3), rel=STD_RTOL)
        assert not n(layer.bias).any()


def test_dense_rejects_unknown_init():
    with pytest.raises(ValueError, match="init must be"):
        Dense(input_size=4, init="xavier")


@pytest.mark.parametrize("mode", ["float", "high"])
def test_input_net_starts_from_torch_init(mode):
    """The hybrid's input nets are ported torch Dense modules, so the native InputNet defaults to init='torch'."""
    factory = LayerFactory(QUANT_SPECS[mode])
    with factory.scopes():
        net = InputNet("node", net={"input_size": 512, "output_size": 256}, fields=["features"], factory=factory, name="innet0")
        net.build({"node_features": (None, 160, 512)})
    for layer in dense_layers(net.net):
        bound = 1.0 / math.sqrt(layer.kernel.shape[0])
        kernel, bias = n(layer.kernel), n(layer.bias)
        assert np.abs(kernel).max() <= bound * FP32_SLACK
        assert kernel.std() == pytest.approx(bound / math.sqrt(3), rel=STD_RTOL)
        assert np.abs(bias).max() <= bound * FP32_SLACK
        assert np.abs(bias).max() > 0.75 * bound
    explicit = InputNet("node", net={"input_size": 512, "output_size": 256, "init": "keras"}, fields=["features"])
    explicit.build({"node_features": (None, 160, 512)})
    assert not n(explicit.net.final.bias).any(), "an explicit init in net must override the InputNet default"
