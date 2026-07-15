"""Functional export and hls4ml conversion gates + characterization tests.

Layered by what the toolchain supports:
- functional assembly / save-load: run everywhere;
- hls4ml CONVERSION (graph parse, bit-exact precision propagation, HLS codegen):
  run everywhere hls4ml is installed;
- hls4ml C-SIMULATION bit-exactness (hls_model.predict == keras, exact): requires a
  toolchain compatible with hls4ml's ap_types headers — Apple clang/libc++ is not,
  so these skip on macOS and gate on linux;
- characterization tests PIN the currently unsupported constructs (inline silu
  activation, rank-3 pointwise Dense) so an hls4ml upgrade that adds support is
  noticed. The deployment implications are documented in docs/hgq/FPGA_STATUS.md.
"""

import importlib.util

import numpy as np
import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from test_maskformer_parity import clic_dummy_batch, make_keras_model
from test_quantized import HIGH_QUANT

from hepattn.keras import keras
from hepattn.keras.export import build_functional_dense, convert_to_hls, load_keras_model, save_keras_model

HAS_HLS4ML = importlib.util.find_spec("hls4ml") is not None

TOY_SCOPES_KW = {
    "weight": {"place": "weight", "default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"place": "datalane", "default_q_type": "kif", "i0": 4, "f0": 8},
}


def toy_scopes():
    import contextlib  # noqa: PLC0415

    from hgq.config import LayerConfigScope, QuantizerConfigScope  # noqa: PLC0415

    stack = contextlib.ExitStack()
    stack.enter_context(QuantizerConfigScope(**TOY_SCOPES_KW["weight"]))
    stack.enter_context(QuantizerConfigScope(**TOY_SCOPES_KW["datalane"]))
    stack.enter_context(LayerConfigScope(enable_ebops=False))
    return stack


def build_toy_mlp() -> keras.Model:
    """CLIC-head-shaped pointwise QDense MLP in the convertible form (relu; silu needs a LUT layer)."""
    from hgq.layers import QDense  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(150, 32), name="mlp_in")
        x = QDense(16, activation="relu", name="mlp_hidden")(inp)
        x = QDense(6, name="mlp_out")(x)
        return keras.Model(inp, x, name="toy_mlp")


def build_toy_lut_mlp() -> keras.Model:
    """Per-token MLP with silu expressed as QUnaryFunctionLUT — the deployable form of CLIC's heads."""
    from hgq.layers import QDense, QUnaryFunctionLUT  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(32,), name="lut_mlp_in")
        x = QDense(16, name="lut_mlp_hidden")(inp)
        x = QUnaryFunctionLUT(keras.activations.silu, allow_heterogeneous_table=False, name="lut_mlp_silu")(x)
        x = QDense(6, name="lut_mlp_out")(x)
        return keras.Model(inp, x, name="toy_lut_mlp")


def build_toy_attention_core() -> keras.Model:
    """The norm-free core of KerasAttention at toy size with fixed shapes."""
    from hgq.layers import QDense, QEinsum, QSoftmax  # noqa: PLC0415

    seq, dim, heads = 8, 16, 4
    head_dim = dim // heads
    with toy_scopes():
        inp = keras.Input(shape=(seq, dim), name="core_input")
        q = keras.layers.Reshape((seq, heads, head_dim))(QDense(dim, name="core_q")(inp))
        k = keras.layers.Reshape((seq, heads, head_dim))(QDense(dim, name="core_k")(inp))
        v = keras.layers.Reshape((seq, heads, head_dim))(QDense(dim, name="core_v")(inp))
        scores = QEinsum("bqhd,bkhd->bhqk", name="core_scores")([q, k])
        attn = QSoftmax(axis=-1, name="core_softmax")(scores)
        out = QEinsum("bhqk,bkhd->bqhd", name="core_values")([attn, v])
        out = keras.layers.Reshape((seq, dim))(out)
        out = QDense(dim, name="core_out")(out)
        return keras.Model(inp, out, name="attention_core")


def csim_or_skip(model: keras.Model, output_dir):
    """Convert with C simulation; skip the test where the local toolchain cannot build it."""
    try:
        return convert_to_hls(model, output_dir, compile_csim=True)
    except Exception as err:
        if "Failed to compile project" in str(err):
            pytest.skip("hls4ml csim toolchain unavailable (ap_types headers incompatible with this compiler, e.g. Apple libc++)")
        raise


@pytest.fixture(scope="module")
def quant_model():
    model = make_keras_model(70, quant=HIGH_QUANT).eval()
    inputs, _ = clic_dummy_batch()
    with torch.no_grad():
        model(inputs)  # materialize lazily-built layers
    return model


def test_functional_head_matches_orchestrated(quant_model):
    """The functional assembly must share weights and reproduce the orchestrated output bit-identically."""
    head = quant_model.tasks[0].net  # classification head: (B, 150, 32) -> (B, 150, 6)
    functional = build_functional_dense(head, input_shape=(150, 32), name="classification_head")

    x = torch.randn(3, 150, 32)
    with torch.no_grad():
        out_orchestrated = head(x)
        out_functional = functional(x, training=False)
    assert torch.equal(out_orchestrated, out_functional), "functional assembly diverges from orchestrated layers"


def test_functional_save_load_roundtrip(quant_model, tmp_path):
    head = quant_model.tasks[0].net
    functional = build_functional_dense(head, input_shape=(150, 32), name="classification_head")
    path = tmp_path / "head.keras"
    save_keras_model(functional, path)
    loaded = load_keras_model(path)

    x = torch.randn(2, 150, 32)
    with torch.no_grad():
        a = functional(x, training=False)
        b = loaded(x, training=False)
    assert torch.equal(a, b), "save/load roundtrip changed outputs"


@pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
def test_hls4ml_mlp_conversion_and_codegen(tmp_path):
    """Graph parse + bit-exact precision propagation + HLS codegen must succeed for the MLP form."""
    hls_model = convert_to_hls(build_toy_mlp(), tmp_path / "hls_mlp", compile_csim=False)
    assert (tmp_path / "hls_mlp" / "firmware").exists(), "no HLS firmware generated"
    assert hls_model is not None


@pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
def test_hls4ml_lut_activation_conversion_and_codegen(tmp_path):
    """Silu as an explicit QUnaryFunctionLUT (homogeneous table, rank-2) converts — the deployable head form."""
    hls_model = convert_to_hls(build_toy_lut_mlp(), tmp_path / "hls_lut", compile_csim=False)
    assert (tmp_path / "hls_lut" / "firmware").exists(), "no HLS firmware generated"
    assert hls_model is not None


@pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
def test_hls4ml_attention_core_conversion_and_codegen(tmp_path):
    """QEinsum + QSoftmax + QDense attention core converts and generates HLS."""
    hls_model = convert_to_hls(build_toy_attention_core(), tmp_path / "hls_core", compile_csim=False)
    assert (tmp_path / "hls_core" / "firmware").exists(), "no HLS firmware generated"
    assert hls_model is not None


@pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
def test_hls4ml_mlp_bit_exact_csim(tmp_path):
    """The FPGA gate: hls4ml C simulation must equal keras EXACTLY (linux toolchain)."""
    model = build_toy_mlp()
    hls_model = csim_or_skip(model, tmp_path / "hls_mlp_csim")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 150, 32)).astype(np.float32)
    with torch.no_grad():
        y_keras = np.asarray(model(torch.from_numpy(x), training=False))
    y_hls = np.asarray(hls_model.predict(x)).reshape(y_keras.shape)
    assert np.array_equal(y_hls, y_keras), f"hls4ml csim differs from keras (max abs diff {np.abs(y_hls - y_keras).max():.3e})"


@pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
def test_hls4ml_attention_core_bit_exact_csim(tmp_path):
    model = build_toy_attention_core()
    hls_model = csim_or_skip(model, tmp_path / "hls_core_csim")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 8, 16)).astype(np.float32)
    y_keras = np.asarray(model(torch.from_numpy(x), training=False))
    y_hls = np.asarray(hls_model.predict(x)).reshape(y_keras.shape)
    assert np.array_equal(y_hls, y_keras), f"attention core csim not bit-exact (max abs diff {np.abs(y_hls - y_keras).max():.3e})"


@pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
def test_hls4ml_characterized_limitation_inline_silu(tmp_path):
    """PINS a current limitation: inline silu inside QDense is NOT convertible (hls4ml 1.3).

    Deployable heads must express nonlinear activations as explicit QUnaryFunctionLUT
    layers instead. If this test ever fails, hls4ml gained support — update
    docs/hgq/FPGA_STATUS.md and the export path.
    """
    from hgq.layers import QDense  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(32,))
        x = QDense(16, activation="silu")(inp)
        model = keras.Model(inp, QDense(6)(x))
    with pytest.raises(Exception):  # noqa: B017  — converter raises bare AssertionError
        convert_to_hls(model, tmp_path / "hls_silu", compile_csim=False)


@pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
def test_hls4ml_characterized_limitation_lut_rank3(tmp_path):
    """PINS a current limitation: QUnaryFunctionLUT on rank-3 (pointwise) inputs fails to convert.

    (HGQ2 builds its table-domain variables at the input rank, and the conversion-time
    table materialization evaluates on a rank-2 grid.) Per-query heads using LUT
    activations must therefore be exported per-token (rank-2). If this test fails,
    support was added upstream — update docs/hgq/FPGA_STATUS.md.
    """
    from hgq.layers import QDense, QUnaryFunctionLUT  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(8, 32))
        x = QDense(16)(inp)
        x = QUnaryFunctionLUT(keras.activations.silu, allow_heterogeneous_table=False)(x)
        model = keras.Model(inp, QDense(6)(x))
    with pytest.raises(Exception):  # noqa: B017  — HGQ2/converter raises RuntimeError today
        convert_to_hls(model, tmp_path / "hls_lut_rank3", compile_csim=False)
