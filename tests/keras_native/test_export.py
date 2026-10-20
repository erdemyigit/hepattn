"""S7 export gates under KERAS_BACKEND=jax (``hepattn.keras_native.export``).

Part 1 mirrors the 9 tests of ``tests/keras/test_export.py`` (the hybrid, torch backend) on the native code: functional
assembly and save / load of a trained head, hls4ml conversion + code generation of the toy MLP, LUT MLP and attention
core, their C-simulation bit-exactness (Linux; skipped where hls4ml's ap_types headers do not build, e.g. Apple
clang), and the two characterized hls4ml limitations (inline silu, rank-3 LUT). Under JAX the inline-silu row changes
form: a rank-2 ``QDense(activation="silu")`` now converts, but the code calls ``nnet::silu``, which hls4ml 1.3.0 does
not define; a rank-3 one raises (the shape of the CLIC heads). Both are pinned.

Part 2 (S7) runs on the org QAT model converted from the hybrid Lightning checkpoint (``s7_fixtures/hybrid``):
- functional assembly of a SiLU head and of SwiGLU nets equals the trained layers bit for bit;
- the renorm ``kept`` export form equals the trained ``kept`` leaf exactly (control: a table lane one fractional bit
  coarser differs);
- every HGQ2 leaf kind of the org model generates hls4ml code, the kinds that do not as trained (the ``kept``
  einsum, "Batch dim mismatch"; ``QDense`` with inline silu) and the inline sigmoid (converts, C simulation differs)
  through their export forms. Controls: kept and silu still fail as trained; the inline sigmoid's C simulation differs
  (Linux);
- the activation export form (silu, sigmoid) deviates from the trained layer by at most half a step of its table lanes (approximate by
  design), and is not exact (so the bound is the claim);
- C simulation of every export form and leaf kind equals keras bit for bit (Linux).
"""

import collections
import importlib.util
import math
import re
import sys
from pathlib import Path

import keras
import numpy as np
import pytest

from hepattn.keras_native.export import (
    activation_export_form,
    build_functional_dense,
    convert_to_hls,
    kept_export_form,
    leaf_model,
    load_keras_model,
    needs_activation_export,
    save_keras_model,
)

HAS_HLS4ML = importlib.util.find_spec("hls4ml") is not None
needs_hls4ml = pytest.mark.skipif(not HAS_HLS4ML, reason="hls4ml not installed")
FIX = Path(__file__).parent / "s7_fixtures/hybrid"

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
    from hgq.layers import QDense  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(150, 32), name="mlp_in")
        x = QDense(16, activation="relu", name="mlp_hidden")(inp)
        return keras.Model(inp, QDense(6, name="mlp_out")(x), name="toy_mlp")


def build_toy_lut_mlp() -> keras.Model:
    from hgq.layers import QDense, QUnaryFunctionLUT  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(32,), name="lut_mlp_in")
        x = QDense(16, name="lut_mlp_hidden")(inp)
        x = QUnaryFunctionLUT(keras.activations.silu, allow_heterogeneous_table=False, name="lut_mlp_silu")(x)
        return keras.Model(inp, QDense(6, name="lut_mlp_out")(x), name="toy_lut_mlp")


def build_toy_attention_core() -> keras.Model:
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
        out = keras.layers.Reshape((seq, dim))(QEinsum("bhqk,bkhd->bqhd", name="core_values")([attn, v]))
        return keras.Model(inp, QDense(dim, name="core_out")(out), name="attention_core")


def csim_or_skip(model: keras.Model, output_dir):
    """Convert with C simulation; skip where the local toolchain cannot build hls4ml's headers."""
    try:
        return convert_to_hls(model, output_dir, compile_csim=True)
    except Exception as err:
        if "Failed to compile project" in str(err):
            pytest.skip(f"hls4ml csim toolchain unavailable on {sys.platform} (ap_types headers do not build, e.g. Apple libc++)")
        raise


def assert_csim_equal(model: keras.Model, hls_model, inputs) -> None:
    y_keras = np.asarray(model(inputs, training=False))
    y_hls = np.asarray(hls_model.predict(inputs if not isinstance(inputs, list) else [np.ascontiguousarray(x) for x in inputs]))
    y_hls = y_hls.reshape(y_keras.shape)
    assert np.array_equal(y_hls, y_keras), f"hls4ml csim differs from keras (max abs diff {np.abs(y_hls - y_keras).max():.3e})"


# ------------------------------------------------------------------------------------- the converted org QAT model


@pytest.fixture(scope="module")
def org():
    from hepattn.experiments.clic import convert_to_native as C  # noqa: PLC0415  (imports torch: test-only)

    _, model, _ = C.convert([FIX / "config.yaml"], FIX / "model.ckpt")
    return model


def leaves_by_kind(model) -> dict[tuple, list]:
    from hgq.layers.core.base import QLayerBase  # noqa: PLC0415
    from hgq.layers.softmax import QSoftmax  # noqa: PLC0415

    softmax_tables = set()
    found = list(model._flatten_layers(include_self=False, recursive=True))  # noqa: SLF001
    for layer in found:
        if isinstance(layer, QSoftmax):
            softmax_tables |= {id(t) for t in layer._flatten_layers(include_self=False, recursive=True)}  # noqa: SLF001
    kinds = collections.defaultdict(list)
    for layer in found:
        if not isinstance(layer, QLayerBase) or id(layer) in softmax_tables:
            continue  # QSoftmax's exp / inv tables convert as part of the QSoftmax
        cfg = layer.get_config()
        act = cfg.get("activation")
        act = act if isinstance(act, str) or act is None else getattr(layer.activation, "__name__", str(act))
        kinds[type(layer).__name__, act, cfg.get("equation")].append(layer)
    return dict(kinds)


def random_inputs(model: keras.Model, rng, batch: int = 4):
    xs = []
    for spec in model.inputs:
        shape = tuple(spec.shape)
        shape = (batch, *shape[1:]) if shape[0] is None else shape
        xs.append(rng.standard_normal(shape).astype("float32"))
    return xs if len(xs) > 1 else xs[0]


# ------------------------------------------------------------------------------------- part 1: mirror of tests/keras/test_export.py


def test_functional_head_matches_orchestrated(org):
    """A SiLU head (classification) assembled from the live layers reproduces the native Dense bit for bit."""
    head = org.tasks[0].net
    shape = tuple(head.hidden[0]._build_shapes_dict["input_shape"][1:])  # noqa: SLF001
    functional = build_functional_dense(head, input_shape=shape, name="classification_head")
    x = np.random.default_rng(0).standard_normal((3, *shape)).astype("float32")
    assert np.array_equal(np.asarray(head(x, training=False)), np.asarray(functional(x, training=False)))


def test_functional_save_load_roundtrip(org, tmp_path):
    head = org.tasks[0].net
    shape = tuple(head.hidden[0]._build_shapes_dict["input_shape"][1:])  # noqa: SLF001
    functional = build_functional_dense(head, input_shape=shape, name="classification_head")
    save_keras_model(functional, tmp_path / "head.keras")
    loaded = load_keras_model(tmp_path / "head.keras")
    x = np.random.default_rng(1).standard_normal((2, *shape)).astype("float32")
    assert np.array_equal(np.asarray(functional(x, training=False)), np.asarray(loaded(x, training=False)))


@needs_hls4ml
def test_hls4ml_mlp_conversion_and_codegen(tmp_path):
    hls_model = convert_to_hls(build_toy_mlp(), tmp_path / "hls_mlp", compile_csim=False)
    assert (tmp_path / "hls_mlp" / "firmware").exists() and hls_model is not None


@needs_hls4ml
def test_hls4ml_lut_activation_conversion_and_codegen(tmp_path):
    hls_model = convert_to_hls(build_toy_lut_mlp(), tmp_path / "hls_lut", compile_csim=False)
    assert (tmp_path / "hls_lut" / "firmware").exists() and hls_model is not None


@needs_hls4ml
def test_hls4ml_attention_core_conversion_and_codegen(tmp_path):
    hls_model = convert_to_hls(build_toy_attention_core(), tmp_path / "hls_core", compile_csim=False)
    assert (tmp_path / "hls_core" / "firmware").exists() and hls_model is not None


@needs_hls4ml
def test_hls4ml_mlp_bit_exact_csim(tmp_path):
    model = build_toy_mlp()
    hls_model = csim_or_skip(model, tmp_path / "hls_mlp_csim")
    assert_csim_equal(model, hls_model, np.random.default_rng(0).standard_normal((256, 150, 32)).astype(np.float32))


@needs_hls4ml
def test_hls4ml_attention_core_bit_exact_csim(tmp_path):
    model = build_toy_attention_core()
    hls_model = csim_or_skip(model, tmp_path / "hls_core_csim")
    assert_csim_equal(model, hls_model, np.random.default_rng(1).standard_normal((64, 8, 16)).astype(np.float32))


def _hls4ml_defines(symbol: str) -> bool:
    import hls4ml  # noqa: PLC0415

    templates = Path(hls4ml.__file__).parent / "templates"
    pattern = re.compile(rf"\b{symbol}\s*\(")
    return any(pattern.search(p.read_text(errors="ignore")) for p in templates.rglob("*.h"))


@needs_hls4ml
def test_hls4ml_characterized_limitation_inline_silu(tmp_path):
    """PINS: an inline silu is not deployable with hls4ml 1.3.0 (see the export module docstring).

    hls4ml's bit-exact pass gives an unknown activation ``np.ones(k)`` (an array as a shape; bit_exact.py:602): rank 3
    (the CLIC heads) raises there, rank 2 followed by another layer fails the next layer's shape assert, and a rank-2
    silu as the last layer converts to code that calls ``nnet::silu``, which no hls4ml template defines. If this
    fails, hls4ml changed: update docs/hgq/FPGA_STATUS.md and ``activation_export_form``.
    """
    from hgq.layers import QDense  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(8, 32))
        rank3 = keras.Model(inp, QDense(6)(QDense(16, activation="silu")(inp)))
    with pytest.raises(TypeError, match="only integer scalar arrays"):
        convert_to_hls(rank3, tmp_path / "hls_silu3", compile_csim=False)
    with toy_scopes():
        inp = keras.Input(shape=(32,))
        rank2 = keras.Model(inp, QDense(6)(QDense(16, activation="silu")(inp)))
    with pytest.raises(AssertionError):
        convert_to_hls(rank2, tmp_path / "hls_silu2", compile_csim=False)
    with toy_scopes():
        inp = keras.Input(shape=(32,))
        last = keras.Model(inp, QDense(16, activation="silu")(inp))
    convert_to_hls(last, tmp_path / "hls_silu_last", compile_csim=False)
    assert "nnet::silu<" in (tmp_path / "hls_silu_last/firmware/myproject.cpp").read_text()
    assert not _hls4ml_defines("silu"), "hls4ml now defines nnet::silu"
    assert _hls4ml_defines("sigmoid"), "control: the template search finds a defined activation"


@needs_hls4ml
def test_hls4ml_characterized_limitation_lut_rank3(tmp_path):
    """PINS: QUnaryFunctionLUT on rank-3 inputs fails to convert (export LUTs per token, rank 2)."""
    from hgq.layers import QDense, QUnaryFunctionLUT  # noqa: PLC0415

    with toy_scopes():
        inp = keras.Input(shape=(8, 32))
        x = QUnaryFunctionLUT(keras.activations.silu, allow_heterogeneous_table=False)(QDense(16)(inp))
        model = keras.Model(inp, QDense(6)(x))
    with pytest.raises(Exception):  # noqa: B017  (HGQ2 / converter error type varies)
        convert_to_hls(model, tmp_path / "hls_lut_rank3", compile_csim=False)


# ------------------------------------------------------------------------------------- part 2: S7 on the org model


def swiglu_nets(model) -> dict[str, object]:
    nets = {"innet0": model.input_nets[0].net, "mask_object": model.tasks[1].object_net, "encoder_l0_ffn": model.encoder.layers[0].dense.fn}
    assert all(net.gate for net in nets.values())
    return nets


@pytest.mark.parametrize("which", ["innet0", "mask_object", "encoder_l0_ffn"])
def test_functional_swiglu_net_matches_orchestrated(org, which):
    net = swiglu_nets(org)[which]
    shape = tuple(net.hidden[0]._build_shapes_dict["input_shape"][1:])  # noqa: SLF001
    functional = build_functional_dense(net, input_shape=shape, name=f"{which}_net")
    x = np.random.default_rng(2).standard_normal((3, *shape)).astype("float32")
    ref = np.asarray(net(x, training=False))
    assert np.array_equal(ref, np.asarray(functional(x, training=False)))
    # control: the product leaf matters (x1 alone through the final layer is another function)
    assert not np.allclose(ref, np.asarray(net.final(net.hidden[0](x, training=False), training=False)), atol=1e-3)


def kept_cases(model) -> dict[str, tuple]:
    enc = model.encoder.layers[1].attn.fn.core  # ProjectCore (bn,znk->bk)
    dec = model.decoder.layers[0].q_ca.fn.core  # AttendCore (bqn,znk->bqk)
    return {"project_k": (enc.kept_k, enc.E), "project_v": (enc.kept_v, enc.F), "attend_k": (dec.kept_k, dec.E), "attend_v": (dec.kept_v, dec.F)}


def masks_for(kept, rng, batch: int = 8) -> np.ndarray:
    shape = tuple(kept._build_shapes_dict["input_shape"][0][1:])  # noqa: SLF001
    return (rng.random((batch, *shape)) < 0.7).astype("float32")


@pytest.mark.parametrize("case", ["project_k", "project_v", "attend_k", "attend_v"])
def test_kept_export_form_equals_the_kept_leaf(org, case):
    kept, proj = kept_cases(org)[case]
    table = proj.energy(training=False)
    export = kept_export_form(kept, table)
    m = masks_for(kept, np.random.default_rng(3))
    ref = np.asarray(kept([m, keras.ops.convert_to_numpy(table)[None]], training=False))
    assert np.array_equal(ref, np.asarray(export(m, training=False)))
    assert ref.std() > 0
    # control: a table lane one fractional bit coarser is another layer
    coarse = kept_export_form(kept, table, name=f"{kept.name}_coarse")
    f_var = next(v for v in coarse.kq.variables if v.path.endswith("/f"))
    f_var.assign(np.asarray(f_var.value) - 1)
    assert not np.array_equal(ref, np.asarray(coarse(m, training=False)))


def export_models(org) -> dict[str, keras.Model]:
    """One functional model per HGQ2 leaf kind of the org model, the kept and inline-silu kinds in export form."""
    models = {}
    for (cls, act, eq), layers in sorted(leaves_by_kind(org).items(), key=str):
        layer = layers[0]
        key = f"{cls}:{act or ''}:{eq or ''}"
        if eq in {"bqn,znk->bqk", "bn,znk->bk"}:
            core = next(c for c in org._flatten_layers(recursive=True) if getattr(c, "kept_k", None) is layer or getattr(c, "kept_v", None) is layer)  # noqa: SLF001
            proj = core.E if core.kept_k is layer else core.F
            export = kept_export_form(layer, proj.energy(training=False))
            spec = keras.Input(shape=tuple(layer._build_shapes_dict["input_shape"][0][1:]))  # noqa: SLF001
            models[f"{key} (export form)"] = keras.Model(spec, export(spec))
        elif needs_activation_export(layer):
            models[f"{key} (export form)"] = activation_export_form(layer)
        else:
            models[key] = leaf_model(layer)
    return models


ORG_KINDS = 19  # measured on the converted fixture model (docs/hgq/port_log/S7.md)


@needs_hls4ml
def test_every_org_leaf_kind_generates_hls4ml_code(org, tmp_path):
    models = export_models(org)
    assert len(models) == ORG_KINDS, sorted(models)
    rng = np.random.default_rng(4)
    for i, (key, model) in enumerate(models.items()):
        model(random_inputs(model, rng), training=True)  # as R5: a training call before conversion
        hls = convert_to_hls(model, tmp_path / f"k{i:02d}", compile_csim=False)
        assert (tmp_path / f"k{i:02d}/firmware/myproject.cpp").is_file(), key
        assert hls is not None


@needs_hls4ml
def test_controls_kept_and_inline_silu_fail_as_trained(org, tmp_path):
    kinds = leaves_by_kind(org)
    kept = next(layers[0] for (cls, _, eq), layers in kinds.items() if eq == "bqn,znk->bqk")
    with pytest.raises(AssertionError, match="Batch dim mismatch"):
        convert_to_hls(leaf_model(kept), tmp_path / "kept", compile_csim=False)
    silu = next(layers[0] for (cls, act, _), layers in kinds.items() if cls == "QDense" and act == "silu")
    assert len(silu._build_shapes_dict["input_shape"]) == 3  # noqa: SLF001  (a per-token head, rank 3)
    with pytest.raises(TypeError, match="only integer scalar arrays"):
        convert_to_hls(leaf_model(silu), tmp_path / "silu", compile_csim=False)


MAX_SLOPE = {"silu": 1.1, "sigmoid": 0.25}  # max |f'|


@pytest.mark.parametrize("act", ["silu", "sigmoid"])
def test_activation_export_form_deviation_is_bounded(org, act):
    """Approximate by design: |export - trained| <= half an input step x max|f'| + half an output step."""
    layer = org.tasks[0].net.hidden[0] if act == "silu" else org.encoder.layers[1].attn.fn.value_residual_mix
    assert needs_activation_export(layer) and layer.activation.__name__ == act
    export = activation_export_form(layer, in_bits=(4, 6), out_bits=(4, 8))
    shape = tuple(layer._build_shapes_dict["input_shape"][1:])  # noqa: SLF001
    x = np.random.default_rng(5).standard_normal((16, *shape)).astype("float32")
    ref, got = np.asarray(layer(x, training=False)), np.asarray(export(x, training=False))
    pre = np.asarray(keras.ops.convert_to_numpy(export.layers[1](x, training=False)))
    inside = np.abs(pre) < 2**4 - 2**-6  # no saturation of the table input
    assert inside.mean() > 0.99
    bound = 2**-7 * MAX_SLOPE[act] + 2**-9 + 1e-6
    dev = np.abs(ref - got)[inside.reshape(ref.shape)]
    assert dev.max() <= bound, (dev.max(), bound)
    assert dev.max() > 0, "the export form equals the trained layer: then it is not approximate and the docs are wrong"


# ------------------------------------------------------------------------------------- C simulation (Linux)


@needs_hls4ml
def test_export_forms_and_leaf_kinds_csim_bit_exact(org, tmp_path):
    """C simulation of every org leaf kind (export forms included) equals keras exactly. Linux only (skips on macOS)."""
    rng = np.random.default_rng(6)
    for i, (key, model) in enumerate(export_models(org).items()):
        model(random_inputs(model, rng), training=True)
        hls = csim_or_skip(model, tmp_path / f"c{i:02d}")
        inputs = random_inputs(model, rng, batch=16)
        if "znk" in key:  # masks
            inputs = (np.abs(inputs) > 0.5).astype("float32")
        assert_csim_equal(model, hls, inputs)


@needs_hls4ml
def test_control_inline_sigmoid_converts_but_csim_differs(org, tmp_path):
    """hls4ml's own sigmoid table is not HGQ2's float sigmoid: why the export form replaces it (Linux)."""
    layer = org.encoder.layers[1].attn.fn.value_residual_mix
    model = leaf_model(layer)
    hls = csim_or_skip(model, tmp_path / "sigmoid")
    x = np.random.default_rng(8).standard_normal((16, *model.inputs[0].shape[1:])).astype("float32")
    y_keras = np.asarray(model(x, training=False))
    assert not np.array_equal(np.asarray(hls.predict(x)).reshape(y_keras.shape), y_keras)


@needs_hls4ml
def test_functional_head_lut_form_csim_bit_exact(org, tmp_path):
    head = org.tasks[0].net
    shape = tuple(head.hidden[0]._build_shapes_dict["input_shape"][1:])  # noqa: SLF001
    functional = build_functional_dense(head, input_shape=shape, name="clf_head_lut", activations="lut")
    hls = csim_or_skip(functional, tmp_path / "head")
    assert_csim_equal(functional, hls, np.random.default_rng(7).standard_normal((32, *shape)).astype("float32"))
    assert math.isfinite(float(np.asarray(functional(np.zeros((1, *shape), "float32"))).sum()))
