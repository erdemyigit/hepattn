"""Padded tokens are 0 where the node stream enters a quantized leaf (``encoder.zero_padding``), on the tiny Polaris config.

HGQ2 sets a WRAP quantizer's integer bits from the max over the whole tensor (training mode and ``trace_minmax``), so
padded rows would widen the traced range of every lane they reach. Checks:

- float model: the task heads read node embeddings with zero padded rows; the valid outputs do not change when
  ``zero_padding`` is replaced by the identity (the code before the change), and do not depend on padded inputs;
- quantized model: with large padded node features, the traced integer bits change only in the input net (which reads
  the raw features); with ``zero_padding`` as the identity they change in the encoder, the decoder and the task heads;
- HGQ2's QSoftmax with a mask fills the masked lanes inside the layer (``-1e9``), so its WRAP exp-table input traces
  23 integer bits (the cap of the kif datalane default ``ic = MinMax(-23, 23)``; 1e9 needs 30): why keras_native never
  passes a mask to QSoftmax in the Linformer cores.
"""

import copy

import keras
import numpy as np
import pytest
from hgq.utils.minmax_trace import TrainingFlagWrapper, _reset_minmax  # noqa: PLC2701

from hepattn.experiments.clic.convert_to_native import sample_batch
from hepattn.keras_native import decoder, encoder
from hepattn.keras_native.maskformer import from_config

from .s6b_utils import tiny_config

PADDED_VALUE = 50.0  # the dummy features are N(0, 1): padded rows at 50 dominate every range they reach


def identity(x, valid):  # noqa: ARG001
    return x


@pytest.fixture(scope="module")
def setup():
    rc = tiny_config()
    inputs, _ = sample_batch(rc, 4)
    valid = inputs["node_valid"]
    assert 0 < valid.sum() < valid.size  # padded nodes exist
    loud = dict(inputs)
    loud["node_features"] = np.where(valid[..., None], inputs["node_features"], PADDED_VALUE).astype(np.float32)
    return rc, inputs, loud


def build(rc, quantized: bool, inputs):
    section = copy.deepcopy(rc.model_section)
    if not quantized:
        section["model"]["init_args"].pop("quant")
    keras.utils.set_random_seed(7)
    model = from_config(section)
    model(inputs, training=False)
    # trained-like weights: at init the biases and norm offsets are 0, so a zero row stays 0 through a layer and a
    # missing fill would not show; the quantizers' own variables keep their spec values
    rng = np.random.default_rng(3)
    for w in model.trainable_weights:
        if "quantizer" not in w.path:
            w.assign(rng.normal(0.0, 0.3, w.shape).astype(w.dtype))
    return model


def patch_zero_padding(monkeypatch, fn) -> None:
    monkeypatch.setattr(encoder, "zero_padding", fn)
    monkeypatch.setattr(decoder, "zero_padding", fn)


def valid_outputs(outputs: dict, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Every output leaf, with the node axis (size N, last) restricted to valid nodes (padded mask logits are finfo.min)."""
    flat = {}

    def walk(tree, prefix):
        for key, value in tree.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            else:
                flat[f"{prefix}{key}"] = np.asarray(value)

    walk(outputs, "")
    n = valid.shape[1]
    out = {}
    for key, value in flat.items():
        if value.ndim >= 2 and value.shape[-1] == n and value.shape[0] == valid.shape[0]:
            sel = np.broadcast_to(valid.reshape(valid.shape[0], *([1] * (value.ndim - 2)), n), value.shape)
            out[key] = value[sel]
        else:
            out[key] = value
    return out


def assert_same_valid_outputs(a: dict, b: dict, rtol=1e-5, atol=1e-6) -> None:
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key].astype(np.float64), b[key].astype(np.float64), rtol=rtol, atol=atol, err_msg=key)


def test_float_valid_outputs_are_unchanged_and_ignore_padded_inputs(setup, monkeypatch):
    rc, inputs, loud = setup
    valid = inputs["node_valid"]
    model = build(rc, False, inputs)
    seen = []  # the node embeddings each task call reads (per decoder layer, then final)
    for task in model.tasks:
        monkeypatch.setattr(task, "call", lambda x, *a, _call=task.call, **k: seen.append(np.asarray(x["node_embed"])) or _call(x, *a, **k))
    new = valid_outputs(model(inputs, training=False), valid)
    assert len(seen) > len(model.tasks)  # every task runs on the final embeddings, some also after the decoder layer
    assert all(np.all(embed[~valid] == 0) and np.any(embed[valid] != 0) for embed in seen)
    new_loud = valid_outputs(model(loud, training=False), valid)
    patch_zero_padding(monkeypatch, identity)
    old = valid_outputs(model(inputs, training=False), valid)
    assert len(new) >= 8  # tiny config: 1 decoder layer + final, 5 task outputs each
    assert_same_valid_outputs(new, old)
    assert_same_valid_outputs(new, new_loud)


def traced_i(model, inputs) -> dict[str, np.ndarray]:
    """The integer bits of every WRAP quantizer after ``trace_minmax`` on ``inputs`` (HGQ2's own eager trace)."""
    _reset_minmax(model)
    model(inputs, training=TrainingFlagWrapper("tracing"))
    quantizers = [q for q in model._flatten_layers(include_self=False) if getattr(q, "overflow_mode", None) == "WRAP"]  # noqa: SLF001
    return {q._i.path: np.asarray(q._i.value) for q in quantizers}  # noqa: SLF001


def moved_by_padding(model, inputs, loud) -> list[str]:
    a, b = traced_i(model, inputs), traced_i(model, loud)
    return sorted(path for path in a if not np.array_equal(a[path], b[path]))


INPUT_NET = "/innet0/"  # the input net reads the raw features


def test_padded_rows_set_no_traced_integer_bits_past_the_input_net(setup, monkeypatch):
    rc, inputs, loud = setup
    model = build(rc, True, inputs)
    moved = moved_by_padding(model, inputs, loud)
    assert moved, "the loud padded rows must reach the input net's quantizers (else the check sees nothing)"
    assert all(INPUT_NET in path for path in moved), [p for p in moved if INPUT_NET not in p]
    # control: without the zero fill (the code before the change) the padded rows set integer bits downstream
    patch_zero_padding(monkeypatch, identity)
    leaked = [p for p in moved_by_padding(model, inputs, loud) if INPUT_NET not in p]
    for part in ("encoder", "decoder"):
        assert any(part in p for p in leaked), (part, leaked[:5])


def test_hgq2_qsoftmax_mask_fills_the_exp_table_input_with_1e9():
    from hgq.config import QuantizerConfigScope  # noqa: PLC0415
    from hgq.layers import QSoftmax  # noqa: PLC0415

    with (
        QuantizerConfigScope(place="datalane", default_q_type="kif", i0=4, f0=8),
        QuantizerConfigScope(place="table", default_q_type="kif", i0=2, f0=10),
    ):
        layer = QSoftmax(axis=-1)
    x = np.random.default_rng(0).normal(size=(2, 3, 8)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    mask[..., 5:] = False
    layer.build(x.shape)
    assert layer.exp_table.iq.quantizer.overflow_mode == "WRAP"

    def exp_input_i(**kwargs) -> float:
        _reset_minmax(layer)
        layer(x, training=TrainingFlagWrapper("tracing"), **kwargs)
        return float(np.max(np.asarray(layer.exp_table.iq.quantizer._i.value)))  # noqa: SLF001

    assert exp_input_i(mask=mask) == 23  # the ic cap; 1e9 + max needs 30
    assert exp_input_i() < 5  # max - x of N(0, 1) scores


def test_decoder_kv_side_ignores_padded_rows(setup, monkeypatch):
    """The decoder loop zeroes the node stream, so the whole-model check cannot see the kv side's own fill (kv_ca and
    kv_ffn take norm(stream) = the norm's bias at a zero row). Here a layer gets loud padded kv rows directly.
    """
    rc, inputs, _ = setup
    model = build(rc, True, inputs)
    layer = model.decoder.layers[0]
    assert layer.kv_ca is not None
    valid = inputs["node_valid"]
    rng = np.random.default_rng(1)
    b, n = valid.shape
    q = rng.normal(size=(b, model.decoder.num_queries, model.dim)).astype(np.float32)
    kv = rng.normal(size=(b, n, model.dim)).astype(np.float32)
    attn_mask = rng.random((b, q.shape[1], n)) < 0.5

    def moved() -> list[str]:
        traced = []
        for rows in (kv, np.where(valid[..., None], kv, PADDED_VALUE).astype(np.float32)):
            _reset_minmax(layer)
            layer(q, rows, attn_mask=attn_mask, kv_mask=valid, training=TrainingFlagWrapper("tracing"))
            quantizers = [x for x in layer._flatten_layers(include_self=False) if getattr(x, "overflow_mode", None) == "WRAP"]  # noqa: SLF001
            traced.append({x._i.path: np.asarray(x._i.value) for x in quantizers})  # noqa: SLF001
        return [p for p in traced[0] if not np.array_equal(traced[0][p], traced[1][p])]

    kv_side = [p for p in moved() if "_kv_" in p]
    assert not kv_side, kv_side
    assert any("_q_ca_" in p for p in moved())  # the q_ca keys read the raw kv here (the decoder loop zeroes it)
    patch_zero_padding(monkeypatch, identity)
    assert any("_kv_ca_" in p for p in moved()) and any("_kv_ffn_" in p for p in moved())
