"""S6a gate: ``keras_native.predict_writer`` against ``MaskFormer.predict`` and ``PflowPredictionWriter``.

Outputs: the native org model (test width, float) on CLIC dummy events. Torch references in this process: the org
torch tasks' ``predict`` on the same outputs, and ``PflowPredictionWriter.on_test_batch_end`` with a stub trainer
(two batches of two events). Gates: the predictions are equal; the h5 files have the same groups, datasets, structured
dtypes, shapes, chunks, compression and attributes, and equal values; ``load_convert_h5`` reads both the same.
Controls: a renamed field or a changed value fails the comparison; padding events are not written.
"""

from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest
import torch

from hepattn.experiments.clic.predictionwriter import PflowPredictionWriter, load_convert_h5
from hepattn.keras_native import predict_writer as W
from hepattn.utils.scaling import FeatureScaler

from .golden.mf_cases import SCALE_DICT, clic_dummy_batch, instantiate, org_init_args
from .mf_golden_utils import build_mf

NUM_BATCHES, EVENTS = 2, 2


def to_torch(tree):
    if isinstance(tree, dict):
        return {k: to_torch(v) for k, v in tree.items()}
    array = np.asarray(tree)
    return torch.from_numpy(array.astype(np.int64) if array.dtype == np.int32 else array.copy())


def to_numpy(tree):
    if isinstance(tree, dict):
        return {k: to_numpy(v) for k, v in tree.items()}
    return np.asarray(tree)


@pytest.fixture(scope="module")
def batches():
    return [clic_dummy_batch(EVENTS, offset=EVENTS * i) for i in range(NUM_BATCHES)]


@pytest.fixture(scope="module")
def model(batches):
    return build_mf("org", None, batches[0][0])


@pytest.fixture(scope="module")
def torch_tasks():
    return instantiate(org_init_args()["tasks"])


@pytest.fixture(scope="module")
def run(model, batches):
    """Native outputs and predictions per batch (``jit(predict)``)."""
    predict = jax.jit(lambda outputs: W.predict(outputs, model.tasks))
    out = []
    for inputs, targets in batches:
        outputs = model(inputs, training=False)
        out.append((to_numpy(outputs), to_numpy(predict(outputs)), targets))
    return out


def torch_predict(tasks, outputs: dict) -> dict:
    """``MaskFormer.predict`` (models/maskformer.py) with the torch tasks, on torch tensors."""
    return {
        layer: {t.name: t.predict(layer_out[t.name]) for t in tasks if t.name in layer_out}
        for layer, layer_out in outputs.items()
        if layer != "encoder"
    }


def test_predict_equals_torch_task_predict(run, torch_tasks):
    flips = 0
    for outputs, preds, _ in run:
        ref = torch_predict(torch_tasks, to_torch(outputs))
        assert ref.keys() == preds.keys()
        for layer in ref:
            assert ref[layer].keys() == preds[layer].keys(), layer
            for task in ref[layer]:
                assert ref[layer][task].keys() == preds[layer][task].keys(), (layer, task)
                for key, value in ref[layer][task].items():
                    value = value.numpy()
                    if key.endswith("valid_prob") and task == "mask":
                        np.testing.assert_allclose(preds[layer][task][key], value, rtol=0, atol=MASK_PROB_ATOL)
                    elif key == "pflow_node_valid":
                        flips += int((preds[layer][task][key] != value).sum())
                    else:
                        np.testing.assert_array_equal(
                            preds[layer][task][key], value.astype(preds[layer][task][key].dtype), err_msg=f"{layer}/{task}/{key}"
                        )
    assert flips == 0


def test_predict_with_null_class_queries_equals_torch(run, model, torch_tasks):
    """Half of the queries get the null class as argmax (the untrained model predicts it nowhere)."""
    outputs = jax.tree_util.tree_map(np.copy, run[0][0])
    probs = outputs["final"]["classification"]["pflow_class_prob"]
    probs[:, ::2, -1] = probs.max(-1)[:, ::2] + 0.5
    native = to_numpy(W.predict(outputs, model.tasks))["final"]["classification"]
    ref = torch_predict(torch_tasks, to_torch(outputs))["final"]["classification"]
    assert 0 < int((native["pflow_class"] == 5).sum()) < native["pflow_class"].size
    for key, value in ref.items():
        np.testing.assert_array_equal(native[key], value.numpy().astype(native[key].dtype), err_msg=key)


# measured max |sigmoid_jax - sigmoid_torch| of the mask probabilities (S6a log); tolerance 2x
MASK_PROB_ATOL = 2 * 1.1920928955078125e-07


def torch_writer(tmp_path, run, torch_tasks) -> str:
    writer = PflowPredictionWriter()
    writer.writer = None
    writer.trainer = SimpleNamespace(ckpt_path=str(tmp_path / "torch" / "epoch=1.ckpt"))
    writer.batch_size = EVENTS
    writer.test_suff = None
    writer.num_events = NUM_BATCHES * EVENTS
    writer.var_transform = FeatureScaler(SCALE_DICT).transforms
    for i, (outputs, _, targets) in enumerate(run):
        outputs_t = to_torch(outputs)
        preds_t = torch_predict(torch_tasks, outputs_t)
        writer.on_test_batch_end(None, None, (outputs_t, preds_t, {}), (None, to_torch(targets)), i)
    writer.writer.close()
    return str(writer.output_path)


def native_writer(path, run, pad: bool = False) -> str:
    writer = W.PredictionWriter(path, NUM_BATCHES * EVENTS, SCALE_DICT)
    for outputs, preds, targets in run:
        event_number = targets["event_number"]
        event_valid = None
        if pad:  # one padding event per batch, as HostBatches gives them
            outputs, preds, targets = (jax.tree_util.tree_map(lambda x: np.concatenate([x, x[:1]]), t) for t in (outputs, preds, targets))
            event_number = np.concatenate([event_number, [-1]])
            event_valid = np.array([True] * EVENTS + [False])
        writer.write_batch(outputs["final"], preds["final"], targets, event_number, event_valid)
    return str(writer.close())


def h5_layout(path) -> dict:
    layout = {"attrs": {}}
    with h5py.File(path) as f:
        layout["attrs"] = {k: str(v) for k, v in f.attrs.items()}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                layout[name] = (obj.dtype.descr, obj.shape, obj.chunks, obj.compression, dict(obj.attrs))

        f.visititems(visit)
    return layout


# Inverse transforms (sqrt fn: square) in jax vs torch, measured max abs difference over the regression fields; 2x
INVERSE_ATOL = 0.0


def h5_values_equal(a, b) -> list[str]:
    bad = []
    with h5py.File(a) as fa, h5py.File(b) as fb:
        for name in fa:
            da, db = fa[name][:], fb[name][:]
            for field in da.dtype.names:
                x, y = da[field], db[field]
                ok = np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
                if not ok and x.dtype.kind == "f" and name == "regression":
                    ok = np.allclose(x, y, rtol=0, atol=INVERSE_ATOL, equal_nan=True)
                if not ok:
                    bad.append(f"{name}/{field}")
    return bad


@pytest.fixture
def files(tmp_path, run, torch_tasks):
    return torch_writer(tmp_path, run, torch_tasks), native_writer(tmp_path / "native.h5", run)


def test_h5_layout_and_values_equal_the_torch_writer(files):
    ref, ours = files
    layout = h5_layout(ref)
    assert set(layout) == {"attrs", "events", "object_class", "object_masks", "incidence", "regression"}
    assert layout["object_masks"][1] == (NUM_BATCHES * EVENTS, 150, 160)
    assert h5_layout(ours) == layout
    assert h5_values_equal(ref, ours) == []
    for a, b in zip(load_convert_h5(ref), load_convert_h5(ours), strict=True):
        np.testing.assert_array_equal(a, b)


def test_output_path_follows_the_torch_naming(tmp_path):
    writer = PflowPredictionWriter()
    writer.trainer = SimpleNamespace(ckpt_path=str(tmp_path / "epoch=3-step=10.ckpt"))
    writer.test_suff = "x"
    assert W.output_path(tmp_path / "epoch=3-step=10.ckpt", "x") == writer.output_path
    writer.test_suff = None
    assert W.output_path(tmp_path / "epoch=3-step=10.ckpt") == writer.output_path


def test_padding_events_are_not_written(tmp_path, files, run):
    ref, _ = files
    padded = native_writer(tmp_path / "padded.h5", run, pad=True)
    assert h5_layout(padded) == h5_layout(ref)
    assert h5_values_equal(ref, padded) == []


def test_control_a_renamed_field_fails_the_layout_check(tmp_path, files, run, monkeypatch):
    ref, _ = files
    original = W.batch_records

    def renamed(*args, **kwargs):
        rec = original(*args, **kwargs)
        array = rec["regression"].pop("pred_e")
        rec["regression"]["pred_E"] = array.astype(np.dtype([("pred_E", np.float32)]))
        return rec

    monkeypatch.setattr(W, "batch_records", renamed)
    assert h5_layout(native_writer(tmp_path / "renamed.h5", run)) != h5_layout(ref)


def test_control_a_changed_value_fails_the_value_check(tmp_path, files, run):
    ref, _ = files
    changed = [
        (
            o,
            {
                **p,
                "final": {
                    **p["final"],
                    "classification": {**p["final"]["classification"], "pflow_class": (p["final"]["classification"]["pflow_class"] + 1) % 6},
                },
            },
            t,
        )
        for o, p, t in run
    ]
    assert h5_values_equal(ref, native_writer(tmp_path / "changed.h5", changed)) == ["object_class/pflow_class"]
