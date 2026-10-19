"""S6a gate: the data adapter (``keras_native.data`` + ``experiments.clic.native_data``) on CLIC dummy events.

- train: ``drop_last`` (10 events, batch 4 -> 2 batches); control: without it the short batch raises;
- every batch has static shapes and only int32 / float32 / bool device arrays; control: the torch batch has int64;
- val / test: the last batch is padded with invalid events to the fixed size, ``event_valid`` marks the real ones, the
  real events are unchanged, ``event_number`` stays int64 on the host; the native forward on the padded batch gives
  the real events the outputs of an unpadded batch, and a finite matched loss;
- one epoch of padded val batches compiles the jitted eval forward + ``predict`` once (plan S6 gate, eval and
  predict part); control: the unpadded short batch compiles a second executable;
- order: the train permutation depends on (seed, epoch) only, and ``start_batch`` resumes inside an epoch;
- workers: never forked (JAX is multithreaded); spawned persistent workers give the in-process batches over two
  epochs and a resume, with no fork warning after JAX has a backend; control: a fork context is refused, and
  forking after JAX init warns;
- a batch with the shapes of the first one but another dtype raises (a new compile).
"""

import dataclasses
import multiprocessing
import warnings

import jax
import numpy as np
import pytest
import torch

from hepattn.experiments.clic.native_data import EpochSampler, SplitBatches, host_batches, make_dataset, make_loader
from hepattn.keras_native.config import load_config
from hepattn.keras_native.data import HostBatches, pad_events
from hepattn.keras_native.objective import clic_specs, matched_loss
from hepattn.keras_native.predict_writer import predict

from .golden.mf_cases import CONFIGS
from .mf_golden_utils import build_mf

NUM_EVENTS, BATCH = 10, 4
STATIC = {np.dtype("int32"), np.dtype("float32"), np.dtype("bool")}


@pytest.fixture(scope="module")
def data_config():
    cfg = load_config([CONFIGS / "base.yaml", CONFIGS / "linformer.yaml"]).data
    return dataclasses.replace(cfg, batch_size=BATCH, num_workers=0)


@pytest.fixture(scope="module")
def dataset(data_config):
    return make_dataset(data_config, "val", dummy_data=True, num_events=NUM_EVENTS)


def collect(batches) -> list:
    return list(batches)


def test_train_batches_drop_last_and_are_static_int32(data_config, dataset):
    batches = collect(host_batches(data_config, "train", dataset=dataset))
    assert len(batches) == NUM_EVENTS // BATCH == 2
    shapes = None
    for batch in batches:
        assert batch.event_valid.all()
        arrays = {**{f"in/{k}": v for k, v in batch.inputs.items()}, **{f"tgt/{k}": v for k, v in batch.targets.items()}}
        assert all(isinstance(v, jax.Array) for v in arrays.values())
        assert {np.dtype(v.dtype) for v in arrays.values()} <= STATIC, {k: v.dtype for k, v in arrays.items()}
        assert all(v.shape[0] == BATCH for v in arrays.values())
        current = {k: v.shape for k, v in arrays.items()}
        shapes = shapes or current
        assert current == shapes
        assert batch.host["event_number"].dtype == np.int64
        assert "event_number" not in batch.targets and "event_number" not in batch.inputs  # host only, not sent
    assert batches[0].inputs["node_features"].shape == (BATCH, 160, 27)
    assert batches[0].targets["particle_class"].dtype == np.int32


def test_control_torch_batches_hold_int64_and_a_short_last_batch(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False)
    raw = list(loader)
    assert raw[0][1]["particle_class"].dtype == torch.int64
    assert raw[-1][0]["node_valid"].shape[0] == NUM_EVENTS % BATCH
    with pytest.raises(ValueError, match="drop_last"):
        collect(HostBatches(loader, BATCH, pad_to_batch=False, device_put=False))


def test_val_batches_are_padded_with_invalid_events(data_config, dataset):
    batches = collect(host_batches(data_config, "val", dataset=dataset, device_put=False))
    assert len(batches) == 3
    assert [b.num_real for b in batches] == [4, 4, 2]
    assert all(v.shape[0] == BATCH for b in batches for v in (*b.inputs.values(), *b.targets.values()))
    last = batches[-1]
    assert last.event_valid.tolist() == [True, True, False, False]
    pad = ~last.event_valid
    assert not last.inputs["node_valid"][pad].any() and not last.targets["particle_valid"][pad].any()
    assert not last.targets["particle_node_valid"][pad].any()
    assert (last.targets["particle_class"][pad] == 5).all()
    assert np.isnan(last.targets["particle_e"][pad]).all()
    assert (last.targets["particle_incidence"][pad] == 0).all()
    assert (last.inputs["node_features"][pad] == 0).all()
    assert last.host["event_number"].tolist() == [8, 9, -1, -1]
    assert "event_number" not in last.targets
    # the real events are the dataset's, unchanged
    inputs, targets = dataset[9]
    np.testing.assert_array_equal(last.inputs["node_features"][1], inputs["node_features"].numpy())
    np.testing.assert_array_equal(last.targets["particle_class"][1], targets["particle_class"].numpy())
    np.testing.assert_array_equal(last.targets["particle_e"][1], targets["particle_e"].numpy())
    # control: a full batch is not padded
    assert batches[0].event_valid.all()


def test_padded_events_do_not_change_the_real_events(data_config, dataset):
    """Eval forward of a padded batch: real events equal the unpadded batch's outputs; the matched loss is finite."""
    padded = collect(host_batches(data_config, "val", dataset=dataset, device_put=False))[-1]
    real = {k: v[:2] for k, v in padded.inputs.items()}
    model = build_mf("org", None, real)
    out_padded = jax.tree_util.tree_flatten_with_path(model(padded.inputs, training=False))[0]
    out_real = dict(jax.tree_util.tree_flatten_with_path(model(real, training=False))[0])
    worst = 0.0
    for path, value in out_padded:
        value = np.asarray(value)
        assert np.isfinite(value[2:]).all() or value.dtype == bool, jax.tree_util.keystr(path)
        ref = np.asarray(out_real[path])
        if value.dtype == bool:
            np.testing.assert_array_equal(value[:2], ref)
        else:
            worst = max(worst, float(np.max(np.abs(value[:2].astype(np.float64) - ref))))
    assert worst <= 2 * MEASURED_BATCH_DIFF, worst
    total, _, _ = matched_loss(model(padded.inputs, training=False), padded.targets, clic_specs())
    assert np.isfinite(float(total))


# measured max |padded - unpadded| over every output of the real events (S6a log); 2x below
MEASURED_BATCH_DIFF = 0.0


def test_train_order_depends_on_seed_and_epoch_and_resumes(dataset):
    def order(**kwargs):
        return list(EpochSampler(NUM_EVENTS, batch_size=BATCH, shuffle=True, drop_last=True, **kwargs))

    assert order(seed=1, epoch=0) == order(seed=1, epoch=0)
    assert order(seed=1, epoch=0) != order(seed=1, epoch=1)
    assert len(order(seed=1, epoch=0)) == 8
    assert order(seed=1, epoch=0, start_batch=1) == order(seed=1, epoch=0)[BATCH:]
    loader = make_loader(dataset, "train", batch_size=BATCH, seed=3, epoch=2, start_batch=1)
    assert len(loader) == 1
    assert loader.sampler.order() == EpochSampler(NUM_EVENTS, batch_size=BATCH, shuffle=True, seed=3, epoch=2).order()
    assert list(make_loader(dataset, "val", batch_size=BATCH).sampler) == list(range(NUM_EVENTS))


def test_int64_out_of_int32_range_raises():
    loader = [({"x_valid": np.ones((2, 3), bool), "x_big": np.array([[2**31, 0, 0], [0, 0, 0]], np.int64)}, {})]
    with pytest.raises(OverflowError, match="x_big"):
        collect(HostBatches(loader, 2, pad_to_batch=False, device_put=False))
    ok = [({"x_valid": np.ones((2, 3), bool), "x_big": np.array([[2**31 - 1, 0, 0], [0, 0, 0]], np.int64)}, {})]
    assert collect(HostBatches(ok, 2, pad_to_batch=False, device_put=False))[0].inputs["x_big"].dtype == np.int32


def test_float64_is_cast_to_float32_on_the_host():
    """Without ``device_put`` (where jax's x64-off downcast does not apply) the adapter itself casts float64."""
    loader = [({"x_valid": np.ones((2, 3), bool), "x_e": np.full((2, 3), 0.1, np.float64)}, {"t": np.zeros(2, np.float64)})]
    batch = collect(HostBatches(loader, 2, pad_to_batch=False, device_put=False))[0]
    assert batch.inputs["x_e"].dtype == batch.targets["t"].dtype == np.float32
    assert batch.inputs["x_e"][0, 0] == np.float32(0.1)
    assert batch.inputs["x_valid"].dtype == np.bool_  # other dtypes pass through


@pytest.mark.parametrize("second", [np.zeros((2, 4), np.float32), np.zeros((2, 3), np.int32), np.zeros((2, 3), bool)])
def test_a_new_batch_shape_or_dtype_raises(second):
    first = np.zeros((2, 3), np.float32)
    collect(HostBatches([({"x": first}, {}), ({"x": first.copy()}, {})], 2, pad_to_batch=False, device_put=False))  # control
    with pytest.raises(ValueError, match="new compile"):
        collect(HostBatches([({"x": first}, {}), ({"x": second}, {})], 2, pad_to_batch=False, device_put=False))


def test_pad_values():
    tree = {
        "particle_valid": np.ones((1, 2), bool),
        "particle_class": np.zeros((1, 2), np.int64),
        "particle_e": np.ones((1, 2), np.float32),
        "particle_incidence": np.ones((1, 2, 3), np.float32),
        "node_features": np.ones((1, 3, 4), np.float32),
        "event_number": np.array([7], np.int64),
    }
    out = pad_events(tree, 2)
    assert out["particle_valid"][1].tolist() == [False, False]
    assert out["particle_class"][1].tolist() == [5, 5]
    assert np.isnan(out["particle_e"][1]).all()
    assert (out["particle_incidence"][1] == 0).all() and (out["node_features"][1] == 0).all()
    assert out["event_number"].tolist() == [7, -1]
    with pytest.raises(ValueError, match="larger"):
        pad_events(tree, 0)


def test_one_val_epoch_compiles_eval_and_predict_once(data_config, dataset):
    batches = collect(host_batches(data_config, "val", dataset=dataset))
    model = build_mf("org", None, {k: np.asarray(v) for k, v in batches[0].inputs.items()})
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]

    @jax.jit
    def evaluate(tv, ntv, inputs):
        outputs, _ = model.stateless_call(tv, ntv, inputs, training=False)
        return predict(outputs, model.tasks)

    for batch in batches:
        jax.block_until_ready(evaluate(tv, ntv, batch.inputs))
    assert evaluate._cache_size() == 1  # noqa: SLF001
    # control: the torch loader's short last batch, unpadded
    loader = make_loader(dataset, "val", batch_size=BATCH)
    short = {k: np.asarray(v) for k, v in list(loader)[-1][0].items()}
    short = {k: v.astype(np.int32) if v.dtype == np.int64 else v for k, v in short.items()}
    jax.block_until_ready(evaluate(tv, ntv, short))
    assert evaluate._cache_size() == 2  # noqa: SLF001


def test_workers_are_never_forked(dataset):
    assert make_loader(dataset, "train", batch_size=BATCH).multiprocessing_context is None  # no workers, no processes
    for split in ("train", "val"):
        loader = make_loader(dataset, split, batch_size=BATCH, num_workers=2)
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        assert loader.persistent_workers
    with pytest.raises(ValueError, match="must not be forked"):  # control
        make_loader(dataset, "train", batch_size=BATCH, num_workers=2, multiprocessing_context="fork")


def numpy_batches(batches) -> list[dict]:
    return [{**b.inputs, **{f"t/{k}": v for k, v in b.targets.items()}, "valid": b.event_valid} for b in batches]


def assert_same_batches(a: list[dict], b: list[dict]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.keys() == y.keys()
        for key in x:
            np.testing.assert_array_equal(x[key], y[key], err_msg=key)


def fork_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if "fork" in str(w.message)]


def test_spawned_workers_give_the_in_process_batches_without_a_fork_warning(data_config, dataset):
    """After JAX has a backend: two epochs and a resume through one persistent-worker loader equal ``num_workers=0``."""
    jax.block_until_ready(jax.device_put(np.ones(3)))  # JAX threads running, its at-fork hook installed
    epochs = [(0, 0), (1, 0), (1, 1)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        workers = SplitBatches(dataclasses.replace(data_config, num_workers=2), "train", dataset=dataset, device_put=False)
        with_workers = [numpy_batches(workers.epoch(epoch, start)) for epoch, start in epochs]
    assert fork_warnings(caught) == [], fork_warnings(caught)
    in_process = SplitBatches(data_config, "train", dataset=dataset, device_put=False)
    expected = [numpy_batches(in_process.epoch(epoch, start)) for epoch, start in epochs]
    for got, ref in zip(with_workers, expected, strict=True):
        assert_same_batches(got, ref)
    assert [len(b) for b in expected] == [2, 2, 1]
    assert not np.array_equal(expected[0][0]["node_features"], expected[1][0]["node_features"])  # epochs differ


def test_control_forking_after_jax_init_warns(dataset):
    """What spawn avoids: a fork-started worker after JAX init triggers JAX's deadlock warning (macOS and Linux)."""
    jax.block_until_ready(jax.device_put(np.ones(3)))
    sampler = EpochSampler(len(dataset), batch_size=BATCH, shuffle=False, seed=0)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, sampler=sampler, num_workers=1, multiprocessing_context=multiprocessing.get_context("fork")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        next(iter(loader))
    assert any("JAX is multithreaded" in m for m in fork_warnings(caught)), [str(w.message) for w in caught]
