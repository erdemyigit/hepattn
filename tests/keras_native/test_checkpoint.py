"""S6a gates for ``keras_native.checkpoint`` (plan section 3.9).

Model: org ``base.yaml`` + ``linformer.yaml`` at the test width (``mf_cases`` variant ``org``), HGQ2 ``high`` spec,
AdamW with clipping; state from a jitted stateless train step (matched loss + HGQ2 penalty) on CLIC dummy events.

Gates:
- save -> load into an independently built model and optimizer -> every array bitwise equal (values and dtypes);
- resume: one step from the loaded state equals the uninterrupted second step bitwise on CPU (plan S6 gate); the steps
  change every state group (non-vacuous), and a 1e-3 change of one loaded weight makes the comparison fail (control);
- keys are equal for two independently built models, although ``variable.path`` differs (control);
- a corrupted, truncated, partial (no sidecar), mismatched or foreign checkpoint is rejected, as are a stored dtype
  other than the model's and metadata other than the sidecar's; an interrupted write leaves the previous checkpoint
  intact and no temporary file; an existing checkpoint is not overwritten unless asked; renames are followed by a
  directory fsync;
- optimizer slots of variables whose slot names are prefixes of each other map to their own variables;
- export: ``.weights.h5`` of the state loads into a fresh model bit-exactly;
- policy / manager: every N steps, last, best on ``val/loss``, retention, a second save at one step under a new name.
"""

import json
import stat
import zipfile

import jax
import keras
import numpy as np
import pytest

from hepattn.keras_native import checkpoint as C
from hepattn.keras_native.objective import clic_specs, matched_loss

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .mf_golden_utils import build_mf


def real_like(targets: dict) -> dict:
    """No target at padded nodes, as real CLIC data (dummy data puts 1s there; S4.md)."""
    targets = dict(targets)
    node_valid = targets["node_valid"][:, None, :]
    targets["particle_node_valid"] = targets["particle_node_valid"] & node_valid
    targets["particle_incidence"] = np.where(node_valid, targets["particle_incidence"], 0.0).astype(np.float32)
    return targets


class Fresh:
    """An independently built model and optimizer (other auto-generated layer names than an earlier build)."""

    def __init__(self, inputs):
        self.model = build_mf("org", QUANT_SPECS["high"], inputs)
        self.optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, global_clipnorm=0.1)
        self.optimizer.build(self.model.trainable_variables)


class Run(Fresh):
    """``Fresh`` plus a jitted stateless train step (matched loss + HGQ2 penalty, AdamW)."""

    def __init__(self, inputs):
        super().__init__(inputs)
        specs = clic_specs()
        model, optimizer = self.model, self.optimizer

        def loss_fn(tv, ntv, inputs, targets):
            out, ntv, qlosses = model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)
            total, _, _ = matched_loss(out, targets, specs)
            return total + sum(qlosses), ntv

        @jax.jit
        def step(tv, ntv, ov, inputs, targets):
            (loss, ntv), grads = jax.value_and_grad(loss_fn, has_aux=True)(tv, ntv, inputs, targets)
            tv, ov = optimizer.stateless_apply(ov, grads, tv)
            return tv, ntv, ov, loss

        self.step = step

    def run(self, state, batch):
        tv, ntv, ov, loss = self.step(*state, *batch)
        return host((tv, ntv, ov)), float(loss)


def host(state):
    return tuple([np.array(x) for x in group] for group in state)


def first_difference(a, b) -> str | None:
    for name, ga, gb in zip(C.GROUPS, a, b, strict=True):
        assert len(ga) == len(gb), name
        for i, (x, y) in enumerate(zip(ga, gb, strict=True)):
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                return f"{name}[{i}]"
    return None


def changed_groups(a, b) -> set[str]:
    return {name for name, ga, gb in zip(C.GROUPS, a, b, strict=True) if any(not np.array_equal(x, y) for x, y in zip(ga, gb, strict=True))}


@pytest.fixture(scope="module")
def batches():
    return [(inputs, real_like(targets)) for inputs, targets in (clic_dummy_batch(2, offset=2 * i) for i in range(2))]


@pytest.fixture(scope="module")
def uninterrupted(batches):
    run = Run(batches[0][0])
    s0 = C.current_state(run.model, run.optimizer)
    s1, loss1 = run.run(s0, batches[0])
    s2, loss2 = run.run(s1, batches[1])
    return {"run": run, "s0": s0, "s1": s1, "s2": s2, "losses": (loss1, loss2)}


@pytest.fixture(scope="module")
def fresh(batches, uninterrupted):  # noqa: ARG001  (built after the first model, so its layer names differ)
    return Fresh(batches[0][0])


META = {
    "step": 1,
    "epoch": 0,
    "beta_controller": {"beta": 2.5e-9, "integral": -0.25},
    "data_position": {"epoch": 0, "batch": 1},
    "config_hash": "abc",
}


@pytest.fixture
def saved(tmp_path, uninterrupted):
    run = uninterrupted["run"]
    return C.save_checkpoint(tmp_path / "ck.npz", run.model, run.optimizer, uninterrupted["s1"], META)


def test_steps_change_every_state_group(uninterrupted):
    """Non-vacuous: the step changes weights, quantizer state and optimizer slots, so equality below means something."""
    assert changed_groups(uninterrupted["s0"], uninterrupted["s1"]) == {"tv", "ntv", "ov"}
    assert changed_groups(uninterrupted["s1"], uninterrupted["s2"]) == {"tv", "ntv", "ov"}
    assert all(np.isfinite(loss) for loss in uninterrupted["losses"])


def test_save_load_is_bitwise_into_a_fresh_model(saved, uninterrupted, fresh):
    state, meta = C.load_checkpoint(saved, fresh.model, fresh.optimizer, expect_config_hash="abc")
    assert first_difference(state, uninterrupted["s1"]) is None
    assert meta == {"format_version": C.FORMAT_VERSION, **META}
    assert C.read_meta(saved) == meta


def test_resume_equals_the_uninterrupted_run(saved, uninterrupted, fresh, batches):
    """Load in the fresh model's variable order, take the second step. The step is the compiled one of the first
    model (compiling the fresh model's own step costs 145 s on CPU, measured); the fresh model with the loaded state
    is checked to be the same function eagerly.
    """
    state, _ = C.load_checkpoint(saved, fresh.model, fresh.optimizer)
    resumed, _ = uninterrupted["run"].run(state, batches[1])
    assert first_difference(resumed, uninterrupted["s2"]) is None
    C.assign_state(fresh.model, fresh.optimizer, state)
    run = uninterrupted["run"]
    C.assign_state(run.model, run.optimizer, uninterrupted["s1"])
    out_fresh = jax.tree_util.tree_leaves(fresh.model(batches[1][0], training=False))
    out_first = jax.tree_util.tree_leaves(run.model(batches[1][0], training=False))
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(out_fresh, out_first, strict=True))


def test_control_resume_check_sees_a_small_weight_change(saved, uninterrupted, fresh, batches):
    (tv, ntv, ov), _ = C.load_checkpoint(saved, fresh.model, fresh.optimizer)
    kernel = next(i for i, v in enumerate(fresh.model.trainable_variables) if v.path.endswith("kernel"))
    tv = [x.copy() for x in tv]
    tv[kernel].flat[0] += 1e-3
    resumed, _ = uninterrupted["run"].run((tv, ntv, ov), batches[1])
    assert first_difference(resumed, uninterrupted["s2"]) is not None
    C.assign_state(fresh.model, None, (tv, ntv, []))
    run = uninterrupted["run"]
    C.assign_state(run.model, None, (uninterrupted["s1"][0], uninterrupted["s1"][1], []))
    out_fresh = jax.tree_util.tree_leaves(fresh.model(batches[1][0], training=False))
    out_first = jax.tree_util.tree_leaves(run.model(batches[1][0], training=False))
    assert not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(out_fresh, out_first, strict=True))


def test_keys_are_stable_across_independent_builds(uninterrupted, fresh):
    a, b = uninterrupted["run"], fresh
    keys_a, keys_b = C.state_keys(a.model, a.optimizer), C.state_keys(b.model, b.optimizer)
    assert keys_a == keys_b
    for group in C.GROUPS:
        assert len(set(keys_a[group])) == len(keys_a[group]), group
    assert len(keys_a["ov"]) == len(a.optimizer.variables) > 2 * len(a.model.trainable_variables)
    # control: the raw paths and optimizer names are not stable
    paths = sum(va.path != vb.path for va, vb in zip(a.model.variables, b.model.variables, strict=True))
    names = sum(va.name != vb.name for va, vb in zip(a.optimizer.variables, b.optimizer.variables, strict=True))
    assert paths > 100 and names > 100, (paths, names)


def test_optimizer_slots_are_keyed_by_their_variable(uninterrupted):
    run = uninterrupted["run"]
    keys = C.state_keys(run.model, run.optimizer)
    tv_keys = set(keys["tv"])
    slots = [k for k in keys["ov"] if "::" in k]
    assert len(slots) == 2 * len(tv_keys)  # AdamW: momentum and velocity per trainable variable
    assert {k.rsplit("::", 1)[0] for k in slots} == tv_keys
    assert {k.rsplit("::", 1)[1] for k in slots} == {"momentum", "velocity"}


def flip_byte(path, offset_fraction=0.5):
    data = bytearray(path.read_bytes())
    data[int(len(data) * offset_fraction)] ^= 0xFF
    path.write_bytes(bytes(data))


def test_corrupted_npz_is_rejected(saved, fresh):
    flip_byte(saved)
    with pytest.raises(C.CheckpointError, match="sha256"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer)


def test_truncated_npz_with_a_matching_sidecar_is_rejected(saved, fresh):
    """Even if the sidecar hash were rewritten, the damaged zip does not load."""
    saved.write_bytes(saved.read_bytes()[: saved.stat().st_size // 2])
    side = C.sidecar_path(saved)
    meta = json.loads(side.read_text())
    meta["sha256"] = C._sha256(saved)  # noqa: SLF001
    side.write_text(json.dumps(meta))
    with pytest.raises(C.CheckpointError, match="cannot read"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer)


def test_crc_damage_inside_the_zip_is_rejected(saved, fresh):
    """A flipped byte inside an array member with a rewritten sidecar hash: the zip CRC catches it."""
    with zipfile.ZipFile(saved) as zf:
        info = max(zf.infolist(), key=lambda i: i.file_size)
    flip_byte(saved, (info.header_offset + info.compress_size // 2 + 100) / saved.stat().st_size)
    side = C.sidecar_path(saved)
    meta = json.loads(side.read_text())
    meta["sha256"] = C._sha256(saved)  # noqa: SLF001
    side.write_text(json.dumps(meta))
    with pytest.raises(C.CheckpointError, match="cannot read"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer)


def test_missing_or_foreign_sidecar_is_rejected(tmp_path, saved, uninterrupted, fresh):
    run = uninterrupted["run"]
    other = C.save_checkpoint(tmp_path / "other.npz", run.model, run.optimizer, uninterrupted["s2"], {**META, "step": 2})
    C.sidecar_path(saved).write_text(C.sidecar_path(other).read_text())
    with pytest.raises(C.CheckpointError, match="sha256"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer)
    C.sidecar_path(saved).unlink()
    with pytest.raises(C.CheckpointError, match="incomplete"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer)


def test_config_hash_mismatch_is_rejected(saved, fresh):
    with pytest.raises(C.CheckpointError, match="written for config"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer, expect_config_hash="other")


def test_checkpoint_of_another_model_is_rejected(saved, batches):
    float_model = build_mf("org", None, batches[0][0])  # no quantizer variables: other keys
    with pytest.raises(C.CheckpointError, match="keys differ"):
        C.load_checkpoint(saved, float_model)
    unsorted = build_mf("org", QUANT_SPECS["high"], batches[0][0], sort=False)
    # the same variables without input sorting (sorting has no variables): the layout loads, so a config change of
    # this kind is caught only by expect_config_hash (test_config_hash_mismatch_is_rejected)
    C.load_checkpoint(saved, unsorted)


def test_interrupted_write_keeps_the_previous_checkpoint(saved, uninterrupted, fresh, monkeypatch):
    before = saved.read_bytes()
    run = uninterrupted["run"]

    def broken_savez(f, **_arrays):
        f.write(b"PK partial")
        raise OSError("disk full")

    monkeypatch.setattr(C.np, "savez", broken_savez)
    with pytest.raises(OSError, match="disk full"):
        C.save_checkpoint(saved, run.model, run.optimizer, uninterrupted["s2"], META, overwrite=True)
    monkeypatch.undo()
    assert saved.read_bytes() == before
    assert sorted(p.name for p in saved.parent.iterdir()) == ["ck.json", "ck.npz"]
    C.load_checkpoint(saved, fresh.model, fresh.optimizer)


def rewrite_npz(path, edit_arrays, edit_sidecar=None) -> None:
    """Rewrite a checkpoint's arrays and keep its sidecar hash (and, with ``edit_sidecar``, more) consistent."""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    edit_arrays(arrays)
    with path.open("wb") as f:
        np.savez(f, **arrays)
    side = C.sidecar_path(path)
    sidecar = json.loads(side.read_text())
    sidecar["sha256"] = C._sha256(path)  # noqa: SLF001
    if edit_sidecar is not None:
        edit_sidecar(sidecar)
    side.write_text(json.dumps(sidecar))


def test_stored_dtype_other_than_the_models_is_rejected(saved, fresh):
    """E.g. an fp32 checkpoint for a later bf16 model: same shape, another dtype, a consistent sidecar."""
    name = f"tv/{C.state_keys(fresh.model)['tv'][0]}"

    def cast(arrays):
        arrays[name] = arrays[name].astype(np.float16)

    def layout(sidecar):
        group, key = name.split("/", 1)
        sidecar["layout"][group][key][1] = np.dtype(np.float16).str

    rewrite_npz(saved, cast, layout)
    with pytest.raises(C.CheckpointError, match="float16"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer)


def test_metadata_other_than_the_sidecars_is_rejected(saved, fresh):
    C.load_checkpoint(saved, fresh.model, fresh.optimizer)  # control: the rewrite alone keeps it loadable
    rewrite_npz(saved, lambda _arrays: None)
    C.load_checkpoint(saved, fresh.model, fresh.optimizer)

    def other_step(arrays):
        meta = json.loads(str(arrays[C.META_KEY]))
        arrays[C.META_KEY] = np.array(json.dumps({**meta, "step": 99}, sort_keys=True))

    rewrite_npz(saved, other_step)
    with pytest.raises(C.CheckpointError, match="metadata differs"):
        C.load_checkpoint(saved, fresh.model, fresh.optimizer)


def test_save_does_not_overwrite_a_checkpoint(saved, uninterrupted, fresh):
    run = uninterrupted["run"]
    with pytest.raises(FileExistsError):
        C.save_checkpoint(saved, run.model, run.optimizer, uninterrupted["s2"], META)
    C.save_checkpoint(saved, run.model, run.optimizer, uninterrupted["s2"], {**META, "step": 2}, overwrite=True)
    state, meta = C.load_checkpoint(saved, fresh.model, fresh.optimizer)
    assert meta["step"] == 2 and first_difference(state, uninterrupted["s2"]) is None


def test_renames_are_made_durable_by_a_directory_fsync(tmp_path, monkeypatch):
    synced = []
    real_fsync = C.os.fsync

    def fsync(fd):
        synced.append(stat.S_ISDIR(C.os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(C.os, "fsync", fsync)
    with C._atomic(tmp_path / "x.json", "w") as f:  # noqa: SLF001
        f.write("{}")
    assert synced == [False, True]  # the file, then its directory after the rename


class TwoWeights(keras.layers.Layer):
    """Weights ``b`` and ``b_c``: the optimizer slot name of ``b`` is a prefix of those of ``b_c``."""

    def build(self, _):
        self.b = self.add_weight(name="b", shape=(2,))
        self.b_c = self.add_weight(name="b_c", shape=(3,))

    def call(self, x):
        return x * keras.ops.sum(self.b) + keras.ops.sum(self.b_c)


class Tiny(keras.Model):
    def __init__(self):
        super().__init__(name="tiny")
        self.inner = TwoWeights(name="a")

    def call(self, x):
        return self.inner(x)


def test_optimizer_slots_take_the_longest_matching_variable():
    model = Tiny()
    model(np.zeros((1, 1), np.float32))
    optimizer = keras.optimizers.AdamW()
    optimizer.build(model.trainable_variables)
    names = [v.name for v in optimizer.variables]
    assert "tiny_a_b_c_momentum" in names and "tiny_a_b_momentum" in names  # keras' slot naming, a prefix pair
    keys = C.state_keys(model, optimizer)["ov"]
    slots = {k for k in keys if "::" in k}
    assert slots == {f"tiny/a/{w}::{slot}" for w in ("b", "b_c") for slot in ("momentum", "velocity")}


def test_export_weights_loads_into_a_fresh_model(tmp_path, uninterrupted, batches):
    run = uninterrupted["run"]
    path = C.export_weights(run.model, uninterrupted["s2"], tmp_path / "model.weights.h5")
    target = build_mf("org", QUANT_SPECS["high"], batches[0][0])
    tv, ntv, _ = C.current_state(target)
    assert first_difference((tv, ntv, []), (uninterrupted["s2"][0], uninterrupted["s2"][1], [])) is not None  # control
    target.load_weights(path)
    tv, ntv, _ = C.current_state(target)
    assert first_difference((tv, ntv, []), (uninterrupted["s2"][0], uninterrupted["s2"][1], [])) is None


def test_policy_every_n_steps_and_best():
    policy = C.CheckpointPolicy(every_n_steps=1000, monitor="val/loss", mode="min")
    assert [s for s in (0, 1, 999, 1000, 1500, 2000) if policy.step_due(s)] == [1000, 2000]
    assert policy.is_better({"val/loss": 5.0})
    policy.best_value = 5.0
    assert not policy.is_better({"val/loss": 5.0})
    assert policy.is_better({"val/loss": 4.9})
    assert not policy.is_better({"val/loss": float("nan")})
    assert not policy.is_better({"train/loss": 1.0})
    assert C.CheckpointPolicy(mode="max", best_value=1.0).is_better({"val/loss": 2.0})
    with pytest.raises(ValueError, match="mode"):
        C.CheckpointPolicy(mode="median")


def test_manager_keeps_last_k_and_the_best(tmp_path, uninterrupted, fresh):
    run = uninterrupted["run"]
    state = uninterrupted["s1"]
    manager = C.CheckpointManager(tmp_path, C.CheckpointPolicy(every_n_steps=1000), keep_last_k=1)
    written = [manager.maybe_save_step(step, run.model, run.optimizer, state, {"epoch": 0}) for step in (999, 1000, 2000, 3000)]
    assert written[0] is None and all(written[1:])
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == ["step-000003000.npz"]
    assert manager.last() == tmp_path / "step-000003000.npz"
    assert manager.maybe_save_best(3000, {"val/loss": 4.0}, run.model, run.optimizer, state, {}) is not None
    assert manager.maybe_save_best(3100, {"val/loss": 4.5}, run.model, run.optimizer, state, {}) is None
    assert manager.maybe_save_best(3200, {"val/loss": 3.0}, run.model, run.optimizer, state, {}) is not None
    assert sorted(p.name for p in tmp_path.glob("best-*.npz")) == ["best-step-000003200.npz"]
    again = manager.save(3000, run.model, run.optimizer, uninterrupted["s2"], {"epoch": 0})  # e.g. a walltime save
    assert again == tmp_path / "step-000003000-r1.npz"
    assert manager.last() == again
    assert sorted(p.name for p in tmp_path.glob("step-*.npz")) == ["step-000003000-r1.npz"]  # keep_last_k=1: the older goes
    _, meta = C.load_checkpoint(manager.last(), fresh.model, fresh.optimizer)
    assert meta["step"] == 3000
    reopened = C.CheckpointManager(tmp_path)
    assert reopened.policy.best_value == 3.0
    assert reopened.best() == tmp_path / "best-step-000003200.npz"
    _, meta = C.load_checkpoint(reopened.best(), fresh.model, fresh.optimizer)
    assert (meta["step"], meta["value"], meta["monitor"]) == (3200, 3.0, "val/loss")
