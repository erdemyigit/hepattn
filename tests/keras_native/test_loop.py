"""S6b gates of the training driver (``keras_native.loop``) on the tiny generated Polaris config and CLIC dummy events.

Run A trains 3 epochs uninterrupted (32 events, batch 4, accumulation 2: 4 optimizer steps per epoch, 12 in all; BetaPID
with a 1-epoch warm-up; PeriodicEBOPs every 2 steps; a step checkpoint after every optimizer step, all kept). Then:

- resume (plan S6 gate, CPU): a fresh ``Trainer`` restored from a checkpoint of A continues to A's final state bit for
  bit, from inside an epoch, from the last step of an epoch (end hooks pending), from an epoch checkpoint, and after
  the PID engaged; controls: skipping one micro-batch, or dropping the PID state, changes the final state;
- trace counts over the whole run (plan S6 gate: <= 2 train, <= 1 eval / predict): train 2, val 1 (eval + predict),
  tail 1, bits 1; control: a batch of another shape adds a val trace;
- val/loss: the task loss, event-weighted, with the short batch's loss on its real events; controls: the padded loss
  differs, and task + regularizers (S5's reading of the hybrid) differs;
- PeriodicEBOPs at the optimizer step, BetaPID per epoch on the latest read-out, the BetaScheduler ramp per step;
- the checkpoint policy and the resume-checkpoint protection; one optimizer reader for every config form; refusals;
- the val metrics drop the padding events and are pooled over layers (as the hybrid logs them);
- a second run (log every 3 steps, validation every 2 epochs): the logged steps, the logged LR, the bitwidth counts
  over trainable and non-trainable variables, the validation interval, "highest" in the val / tail programs;
- a resume into the same run directory drops the CSV rows after the checkpoint and the stale checkpoint temporaries.
"""

import dataclasses
import math
import operator
import shutil
from collections import defaultdict
from itertools import starmap

import jax
import keras
import numpy as np
import pytest

from hepattn.experiments.clic.main_jax import ClicSource
from hepattn.keras_native import loop
from hepattn.keras_native.beta_control import BetaPIDController, ramp_beta
from hepattn.keras_native.config import load_config
from hepattn.keras_native.metrics import ClicMetrics
from hepattn.keras_native.optim import optimizer_from_config

from .golden.mf_cases import CONFIGS
from .s6b_utils import REPO, differing, npz_arrays, replace_quant, sources, tiny_config, trainer

STEPS_PER_EPOCH, EPOCHS = 4, 3
FINAL = "step-000000012.npz"


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return loop.enable_compilation_cache(tmp_path_factory.mktemp("jax_cache"))


@pytest.fixture(scope="module")
def run_a(tmp_path_factory, cache):  # noqa: ARG001  (the cache makes the later trainers' compiles hits)
    rc = tiny_config(ckpt_every_n_steps=1, ckpt_save_top_k=-1)
    train, val = sources(rc)
    run_dir = tmp_path_factory.mktemp("run_a")
    t = trainer(rc, run_dir, train)
    summary = t.fit(train, val)
    return {"rc": rc, "train": train, "val": val, "dir": run_dir, "trainer": t, "summary": summary, "counts": t.compile_counts()}


def state_arrays(t) -> dict:
    """The trainer's current state keyed as in a checkpoint file."""
    keys = loop.ckpt.state_keys(t.steps.model, t.steps.optimizer)
    return {f"{g}/{k}": v for g, values in zip(loop.ckpt.GROUPS, loop.host_state(t.state), strict=True) for k, v in zip(keys[g], values, strict=True)}


def same_values(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a)


def untimed(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("time/")}


def resume(run_a, tmp_path, checkpoint: str, *, tamper=None, rc=None):
    rc = rc or run_a["rc"]
    t = trainer(rc, tmp_path, run_a["train"])
    t.restore(run_a["dir"] / "ckpts" / checkpoint)
    if tamper is not None:
        tamper(t)
    t.fit(run_a["train"], run_a["val"])
    return t


def test_run_a_is_non_vacuous(run_a):
    """Every state group changes over the run, the loss falls, the quantizers move and beta changes after the warm-up."""
    t, ckpts = run_a["trainer"], run_a["dir"] / "ckpts"
    assert (t.global_step, t.micro_step, t.epoch, t.batch) == (12, 24, 3, 0)
    first, last = npz_arrays(ckpts / "step-000000001.npz"), npz_arrays(ckpts / FINAL)
    changed = differing(first, last)
    assert {k.split("/", 1)[0] for k in changed} == {"tv", "ntv", "ov"}
    assert any(k.startswith("tv/") and k.rsplit("/", 1)[-1] in {"b", "f", "i"} for k in changed)
    rows = loop.read_csv(run_a["dir"] / "steps.csv")
    assert len(rows) == 12
    losses = [r["train/loss"] for r in rows]
    assert np.mean(losses[-4:]) < np.mean(losses[:4])
    epochs = loop.read_csv(run_a["dir"] / "metrics.csv")
    assert epochs[-1]["val/loss"] < epochs[0]["val/loss"]
    assert epochs[2]["train/quant_beta"] != epochs[0]["train/quant_beta"]


@pytest.mark.parametrize(
    ("checkpoint", "position"),
    [
        ("step-000000003.npz", (0, 6)),  # inside epoch 0
        ("step-000000004.npz", (0, 8)),  # last step of epoch 0: validation and epoch checkpoint still due
        ("epoch-000-step-000000004.npz", (1, 0)),  # after epoch 0's end hooks
        ("step-000000006.npz", (1, 4)),  # after the PID's first epoch step
    ],
)
def test_resume_continues_bit_for_bit(run_a, tmp_path, checkpoint, position):
    meta = loop.ckpt.read_meta(run_a["dir"] / "ckpts" / checkpoint)
    assert (meta["data_position"]["epoch"], meta["data_position"]["batch"]) == position
    t = resume(run_a, tmp_path, checkpoint)
    assert differing(npz_arrays(tmp_path / "ckpts" / FINAL), npz_arrays(run_a["dir"] / "ckpts" / FINAL)) == []
    assert differing(state_arrays(t), npz_arrays(run_a["dir"] / "ckpts" / FINAL)) == []
    a_rows = {r["step"]: r for r in loop.read_csv(run_a["dir"] / "steps.csv")}
    for row in loop.read_csv(tmp_path / "steps.csv"):
        assert same_values(untimed(row), untimed(a_rows[row["step"]]))
    a_val = loop.read_csv(run_a["dir"] / "metrics.csv")
    b_val = loop.read_csv(tmp_path / "metrics.csv")
    assert len(b_val) == EPOCHS - position[0]
    # every column (BetaPID diagnostics and the last train row from the checkpoint metadata included), not only val/loss
    for b, a in zip(b_val, a_val[len(a_val) - len(b_val) :], strict=True):
        assert same_values(untimed(b), untimed(a))
    assert t.beta == run_a["trainer"].beta
    assert t.last_ebops == run_a["trainer"].last_ebops


def _skip_one_batch(t):
    t.batch += 1  # the data position of a lossy checkpoint


def _drop_pid_state(t):
    t.pid = BetaPIDController(**t.rc.quant.beta_pid)


@pytest.mark.parametrize("tamper", [_skip_one_batch, _drop_pid_state])
def test_control_a_lost_piece_of_resume_state_changes_the_run(run_a, tmp_path, tamper):
    t = resume(run_a, tmp_path, "step-000000006.npz", tamper=tamper)
    assert differing(state_arrays(t), npz_arrays(run_a["dir"] / "ckpts" / FINAL)) != []


def test_trace_counts_over_the_run(run_a):
    """Three epochs (more than the gate's one): 2 train programs (EBOPs on / off), 1 val (= eval + predict), 1 tail."""
    assert run_a["counts"] == {"train": 2, "eval": 0, "predict": 0, "val": 1, "tail": 1, "bits": 1}
    # control: a val batch of another shape is a new trace, so the counter sees retraces
    t = run_a["trainer"]
    batch = next(iter(run_a["val"].epoch(0)))
    short = ({k: v[:3] for k, v in batch.inputs.items()}, {k: v[:3] for k, v in batch.targets.items()})
    jax.block_until_ready(t._val(t.state, *short)["task"])  # noqa: SLF001
    assert t.compile_counts()["val"] == 2


MEASURED_VAL_REL = 3.52e-8  # measured: the 2-ulp tail difference (test below), weighted 2 of 10 events


def test_val_loss_is_the_event_weighted_task_loss_on_real_events(run_a):
    t = run_a["trainer"]
    batches = list(run_a["val"].epoch(0))
    assert [b.num_real for b in batches] == [4, 4, 2]
    weighted, reg, events = 0.0, 0.0, 0
    for batch in batches:
        n = batch.num_real
        inputs, targets = ({k: v[:n] for k, v in tree.items()} for tree in (batch.inputs, batch.targets))
        res = t._val(t.state, inputs, targets)  # noqa: SLF001  (the unpadded batch: another trace)
        weighted += float(res["task"]) * n
        reg += float(res["regularization"]) * n
        events += n
    val = t.validate(run_a["val"])
    assert same_values(val, t.last_val)  # the stored epoch-end values are of the final state
    print(f"val/loss {val['val/loss']!r} vs unpadded {weighted / events!r}: rel {abs(val['val/loss'] / (weighted / events) - 1):.3e}")
    assert val["val/loss"] == pytest.approx(weighted / events, rel=2 * MEASURED_VAL_REL)
    assert val["val/quant_ebops_loss"] == pytest.approx(reg / events, rel=1e-7)
    # controls: the padded batch's own loss is not the real events' loss; task + regularizers is not val/loss
    padded = t._val(t.state, batches[-1].inputs, batches[-1].targets)  # noqa: SLF001
    tail = t._val(t.state, *({k: v[:2] for k, v in tree.items()} for tree in (batches[-1].inputs, batches[-1].targets)))  # noqa: SLF001
    assert float(padded["task"]) != float(tail["task"])
    assert val["val/quant_ebops_loss"] > 0  # so task + regularizers (S5's reading of the hybrid's val/loss) is not val/loss


def ulps(a, b) -> float:
    """|a - b| in float32 spacings of b."""
    return abs(float(a) - float(b)) / float(np.spacing(np.float32(abs(float(b)))))


# Measured (CPU, this config): the loss of the real events from the padded forward differs from the unpadded batch's
# by 2 ulps (task total) and at most MEASURED_TERM_ULPS in a term: two compiled programs of other shapes, so other
# reduction orders. Tolerances are 2x measured. The padded batch's own loss is off by > 1e4 ulps (the control).
MEASURED_TASK_ULPS, MEASURED_TERM_ULPS = 2.0, 2.0


def test_tail_loss_from_the_padded_forward_equals_the_unpadded_batch(run_a):
    t = run_a["trainer"]
    batch = list(run_a["val"].epoch(0))[-1]
    res = t._val(t.state, batch.inputs, batch.targets)  # noqa: SLF001
    outputs = jax.tree_util.tree_map(operator.itemgetter(slice(2)), res["outputs"])
    task, losses = t._tail(outputs, {k: v[:2] for k, v in batch.targets.items()})  # noqa: SLF001
    direct = t._val(t.state, *({k: v[:2] for k, v in tree.items()} for tree in (batch.inputs, batch.targets)))  # noqa: SLF001
    term_ulps = list(starmap(ulps, zip(jax.tree_util.tree_leaves(losses), jax.tree_util.tree_leaves(direct["losses"]), strict=True)))
    print(
        f"tail vs unpadded: task {ulps(task, direct['task'])} ulps, terms max {max(term_ulps)} ulps; padded {ulps(res['task'], direct['task'])} ulps"
    )
    assert ulps(task, direct["task"]) <= 2 * MEASURED_TASK_ULPS
    assert max(term_ulps) <= 2 * MEASURED_TERM_ULPS
    assert ulps(res["task"], direct["task"]) > 1e4


def test_periodic_ebops_follows_the_optimizer_step(run_a):
    """``every_n_steps`` 2 with accumulation 2: optimizer steps 0, 2, 4, ... carry the term (all their micro-batches).
    A micro-step counter would log 0 on every row (the logged micro-batch is the second of its step).
    """
    rows = loop.read_csv(run_a["dir"] / "steps.csv")
    assert [r["train/ebops_active"] for r in rows] == [float((r["step"] - 1) % 2 == 0) for r in rows]
    on = [r["train/quant_ebops_loss"] for r in rows if r["train/ebops_active"]]
    off = [r["train/quant_ebops_loss"] for r in rows if not r["train/ebops_active"]]
    assert min(on) > 5 * max(off) > 0  # off: the beta-free regularizer terms only
    for r in rows:
        assert r["train/total_loss"] == pytest.approx(r["train/loss"] + r["train/quant_ebops_loss"], rel=1e-6)


def test_beta_pid_steps_each_epoch_on_the_latest_readout(run_a):
    rc, rows = run_a["rc"], loop.read_csv(run_a["dir"] / "steps.csv")
    epochs = loop.read_csv(run_a["dir"] / "metrics.csv")
    pid = BetaPIDController(**rc.quant.beta_pid)
    for e, row in enumerate(epochs):
        reading = 0.0 if e == 0 else next(r["train/ebops"] for r in rows if r["step"] == STEPS_PER_EPOCH * e)
        if e == 0:
            assert math.isnan(row["train/pid_ebops"])
        else:
            assert row["train/pid_ebops"] == reading
        assert row["train/quant_beta"] == pid.on_epoch_start(e, reading)
        assert all(r["train/quant_beta"] == row["train/quant_beta"] for r in rows if r["epoch"] == e)
    # control: without the read-out (the hybrid's uint32 _ebops is saturated on JAX) the PID would hold init_beta
    assert epochs[2]["train/quant_beta"] != rc.quant.beta_pid["init_beta"]


def test_beta_scheduler_ramps_at_the_optimizer_step(run_a, tmp_path):
    ramp = {"beta_start": 1e-10, "beta_end": 1e-9, "warmup_steps": 5}
    rc = replace_quant(run_a["rc"], beta_pid=None, beta_scheduler=ramp)
    t = trainer(rc, tmp_path, run_a["train"], max_epochs=2)
    t.fit(run_a["train"], run_a["val"])
    rows = loop.read_csv(tmp_path / "steps.csv")
    logged = [r["train/quant_beta"] for r in rows]
    assert logged == [ramp_beta(int(r["step"]) - 1, **ramp) for r in rows]
    assert logged != [ramp_beta(int(r["step"]), **ramp) for r in rows]  # control: the step after
    assert np.float32(t.steps.read_beta(t.state)) == np.float32(logged[-1])


def test_checkpoint_policy(run_a):
    ckpts = run_a["dir"] / "ckpts"
    assert sorted(p.name for p in ckpts.glob("step-*.npz")) == [f"step-{s:09d}.npz" for s in range(1, 13)]
    assert sorted(p.name for p in ckpts.glob("epoch-*.npz")) == [f"epoch-{e:03d}-step-{4 * (e + 1):09d}.npz" for e in range(3)]
    epochs = loop.read_csv(run_a["dir"] / "metrics.csv")
    best = int(np.argmin([r["val/loss"] for r in epochs]))
    assert loop.json.loads((ckpts / "best.json").read_text())["file"] == f"epoch-{best:03d}-step-{4 * (best + 1):09d}.npz"
    assert loop.json.loads((ckpts / "last.json").read_text())["file"] == FINAL
    meta = loop.ckpt.read_meta(ckpts / FINAL)
    assert meta["config_hash"] == run_a["rc"].model_hash
    assert meta["beta_controller"]["seeded"] is True


def test_resume_checkpoint_survives_pruning(run_a, tmp_path):
    """save_top_k 1 keeps the newest step file only, but never the file the run resumed from (S56 open issue 6)."""
    rc = tiny_config(ckpt_every_n_steps=1, ckpt_save_top_k=1)
    (tmp_path / "ckpts").mkdir()
    for suffix in (".npz", ".json"):
        shutil.copy(run_a["dir"] / "ckpts" / f"step-000000010{suffix}", tmp_path / "ckpts")
    t = trainer(rc, tmp_path, run_a["train"])
    t.restore(tmp_path / "ckpts" / "step-000000010.npz")
    t.fit(run_a["train"], run_a["val"])
    assert sorted(p.name for p in (tmp_path / "ckpts").glob("step-*.npz")) == ["step-000000010.npz", FINAL]
    assert differing(npz_arrays(tmp_path / "ckpts" / FINAL), npz_arrays(run_a["dir"] / "ckpts" / FINAL)) == []


def test_stepping_past_total_steps_raises(run_a):
    """Torch OneCycleLR raises at step total_steps + 1; the native schedule would hold silently (S56 open issue 5)."""
    t = run_a["trainer"]
    assert t.global_step == t.total_steps
    with pytest.raises(RuntimeError, match="exceeds total_steps"):
        t._train_micro(next(iter(run_a["train"].epoch(0))))  # noqa: SLF001
    assert t.micro_step == 24  # nothing ran


def test_restore_refuses_another_schedule_or_model(run_a, tmp_path):
    rc = run_a["rc"]
    t = trainer(rc, tmp_path, run_a["train"], max_epochs=4)  # 16 total steps: another LR schedule
    with pytest.raises(ValueError, match="total_steps"):
        t.restore(run_a["dir"] / "ckpts" / "step-000000006.npz")
    t.total_steps = 12  # control: the same lengths load
    t.restore(run_a["dir"] / "ckpts" / "step-000000006.npz")
    assert t.global_step == 6
    # another seed would continue with another data order (EpochSampler(seed + epoch))
    other = trainer(rc, tmp_path / "seed", run_a["train"], seed=43)
    with pytest.raises(ValueError, match="seed 42 differs from this run's 43"):
        other.restore(run_a["dir"] / "ckpts" / "step-000000006.npz")
    other.restore(run_a["dir"] / "ckpts" / "step-000000006.npz", check_seed=False)  # the test command's path


def test_non_fp32_precision_is_refused(run_a):
    rc = dataclasses.replace(run_a["rc"], trainer={**run_a["rc"].trainer, "precision": "bf16-mixed"})
    batch = next(iter(run_a["train"].epoch(0)))
    with pytest.raises(ValueError, match="fp32 only"):
        loop.Trainer(rc, loop.LoopSettings.from_run_config(rc, "unused"), sample=batch, train_batches=8)


FORMS = {
    "tiny generated": [REPO / "tests/keras_native/s6b_fixtures/tiny_polaris.yaml"],
    "org raw": [CONFIGS / "base.yaml", CONFIGS / "linformer.yaml"],
    "pflow_hgq": [CONFIGS / "pflow_hgq.yaml"],
    "org saved": [REPO / "tests/keras_native/s6_fixtures/org_normalised.yaml"],
    "pflow_hgq saved": [REPO / "tests/keras_native/s6_fixtures/pflow_hgq_normalised.yaml"],
    "polaris saved": [REPO / "tests/keras_native/s6_fixtures/polaris_normalised.yaml"],
}


def _small_model():
    model = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2, name="hid")])
    model(np.zeros((1, 3), np.float32))
    return model


def unnamed(config: dict) -> dict:
    return {k: v for k, v in config.items() if k != "name"}  # keras numbers auto names per instance (adam_w_10, ...)


@pytest.mark.parametrize("form", list(FORMS))
def test_one_optimizer_reader_for_every_form(form):
    """The driver builds from ``RunConfig.optimizer``; S5's ``optimizer_from_config`` reads the same YAML keys."""
    rc = load_config(FORMS[form])
    model = _small_model()
    native = unnamed(loop.build_run_optimizer(rc, model, 400).get_config())
    s5 = unnamed(optimizer_from_config(rc.model_section, rc.raw.get("trainer") or {}, model, 400).get_config())
    assert native == s5
    # control: a different epsilon (keras' default 1e-7) or accumulation is seen
    other = dataclasses.replace(rc, optimizer=dataclasses.replace(rc.optimizer, accumulate_grad_batches=rc.optimizer.accumulate_grad_batches + 1))
    assert unnamed(loop.build_run_optimizer(other, model, 400).get_config()) != s5


def test_clic_source_counts_full_train_batches_and_padded_val_batches(run_a):
    train, val = run_a["train"], run_a["val"]
    assert (train.num_batches, val.num_batches, val.num_events) == (8, 3, 10)
    assert isinstance(train, ClicSource)


def test_val_metrics_drop_the_padding_events_and_pool_the_layers(run_a):
    """``validate``'s metrics equal ``ClicMetrics`` (pooled, as the hybrid logs) on the real events only.

    The last val batch holds 2 real events and 2 padding events. Controls: counting the padding events, or per-layer
    values, give other values (so the comparison sees the ``event_valid`` and ``pool_layers`` wiring).
    """
    t = run_a["trainer"]
    batches = list(run_a["val"].epoch(0))
    assert [b.num_real for b in batches] == [4, 4, 2]
    real, padded, per_layer = ClicMetrics("val", pool_layers=True), ClicMetrics("val", pool_layers=True), ClicMetrics("val")
    for b in batches:
        res = t._val(t.state, b.inputs, b.targets)  # noqa: SLF001  (the same program validate runs)
        n = b.num_real
        real.update(jax.tree_util.tree_map(lambda x, n=n: np.asarray(x)[:n], res["preds"]), {k: np.asarray(v)[:n] for k, v in b.targets.items()})
        padded.update(res["preds"], b.targets)
        per_layer.update(res["preds"], b.targets, event_valid=b.event_valid)
    expected, with_padding, unpooled = real.compute(), padded.compute(), per_layer.compute()
    val = t.validate(run_a["val"])
    assert len(expected) > 20
    assert same_values({k: val[k] for k in expected}, expected)  # measured: bitwise (same predictions, padding masked)
    # pooled (the torchmetrics counters; the per-batch means stay per layer): one value per counter for every layer
    groups = defaultdict(list)
    for name, key in real.reported.items():
        groups[key].append(expected[name])
    assert len(groups) >= 4 and all(len(values) >= 2 for values in groups.values()), groups
    for key, values in groups.items():
        assert all(v == values[0] or (math.isnan(v) and math.isnan(values[0])) for v in values), key
    # controls
    differ = [k for k in expected if not (expected[k] == with_padding[k] or (math.isnan(expected[k]) and math.isnan(with_padding[k])))]
    print(f"{len(differ)} of {len(expected)} val metrics change when the padding events count")
    assert len(differ) > 5
    assert any(expected[k] != unpooled[k] for k in expected if not math.isnan(expected[k]))
    metrics_csv = loop.read_csv(run_a["dir"] / "metrics.csv")
    assert set(expected) <= set(metrics_csv[-1])


# Measured (CPU, this run): the logged float32 mean of per-variable means is 1.915e-6 relative from the float64 numpy
# value (summation order and precision); the tolerance is 2x. Min and max are compared exactly.
MEASURED_BITS_MEAN_REL = 1.915e-6


def bit_reduction(model, tv, ntv, suffix: str) -> tuple[float, float, float, int]:
    """``BitwidthMonitor``'s reduction in numpy over the variables whose path ends in ``/<suffix>`` (``tv`` / ``ntv`` in
    the order of ``model.trainable_variables`` / ``non_trainable_variables``): mean of the per-variable means, min of
    the mins, max of the maxes, count.
    """
    pairs = [*zip(model.trainable_variables, tv, strict=True), *zip(model.non_trainable_variables, ntv, strict=True)]
    sel = [np.asarray(a, np.float64) for v, a in pairs if v.path.rsplit("/", 1)[-1] == suffix]
    return float(np.mean([a.mean() for a in sel])), float(min(a.min() for a in sel)), float(max(a.max() for a in sel)), len(sel)


def test_log_cadence_lr_bits_val_interval_and_precision(run_a, tmp_path):
    """Log every 3 optimizer steps (across the epoch boundary: 4 steps per epoch) and validate every 2nd epoch.

    Checks the logged steps (Lightning: ``global_step % n == 0``), ``train/lr`` = the schedule at that step's update
    (``(step - 1) * k`` micro-batches), the bitwidth values = ``BitwidthMonitor``'s reduction over trainable and
    non-trainable quantizer variables, one validation row (epoch 1) and one epoch checkpoint (Lightning saves only at a
    validation end when ``check_val_every_n_epoch != 1``), and "highest" on every dot of the val and tail programs.
    """
    rc = run_a["rc"]
    t = trainer(rc, tmp_path, run_a["train"], max_epochs=3, log_every_n_steps=3, check_val_every_n_epoch=2)
    t.fit(run_a["train"], run_a["val"])
    rows = loop.read_csv(tmp_path / "steps.csv")
    assert [r["step"] for r in rows] == [3.0, 6.0, 9.0, 12.0]
    # epochs 0 and 2 run no validation: no epoch file (the epoch-2 one would carry epoch 1's val/loss)
    assert sorted(p.name for p in (tmp_path / "ckpts").glob("epoch-*.npz")) == ["epoch-001-step-000000008.npz"]
    for r in rows:
        assert r["train/lr"] == float(t.schedule((int(r["step"]) - 1) * t.k))
        assert r["train/lr"] != float(t.schedule(int(r["step"]) * t.k))  # control: the next update's LR
    epochs = loop.read_csv(tmp_path / "metrics.csv")
    assert [r["epoch"] for r in epochs] == [1.0]
    assert epochs[0]["train/loss"] == rows[1]["train/loss"]  # the epoch row carries the last logged train row (step 6)
    model = t.steps.model
    for suffix in loop.BIT_SUFFIXES:
        trainable = sum(v.path.rsplit("/", 1)[-1] == suffix for v in model.trainable_variables)
        frozen = sum(v.path.rsplit("/", 1)[-1] == suffix for v in model.non_trainable_variables)
        assert rows[-1][f"train/bits_{suffix}_n"] == trainable + frozen == epochs[0][f"val/bits_{suffix}_n"]
    assert sum(v.path.rsplit("/", 1)[-1] in loop.BIT_SUFFIXES for v in model.non_trainable_variables) > 0  # else vacuous
    # the values: the step-12 row from the final state, the epoch-1 val row from the epoch-1 checkpoint (step 8)
    keys = loop.ckpt.state_keys(model, t.steps.optimizer)
    saved = npz_arrays(tmp_path / "ckpts/epoch-001-step-000000008.npz")
    epoch1 = tuple([saved[f"{g}/{k}"] for k in keys[g]] for g in ("tv", "ntv"))
    spread, rel = [], []
    for stage, row, (tv, ntv) in (("train", rows[-1], (t.state.tv, t.state.ntv)), ("val", epochs[0], epoch1)):
        for suffix in loop.BIT_SUFFIXES:
            mean, lo, hi, n = bit_reduction(model, tv, ntv, suffix)
            assert row[f"{stage}/bits_{suffix}_n"] == n
            assert (row[f"{stage}/bits_{suffix}_min"], row[f"{stage}/bits_{suffix}_max"]) == (lo, hi), (stage, suffix)
            rel.append(abs(row[f"{stage}/bits_{suffix}_mean"] / mean - 1))
            spread.append(hi - lo)
    print(f"bits mean: float32 device vs float64 numpy, max rel {max(rel):.3e}")
    assert max(rel) <= 2 * MEASURED_BITS_MEAN_REL
    assert max(spread) > 0  # else a min / max mix-up would pass
    batch = list(run_a["val"].epoch(0))[-1]
    val_text = t._val.lower(t.state, batch.inputs, batch.targets).as_text()  # noqa: SLF001
    res = t._val(t.state, batch.inputs, batch.targets)  # noqa: SLF001
    outputs = jax.tree_util.tree_map(operator.itemgetter(slice(2)), res["outputs"])
    tail_text = t._tail.lower(outputs, {k: v[:2] for k, v in batch.targets.items()}).as_text()  # noqa: SLF001
    for name, text in (("val", val_text), ("tail", tail_text)):
        dots = [ln for ln in text.splitlines() if "dot_general" in ln]
        loose = [ln.strip()[:160] for ln in dots if "precision = [HIGHEST, HIGHEST]" not in ln]
        assert not loose, (name, len(dots), loose[:3])
        assert len(dots) > (20 if name == "val" else 0), (name, len(dots))


def test_resume_in_the_same_run_directory(run_a, tmp_path):
    """A walltime kill leaves CSV rows after the last checkpoint and a checkpoint temporary; a resume in the same
    directory drops both, and its logs equal the uninterrupted run's. Control: before the resume the copy holds the
    rows of steps 7-12, which the resume writes again.
    """
    run = tmp_path / "run"
    shutil.copytree(run_a["dir"], run)
    stale = run / "ckpts" / ".step-000000008.npz.abc123.tmp"
    stale.write_bytes(b"partial")
    assert len(loop.read_csv(run / "steps.csv")) == 12
    t = trainer(run_a["rc"], run, run_a["train"])
    t.restore(run / "ckpts" / "step-000000006.npz")
    t.fit(run_a["train"], run_a["val"])
    assert not stale.exists()
    a_steps, b_steps = loop.read_csv(run_a["dir"] / "steps.csv"), loop.read_csv(run / "steps.csv")
    assert [r["step"] for r in b_steps] == [float(s) for s in range(1, 13)]
    assert all(same_values(untimed(b), untimed(a)) for a, b in zip(a_steps, b_steps, strict=True))
    a_val, b_val = loop.read_csv(run_a["dir"] / "metrics.csv"), loop.read_csv(run / "metrics.csv")
    assert [r["epoch"] for r in b_val] == [0.0, 1.0, 2.0]
    assert all(same_values(untimed(b), untimed(a)) for a, b in zip(a_val, b_val, strict=True))
    assert differing(state_arrays(t), npz_arrays(run_a["dir"] / "ckpts" / FINAL)) == []


def test_csv_log_grows_its_header(tmp_path):
    """A row with new keys rewrites the file with the new columns appended.

    Earlier rows keep their values and get empty cells in the new columns. (The hybrid MetricsCsvWriter fixes the
    header at the first row and drops later keys.)
    """
    log = loop.CsvLog(tmp_path / "m.csv")
    log.append({"epoch": 0.0, "step": 4.0, "val/loss": 2.0})
    log.append({"epoch": 1.0, "step": 8.0, "val/loss": 1.5, "train/loss": 3.0})
    rows = loop.read_csv(tmp_path / "m.csv")
    assert list(rows[0]) == ["epoch", "step", "val/loss", "train/loss"]
    assert rows[1]["train/loss"] == 3.0
    assert math.isnan(rows[0]["train/loss"])
    again = loop.CsvLog(tmp_path / "m.csv")  # a resumed process reads the header back
    again.append({"epoch": 2.0, "step": 12.0, "val/loss": 1.0, "train/loss": 2.5})
    assert [r["val/loss"] for r in loop.read_csv(tmp_path / "m.csv")] == [2.0, 1.5, 1.0]
    assert again.keep(lambda r: r["epoch"] < 1) == 2
    assert [r["epoch"] for r in loop.read_csv(tmp_path / "m.csv")] == [0.0]


def test_max_steps_stops_inside_an_epoch(run_a, tmp_path):
    """``max_steps`` 5 with 4 steps per epoch: the LR schedule is 5 steps long, the run stops after step 5 (inside epoch
    1, so no epoch-1 validation), and ``last`` is the step-5 file. Control: the 3-epoch schedule (12 steps) differs.
    """
    rc = run_a["rc"]
    t = trainer(rc, tmp_path, run_a["train"], max_steps=5)
    summary = t.fit(run_a["train"], run_a["val"])
    assert (summary["global_step"], summary["total_steps"], t.micro_step, t.epoch, t.batch) == (5, 5, 10, 1, 2)
    rows = loop.read_csv(tmp_path / "steps.csv")
    assert [r["step"] for r in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
    five = loop.learning_rate(loop.lrs_config(rc), 5, steps_per_update=t.k)
    twelve = loop.learning_rate(loop.lrs_config(rc), 12, steps_per_update=t.k)
    assert [r["train/lr"] for r in rows] == [float(five((int(r["step"]) - 1) * t.k)) for r in rows]
    assert [r["train/lr"] for r in rows] != [float(twelve((int(r["step"]) - 1) * t.k)) for r in rows]
    assert [r["epoch"] for r in loop.read_csv(tmp_path / "metrics.csv")] == [0.0]
    assert loop.json.loads((tmp_path / "ckpts/last.json").read_text())["file"] == "step-000000005.npz"


def test_limit_val_batches_validates_the_first_batches_only(run_a):
    """``limit_val_batches`` 1: val/loss is the first batch's task loss (4 real events). Control: all 3 batches differ."""
    t = run_a["trainer"]
    first = next(iter(run_a["val"].epoch(0)))
    assert first.num_real == 4
    expected = float(t._val(t.state, first.inputs, first.targets)["task"])  # noqa: SLF001
    full = t.validate(run_a["val"])
    settings = t.s
    try:
        t.s = dataclasses.replace(settings, limit_val_batches=1)
        limited = t.validate(run_a["val"])
    finally:
        t.s = settings
    assert limited["val/loss"] == expected
    assert full["val/loss"] != expected


def test_seed_sets_the_model_init_and_the_train_order(run_a, tmp_path):
    """Seeds 7, 7 and 8 (not 42: ``CLICDataset`` seeds the global RNGs with 42 in its constructor, so a dropped
    ``set_random_seed`` would still give the seed-42 init): one seed gives the same initial state and epoch-0 order
    bitwise; another seed gives another init and another order.
    """
    rc = run_a["rc"]
    inits, orders = [], []
    for i, seed in enumerate((7, 7, 8)):
        train, _ = sources(rc, seed=seed)
        orders.append([b.host["event_number"].tolist() for b in train.epoch(0)])
        t = trainer(rc, tmp_path / str(i), train, seed=seed)
        inits.append(state_arrays(t))
    assert differing(inits[0], inits[1]) == []
    assert orders[0] == orders[1]
    assert any(k.startswith("tv/") for k in differing(inits[0], inits[2]))
    assert orders[0] != orders[2]
