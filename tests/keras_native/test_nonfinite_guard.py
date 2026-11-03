"""The non-finite update guard of ``TrainStep`` (``skip_nonfinite``, default on) and its count in the loop.

Step level, on a small HGQ2 model (``QDense`` x2: trainable bitwidths, non-trainable integer bits and EBOPs) with AdamW
and a global clip. Two injections: ``grad``, a NaN in the gradient only (``poison``: identity forward, backward times
``p``; the loss stays finite), and ``loss``, a NaN loss with finite gradients (``stop_gradient(nan)`` added):

- a bad step leaves tv, ntv and ov (moments, iteration counter) bit-identical and reports ``skipped``; controls: the
  same step without the guard writes NaN into the weights (``grad``) or applies an update (``loss``) and changes ntv;
- a finite step updates every group, and gives the same state bit for bit with and without the guard;
- accumulation over 2 micro-batches: a bad micro-batch (either one, either injection) skips the whole update (weights,
  moments, iteration counter as before the window, accumulators reset), and the next window updates normally;
- switch off: the NaN reaches the weights (the old behaviour).

Loop level, the tiny config (accumulation 2): a finite run is bit for bit the run without the guard; with NaN inputs in
one micro-batch: the update of that window is skipped
(weights and optimizer state of its checkpoint equal the previous step's), ``train/nonfinite_skipped`` counts it in
steps.csv, metrics.csv and the checkpoint metadata, the EBOPs read-out stays finite; control: with the guard off the
final weights are not finite.
"""

from itertools import starmap

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest

from hepattn.keras_native import loop
from hepattn.keras_native.optim import build_optimizer
from hepattn.keras_native.train_step import TrainStep

from .s6b_utils import differing, npz_arrays, sources, tiny_config, trainer

LRS = {"initial": 1e-2, "max": 1e-2, "end": 1e-2, "pct_start": 0.1, "weight_decay": 1e-4, "skip_scheduler": True}


@jax.custom_vjp
def poison(x, p):  # noqa: ARG001  (p only scales the gradient: _poison_bwd)
    return x


def _poison_fwd(x, p):
    return x, p


def _poison_bwd(p, g):
    return g * p, jnp.zeros_like(p)


poison.defvjp(_poison_fwd, _poison_bwd)


def loss_fn(outputs, targets):
    y, p, q = targets
    loss = jnp.mean(jnp.square(poison(outputs, p) - y)) + jax.lax.stop_gradient(q)
    return loss, {"mse": loss}


def make_steps(accumulate: int = 1, skip_nonfinite: bool = True, model=None):
    from hgq.layers import QDense  # noqa: PLC0415

    if model is None:
        keras.utils.set_random_seed(0)
        inputs = keras.Input((4,))
        hidden = QDense(8, activation="relu", name="h")(inputs)
        model = keras.Model(inputs, QDense(3, name="o")(hidden))
        model(np.zeros((2, 4), np.float32))
    opt = build_optimizer(model, optimizer="AdamW", lrs_config=LRS, total_steps=100, clip_norm=0.1, accumulate=accumulate)
    steps = TrainStep(model, opt, loss_fn, donate=False, skip_nonfinite=skip_nonfinite)
    return steps, steps.set_beta(steps.init_state(), 1e-6)


def batch(seed: int, bad: str | None = None):
    """``bad``: None, ``grad`` (NaN gradient, finite loss) or ``loss`` (NaN loss, finite gradient)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 2, (5, 4)).astype(np.float32)
    p, q = (np.nan if bad == "grad" else 1.0), (np.nan if bad == "loss" else 0.0)
    return x, (rng.normal(size=(5, 3)).astype(np.float32), np.float32(p), np.float32(q))


def host(state) -> dict[str, list[np.ndarray]]:
    return {g: [np.asarray(x) for x in getattr(state, g)] for g in ("tv", "ntv", "ov")}


def same(a, b) -> dict[str, bool]:
    return {g: all(starmap(np.array_equal, zip(a[g], b[g], strict=True))) for g in a}


def finite(state) -> bool:
    return all(np.isfinite(x).all() for g in host(state).values() for x in g if x.dtype.kind == "f")


@pytest.mark.parametrize("bad", ["grad", "loss"])
def test_nonfinite_step_leaves_every_variable_and_the_optimizer_state_unchanged(bad):
    steps, state = make_steps()
    state, _ = steps.train_step(state, batch(0))  # nonzero moments, iteration 1
    before = host(state)
    after, out = steps.train_step(state, batch(1, bad))  # another batch: its forward would change ntv
    assert np.isfinite(float(out.loss)) == (bad == "grad") and not bool(out.finite) and bool(out.skipped)
    assert same(host(after), before) == {"tv": True, "ntv": True, "ov": True}
    # control: without the guard the step writes NaN into the weights (grad) or updates them (loss), and changes ntv
    unguarded, _ = make_steps(skip_nonfinite=False, model=steps.model)
    off, out_off = unguarded.train_step(state, batch(1, bad))
    assert finite(off) == (bad == "loss") and not bool(out_off.skipped)
    assert same(host(off), before) == {"tv": False, "ntv": False, "ov": False}


def test_finite_step_updates_the_state_and_the_guard_does_not_change_it():
    steps, state = make_steps()
    plain, _ = make_steps(skip_nonfinite=False, model=steps.model)
    before = host(state)
    guarded, out = steps.train_step(state, batch(0))
    reference, _ = plain.train_step(state, batch(0))
    assert bool(out.finite) and not bool(out.skipped)
    assert same(host(guarded), before) == {"tv": False, "ntv": False, "ov": False}
    assert same(host(guarded), host(reference)) == {"tv": True, "ntv": True, "ov": True}


@pytest.mark.parametrize("bad", ["grad", "loss"])
@pytest.mark.parametrize("bad_micro", [0, 1])
def test_accumulation_skips_the_whole_update_for_one_bad_micro_batch(bad_micro, bad):
    steps, state = make_steps(accumulate=2)
    for i in range(2):  # one complete update first: nonzero moments, iteration 2, accumulators zero
        state, _ = steps.train_step(state, batch(i))
    before = host(state)
    outs = []
    for i in range(2):
        state, out = steps.train_step(state, batch(10 + i, bad if i == bad_micro else None))
        outs.append(out)
    assert [bool(o.skipped) for o in outs] == [False, True]
    assert [bool(o.finite) for o in outs] == [bad_micro == 1, False]  # the window so far
    after = host(state)
    assert same(after, before)["tv"] and same(after, before)["ov"]  # accumulators back to zero, iteration 2 again
    # the next window is a normal update at the same schedule position
    for i in range(2):
        state, out = steps.train_step(state, batch(20 + i))
    assert not bool(out.skipped) and finite(state) and not same(host(state), before)["tv"]
    assert int(host(state)["ov"][steps._iter_pos]) == int(before["ov"][steps._iter_pos]) + 2  # noqa: SLF001


def test_switch_off_lets_the_nan_reach_the_weights():
    steps, state = make_steps(skip_nonfinite=False)
    state, out = steps.train_step(state, batch(1, "grad"))
    assert not bool(out.skipped) and out.finite is None
    assert not all(np.isfinite(np.asarray(x)).all() for x in state.tv)


# --------------------------------------------------------------------------------------------------------- loop


class NanInputs:
    """A data source whose micro-batch ``(epoch, index)`` has NaN in every float input."""

    def __init__(self, source, bad: tuple[int, int]) -> None:
        self.source, self.bad = source, bad

    def __getattr__(self, name):
        return getattr(self.source, name)

    def epoch(self, epoch: int, start: int = 0):
        for b, item in enumerate(self.source.epoch(epoch, start), start=start):
            if (epoch, b) == self.bad:
                nan = {k: jnp.full_like(v, jnp.nan) if jnp.issubdtype(v.dtype, jnp.floating) else v for k, v in item.inputs.items()}
                item = type(item)(nan, item.targets, item.event_valid, item.host)
            yield item


@pytest.fixture(scope="module")
def tiny():
    rc = tiny_config(ckpt_every_n_steps=1, ckpt_save_top_k=-1)
    train, val = sources(rc)
    return rc, train, val


@pytest.mark.parametrize("guard", [True, False])
def test_loop_skips_and_counts_a_nan_micro_batch(tiny, tmp_path, guard):
    rc, train, val = tiny
    nan_train = NanInputs(train, (0, 5))  # the second micro-batch of optimizer step 3 (accumulation 2)
    t = trainer(rc, tmp_path, nan_train, max_steps=4, skip_nonfinite=guard)
    assert t.k == 2
    t.fit(nan_train, val)
    ckpts = tmp_path / "ckpts"
    last = npz_arrays(ckpts / "step-000000004.npz")
    weights_finite = all(np.isfinite(v).all() for k, v in last.items() if k.startswith("tv/") and v.dtype.kind == "f")
    rows = loop.read_csv(tmp_path / "steps.csv")
    epochs = loop.read_csv(tmp_path / "metrics.csv")
    if not guard:  # control: the old behaviour
        assert not weights_finite
        assert all(np.isnan(r["train/nonfinite_skipped"]) for r in rows)
        return
    assert weights_finite and np.isfinite(t.last_ebops)
    assert [r["train/nonfinite_skipped"] for r in rows] == [0.0, 0.0, 1.0, 1.0]
    assert epochs[-1]["train/nonfinite_skipped"] == 1.0 and t.nonfinite_skipped == 1
    assert loop.ckpt.read_meta(ckpts / "step-000000003.npz")["nonfinite_skipped"] == 1
    two, three = npz_arrays(ckpts / "step-000000002.npz"), npz_arrays(ckpts / "step-000000003.npz")
    moved = [k for k in two if k.startswith(("tv/", "ov/")) and not np.array_equal(two[k], three[k])]
    assert moved == [], moved[:5]
    four = npz_arrays(ckpts / "step-000000004.npz")
    assert any(not np.array_equal(three[k], four[k]) for k in three if k.startswith("tv/"))  # step 4 updates again
    # the logged LR is the optimizer's: step 4 applies the schedule's 3rd update
    schedule = loop.learning_rate(loop.lrs_config(rc), t.total_steps, steps_per_update=t.k)
    assert callable(schedule) and float(schedule(2 * t.k)) != float(schedule(3 * t.k))  # OneCycle: the positions differ
    assert rows[3]["train/lr"] == float(schedule(2 * t.k))


def test_loop_finite_run_is_bit_for_bit_the_run_without_the_guard(tiny, tmp_path):
    """Accumulation 2, PeriodicEBOPs, BetaPID: the guard selects the optimizer's own values on finite steps."""
    rc, train, val = tiny
    final = {}
    for guard in (True, False):
        t = trainer(rc, tmp_path / str(guard), train, max_steps=4, skip_nonfinite=guard)
        t.fit(train, val)
        final[guard] = npz_arrays(tmp_path / str(guard) / "ckpts" / "step-000000004.npz")
        assert t.nonfinite_skipped == 0
    assert differing(final[True], final[False]) == []
    first = npz_arrays(tmp_path / "True" / "ckpts" / "step-000000001.npz")
    assert differing(first, final[True])  # control: the run moved the state
