"""S5 gate: the torch-free beta controller (``keras_native.beta_control``) under the JAX backend.

- ``BetaPIDController`` reproduces HGQ2's own ``hgq.utils.sugar.BetaPID`` (log mode) over the EBOPs sequences of
  ``tests/keras/test_beta_pid.py``, to rel 1e-12; perturbing any gain, the target, init beta or the warm-up, or skipping
  the integral seeding, breaks the agreement (the comparison is not vacuous).
- The ``min_beta`` floor (production ``HGQ_MIN_BETA`` > 0) binds on a below-target sequence with damping and matches
  HGQ2; control: the same run without the floor goes below it.
- The epoch hook: warm-up holds ``init_beta``; no reading (``ebops <= 0``) holds and does not seed; a checkpointed
  controller (``state_dict``) continues the trajectory exactly, a fresh one does not. The resume runs with d > 0, so the
  checkpointed ``prev_error`` is load-bearing (control: resuming with ``prev_error`` 0 differs).
- The module imports no keras, jax, torch or lightning (it is shared with the torch-backend hybrid).
The same control-law tests run against the Lightning wrapper in ``tests/keras/test_beta_pid.py`` (torch env).
"""

import json
import os
import subprocess
import sys
from itertools import starmap
from pathlib import Path

import pytest

from hepattn.keras_native.beta_control import BetaPIDController, ramp_beta

TARGET = 1.0e15
INIT_BETA = 2.08e-9
WARMUP = 2
SEQUENCES = {
    "decaying": [1.9e15, 1.7e15, 1.5e15, 1.3e15, 1.1e15, 0.9e15, 0.8e15, 0.95e15, 1.05e15, 1.0e15],
    "below_target": [4.0e14] * 10,
    "unresponsive": [1.6755e15] * 10,
    "noisy": [1.6e15, 1.65e15, 1.58e15, 1.71e15, 1.62e15, 1.55e15, 1.6e15, 1.59e15, 1.63e15, 1.6e15],
}


def drive_reference(seq, target=TARGET, init_beta=INIT_BETA, warmup=WARMUP, p=1.0, i=0.5, d=0.0, max_beta=1e-7, damp=0.0, min_beta=0.0):
    """HGQ2's BetaPID with ``get_ebops`` / ``set_beta`` stubbed (as in tests/keras/test_beta_pid.py)."""
    hgq_pid = pytest.importorskip("hgq.utils.sugar.beta_pid")
    cb = hgq_pid.BetaPID(
        target_ebops=target,
        init_beta=init_beta,
        p=p,
        i=i,
        d=d,
        warmup=warmup,
        log=True,
        max_beta=max_beta,
        min_beta=min_beta,
        damp_beta_on_target=damp,
    )
    feed = iter(seq)
    current = {"ebops": next(feed)}
    cb.get_ebops = lambda: current["ebops"]
    cb.set_beta = lambda _beta: None
    cb.on_train_begin()
    out = []
    for epoch in range(len(seq)):
        logs: dict = {}
        cb.on_epoch_begin(epoch, logs)
        out.append(cb.beta if epoch >= warmup else init_beta)
        current["ebops"] = next(feed, current["ebops"])
        cb.on_epoch_end(epoch, logs)
    return out


def drive_native(seq, target=TARGET, init_beta=INIT_BETA, warmup=WARMUP, p=1.0, i=0.5, d=0.0, max_beta=1e-7, damp=0.0, min_beta=0.0):
    ctl = BetaPIDController(
        target_ebops=target,
        init_beta=init_beta,
        p=p,
        i=i,
        d=d,
        warmup_epochs=warmup,
        max_beta=max_beta,
        min_beta=min_beta,
        damp_beta_on_target=damp,
    )
    return list(starmap(ctl.on_epoch_start, enumerate(seq)))


@pytest.mark.parametrize("name", sorted(SEQUENCES))
@pytest.mark.parametrize("damp", [0.0, 0.5])
def test_controller_equals_hgq2_beta_pid(name, damp):
    ref, ours = drive_reference(SEQUENCES[name], damp=damp), drive_native(SEQUENCES[name], damp=damp)
    assert ours == pytest.approx(ref, rel=1e-12, abs=0)


@pytest.mark.parametrize("gains", [{"p": 0.7, "i": 0.3, "d": 0.1}, {"p": 1.3, "i": 0.9, "d": 0.0}])
def test_controller_equals_hgq2_beta_pid_with_other_gains(gains):
    seq = SEQUENCES["decaying"]
    assert drive_native(seq, **gains) == pytest.approx(drive_reference(seq, **gains), rel=1e-12, abs=0)


@pytest.mark.parametrize("kwargs", [{"p": 1.05}, {"i": 0.55}, {"d": 0.05}, {"init_beta": INIT_BETA * 1.05}, {"target": TARGET * 1.05}, {"warmup": 3}])
def test_comparison_is_discriminating(kwargs):
    seq = SEQUENCES["decaying"]
    ref = drive_reference(seq)
    assert drive_native(seq) == pytest.approx(ref, rel=1e-12, abs=0)
    assert drive_native(seq, **kwargs) != pytest.approx(ref, rel=1e-9, abs=0), f"{kwargs} did not change the trajectory"


MIN_BETA = 3e-10  # between INIT_BETA and the end of the unfloored below_target/damp 0.5 trajectory (measured 4.2e-11)


def test_min_beta_floor_binds_and_equals_hgq2():
    seq = SEQUENCES["below_target"]
    ref = drive_reference(seq, damp=0.5, min_beta=MIN_BETA)
    ours = drive_native(seq, damp=0.5, min_beta=MIN_BETA)
    assert ours == pytest.approx(ref, rel=1e-12, abs=0)
    assert ours.count(MIN_BETA) >= 3 and min(ours) == MIN_BETA, ours  # the floor binds on several epochs
    unfloored = drive_native(seq, damp=0.5)  # control: without the floor beta goes below MIN_BETA
    assert min(unfloored) < MIN_BETA / 5, unfloored


def test_integral_seeding_is_load_bearing():
    ctl = BetaPIDController(target_ebops=TARGET, init_beta=INIT_BETA, warmup_epochs=0)
    assert ctl.step(1.6755e15) == pytest.approx(INIT_BETA, rel=1e-9, abs=0)
    unseeded = BetaPIDController(target_ebops=TARGET, init_beta=INIT_BETA, warmup_epochs=0)
    unseeded.seeded = True
    assert unseeded.step(1.6755e15) != pytest.approx(INIT_BETA, rel=1e-6, abs=0)


def test_epoch_hook_warmup_and_missing_reading():
    ctl = BetaPIDController(target_ebops=TARGET, init_beta=INIT_BETA, warmup_epochs=2)
    assert [ctl.on_epoch_start(e, 1.7e15) for e in range(2)] == [INIT_BETA, INIT_BETA]
    assert not ctl.seeded, "warm-up must not touch the controller"
    assert ctl.on_epoch_start(2, 0.0) == INIT_BETA
    assert not ctl.seeded, "a missing reading must not seed the integral"
    assert ctl.on_epoch_start(3, 1.7e15) == pytest.approx(INIT_BETA, rel=1e-9, abs=0)
    assert ctl.on_epoch_start(4, 1.7e15) > INIT_BETA


def test_state_dict_resumes_the_trajectory():
    """D = 0.1 makes the derivative term (and so the checkpointed ``prev_error``) part of the trajectory."""
    seq, d = SEQUENCES["decaying"], 0.1

    def controller():
        return BetaPIDController(target_ebops=TARGET, init_beta=INIT_BETA, d=d, warmup_epochs=WARMUP)

    full = drive_native(seq, d=d)
    first = controller()
    for epoch in range(5):
        first.on_epoch_start(epoch, seq[epoch])
    saved = json.loads(json.dumps(first.state_dict()))  # what a JSON checkpoint sidecar keeps
    assert saved["prev_error"] != 0.0
    resumed, fresh, no_prev = controller(), controller(), controller()
    resumed.load_state_dict(saved)
    no_prev.load_state_dict({**saved, "prev_error": 0.0})
    tail = [resumed.on_epoch_start(e, seq[e]) for e in range(5, len(seq))]
    assert tail == full[5:]
    # controls: a fresh controller, and one resumed without prev_error, do not reproduce the run
    for name, ctl in (("fresh", fresh), ("prev_error 0", no_prev)):
        control = [ctl.on_epoch_start(e, seq[e]) for e in range(5, len(seq))]
        assert control != pytest.approx(full[5:], rel=1e-6, abs=0), f"{name} reproduced the resumed run: the state is not load-bearing"


def test_ramp():
    assert [ramp_beta(s, 1e-12, 5e-10, 100) for s in (0, 50, 100, 1000)] == pytest.approx(
        [1e-12, 1e-12 + 0.5 * (5e-10 - 1e-12), 5e-10, 5e-10], rel=1e-12, abs=0
    )
    assert ramp_beta(3, 0.0, 1.0, 0) == 1.0  # warmup_steps 0: full strength at once (max(1, warmup) as the hybrid)


def test_module_imports_no_framework():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys, json, hepattn.keras_native.beta_control; "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & {'keras', 'jax', 'torch', 'lightning', 'numpy'})))"
    )
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
