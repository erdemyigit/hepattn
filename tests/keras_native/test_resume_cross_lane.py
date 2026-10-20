"""Cross-lane resume gate: S5's ``TrainStep`` state through S6a's checkpoint into a freshly built model (plan S6 gate).

The uninterrupted run and the resumed run each compile their own ``TrainStep`` (a new process builds a new model and a
new program), so this also checks that two independently built org models give the same step bit for bit on CPU.

Setup (chosen so that every piece of resumable state matters): org model, HGQ2 ``high`` spec, beta 1e-9 with the EBOPs
term on (beta lives in ``ntv``), ``build_optimizer`` AdamW with the OneCycle schedule (the learning rate reads
``iterations`` in ``ov``), clipping, and gradient accumulation k = 2. The checkpoint is taken after micro-step 1, i.e.
in the middle of an accumulation: the weights have not moved yet and the averaged gradient of micro-batch 0 exists
only in the optimizer's accumulators. The resumed micro-step 2 applies the update.

Controls: one perturbed leaf of the loaded state (a weight by 1 ulp, or an accumulator) makes the comparison fail;
the BetaPID controller state stored in the checkpoint metadata restores the next beta exactly, a fresh one does not.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hepattn.keras_native import checkpoint as C
from hepattn.keras_native.beta_control import BetaPIDController
from hepattn.keras_native.objective import clic_specs
from hepattn.keras_native.optim import build_optimizer
from hepattn.keras_native.train_step import TrainState, TrainStep, maskformer_loss

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .mf_golden_utils import build_mf
from .test_checkpoint import changed_groups, first_difference, host
from .test_lane_integration import real_like

LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-6, "pct_start": 0.3, "weight_decay": 1e-4, "skip_scheduler": False}
BETA = 1e-9
ACCUMULATE = 2
PID = {"target_ebops": 1e5, "init_beta": 1e-9, "d": 0.2, "warmup_epochs": 1, "max_beta": 1e-6, "min_beta": 1e-12}
EBOPS_READINGS = [3e5, 2e5, 1.5e5]  # epochs 1-3 after the warm-up: seeds the PID; d > 0 makes prev_error matter


def build(inputs):
    model = build_mf("org", QUANT_SPECS["high"], inputs)
    opt = build_optimizer(model, optimizer="AdamW", lrs_config=LRS, total_steps=50, clip_norm=0.1, accumulate=ACCUMULATE)
    return TrainStep(model, opt, maskformer_loss(clic_specs()))


def device(state) -> TrainState:
    return TrainState(*([jnp.asarray(x) for x in group] for group in state))


def step(steps: TrainStep, state, batch):
    new, out = steps.train_step(device(state), batch, ebops_on=True)
    return host(new), float(out.loss)


def accumulator_indices(steps: TrainStep) -> list[int]:
    return [i for i, v in enumerate(steps.optimizer.variables) if "accumulat" in v.path]


@pytest.fixture(scope="module")
def batches():
    return [(inputs, real_like(targets)) for inputs, targets in (clic_dummy_batch(2, offset=2 * i) for i in range(2))]


@pytest.fixture(scope="module")
def uninterrupted(batches):
    steps = build(batches[0][0])
    s0 = host(steps.set_beta(steps.init_state(), BETA))
    s1, loss1 = step(steps, s0, batches[0])
    s2, loss2 = step(steps, s1, batches[1])
    pid = BetaPIDController(**PID)
    for epoch, reading in enumerate([0.0, *EBOPS_READINGS]):
        pid.on_epoch_start(epoch, reading)
    return {"steps": steps, "s0": s0, "s1": s1, "s2": s2, "losses": (loss1, loss2), "pid": pid}


@pytest.fixture(scope="module")
def saved(tmp_path_factory, uninterrupted):
    steps = uninterrupted["steps"]
    meta = {"step": 1, "epoch": len(EBOPS_READINGS), "beta_controller": uninterrupted["pid"].state_dict()}
    return C.save_checkpoint(tmp_path_factory.mktemp("ck") / "mid-accumulation.npz", steps.model, steps.optimizer, uninterrupted["s1"], meta)


@pytest.fixture(scope="module")
def resumed(saved, uninterrupted, batches):  # noqa: ARG001  (built after the first model: other auto-generated names)
    steps = build(batches[0][0])
    state, meta = C.load_checkpoint(saved, steps.model, steps.optimizer)
    return {"steps": steps, "state": state, "meta": meta}


def test_the_checkpoint_sits_inside_an_accumulation_and_the_next_step_updates(uninterrupted):
    """Non-vacuous: micro-step 1 only accumulates (weights unchanged, accumulators nonzero); micro-step 2 applies, so
    the resumed step depends on every state group, beta and the schedule position.
    """
    steps, s0, s1, s2 = (uninterrupted[k] for k in ("steps", "s0", "s1", "s2"))
    acc = accumulator_indices(steps)
    assert len(acc) == len(steps.model.trainable_variables)
    assert first_difference((s0[0], [], []), (s1[0], [], [])) is None
    assert sum(np.count_nonzero(s1[2][i]) for i in acc) > 0
    assert changed_groups(s1, s2) == {"tv", "ntv", "ov"}
    assert steps.ebops_leaves.read_beta(s1[1]) == pytest.approx(BETA, rel=1e-7)
    assert all(np.isfinite(loss) for loss in uninterrupted["losses"])


def test_resumed_step_in_a_fresh_model_equals_the_uninterrupted_step(uninterrupted, resumed, batches):
    assert first_difference(resumed["state"], uninterrupted["s1"]) is None
    assert resumed["steps"].model is not uninterrupted["steps"].model
    s2, _ = step(resumed["steps"], resumed["state"], batches[1])
    assert first_difference(s2, uninterrupted["s2"]) is None


@pytest.mark.parametrize("leaf", ["weight", "accumulator"])
def test_control_one_perturbed_leaf_breaks_the_resume(uninterrupted, resumed, batches, leaf):
    steps = resumed["steps"]
    tv, ntv, ov = ([x.copy() for x in group] for group in resumed["state"])
    if leaf == "weight":
        i = next(i for i, v in enumerate(steps.model.trainable_variables) if v.path.endswith("kernel"))
        tv[i].flat[0] = np.nextafter(tv[i].flat[0], np.float32(np.inf))
    else:
        i = next(i for i in accumulator_indices(steps) if np.count_nonzero(ov[i]))
        ov[i] = ov[i] * np.float32(1.5)
    s2, _ = step(steps, (tv, ntv, ov), batches[1])
    assert first_difference(s2, uninterrupted["s2"]) is not None


def test_beta_controller_resumes_from_the_checkpoint_metadata(uninterrupted, resumed):
    reading = 1.2e5
    expected = uninterrupted["pid"].on_epoch_start(len(EBOPS_READINGS) + 1, reading)
    restored = BetaPIDController(**PID)
    restored.load_state_dict(resumed["meta"]["beta_controller"])
    assert restored.on_epoch_start(len(EBOPS_READINGS) + 1, reading) == expected
    # control: a controller that was not restored re-seeds at init_beta and returns another beta
    assert BetaPIDController(**PID).on_epoch_start(len(EBOPS_READINGS) + 1, reading) != expected
