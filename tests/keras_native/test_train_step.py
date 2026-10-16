"""S5 gates for ``keras_native.train_step`` on the org/main model at the test width (``mf_cases`` variant ``org``), quantized.

- ``train_step`` compiles once per ``ebops_on`` value over 3 batches, with one host matching call per step.
- Every trainable variable gets a finite, nonzero gradient from the full objective (matched loss + HGQ2 losses); control:
  the HGQ2 losses alone leave the network weights at zero gradient, and the same check reports them.
- Gradients through the ``pure_callback`` matching equal a fixed-index reference (indices solved outside, passed in as
  constants) on the full model; control: identity indices give different gradients.
- Penalty check (the spike's): the jitted step-0 loss minus an independent eager task loss equals the eager sum of the
  EBOPs terms (+ regularizers), rel <= 1e-5; control: with ``ebops_on=False`` it does not.
- Changing beta between steps keeps the compile count and scales the penalty by the beta ratio.
- PeriodicEBOPs: ``ebops_on=False`` gives exactly 0 penalty and the same regularizer sum, from the same cached executables.
- Port of ``tests/keras/test_training.py``: over 30 steps the loss falls and the quantizer variables move.
- Gradient accumulation through the matched loss: one host call per micro-batch, one update per k micro-batches.
- State handling (small Dense model): the donated state is consumed (control: ``donate=False``); the Keras variables stay
  valid, also after ``assign_to_model``, which round-trips exactly. Eval and predict compile once over 3 batches.
- Matmul precision: every dot of the traced train, eval and predict programs requests HIGHEST precision (control:
  ``matmul_precision=None``).
- ``assign_to_model`` / ``init_state`` round-trip the quantized model's state exactly after a training step, the
  non-trainable state included (control: the step changed that state, so a write that skips it fails).
- ``eval_step``'s loss is the hybrid's val/loss, task + the HGQ2 weight regularizers (no EBOPs term at training=False):
  its regularization equals the training step's on the same weights; control: it is nonzero.
- The non-trainable state the training forward updates (WRAP integer bits ``/i``, ``_ebops``) is kept: the returned
  ``ntv`` equals a reference ``stateless_call(training=True)`` over 2 steps; control: predicting with the stale ``ntv``
  differs.
- eval and predict run the model with ``training=False``; control: ``training=True`` gives different outputs.
- ``compile_counts`` counts traces, not dispatch-cache entries (numpy and ``jax.Array`` batches add a cache entry).
- ``maskformer_loss_from_config`` on the generated Polaris config: the ``clic_specs()`` loss with ``n_jobs`` 16 from the
  matcher block; control: no matcher block gives ``n_jobs`` 1.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest
import yaml

from hepattn.keras_native import matching
from hepattn.keras_native.matching import host_match
from hepattn.keras_native.objective import clic_specs, matched_loss, model_losses, permute_outputs, total_loss
from hepattn.keras_native.optim import build_optimizer
from hepattn.keras_native.train_step import TrainStep, maskformer_loss, maskformer_loss_from_config

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .mf_golden_utils import build_mf
from .test_lane_integration import real_like

LRS = {"initial": 1e-3, "max": 1e-3, "end": 1e-3, "pct_start": 0.1, "weight_decay": 1e-4, "skip_scheduler": True}
BETA = 1e-9  # penalty ~ 30 at the high spec, comparable to the task loss (measured below)
PENALTY_RTOL = 1e-5  # plan section 4 (spike G1 criterion)


def copy_state(state):
    return jax.tree.map(lambda x: jnp.array(x, copy=True), state)


class Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return host_match(*args, **kwargs)


@pytest.fixture(scope="module")
def batches():
    out = []
    for i in range(3):
        inputs, targets = clic_dummy_batch(num_events=2, offset=2 * i)
        out.append((inputs, real_like(targets)))
    return out


@pytest.fixture(scope="module")
def model(batches):
    return build_mf("org", QUANT_SPECS["high"], batches[0][0])


@pytest.fixture(scope="module")
def setup(model):
    counter = Counter()
    opt = build_optimizer(model, optimizer="AdamW", lrs_config=LRS, total_steps=100, clip_norm=0.1)
    steps = TrainStep(model, opt, maskformer_loss(clic_specs(), host_fn=counter))
    state = steps.set_beta(steps.init_state(), BETA)
    return steps, state, counter


def test_compiles_once_per_ebops_value_with_one_host_call_per_step(setup, batches):
    steps, state0, counter = setup
    state, calls0, outs = copy_state(state0), counter.calls, []
    for ebops_on in (True, False):
        for b in batches:
            state, out = steps.train_step(state, b, ebops_on=ebops_on)
            outs.append(out)
    assert steps.compile_counts()["train"] == 2, steps.compile_counts()
    assert counter.calls - calls0 == 6, f"{counter.calls - calls0} host matching calls in 6 steps"
    losses = [float(o.task_loss) for o in outs]
    assert all(np.isfinite(losses)) and len(set(losses)) == len(losses), losses
    assert all(float(o.penalty) > 0 for o in outs[:3]) and all(float(o.penalty) == 0 for o in outs[3:])


@pytest.fixture(scope="module")
def grads_fn(setup):
    steps, _, _ = setup
    return jax.jit(steps.grads, static_argnames=("ebops_on",))


def test_every_trainable_variable_gets_a_finite_nonzero_gradient(setup, batches, grads_fn, model):
    steps, state, _ = setup
    names = [v.path for v in model.trainable_variables]
    reached = np.zeros(len(names), bool)
    for b in batches:
        grads, _ = grads_fn(state, b, ebops_on=True)
        for i, g in enumerate(grads):
            g = np.asarray(g)
            assert np.isfinite(g).all(), f"non-finite gradient for {names[i]}"
            reached[i] |= bool(np.any(g != 0))
    missed = [n for n, ok in zip(names, reached, strict=True) if not ok]
    assert not missed, f"{len(missed)} variables get no gradient: {missed[:5]}"

    # control: the HGQ2 losses alone leave the network weights without a gradient, and the same check reports them
    def hgq2_only(tv, ntv, inputs):
        with steps.ebops_leaves.enabled(True):
            _, _, qlosses = model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)
        return sum(qlosses)

    control = jax.jit(jax.grad(hgq2_only))(state.tv, state.ntv, batches[0][0])
    unreached = [n for n, g in zip(names, control, strict=True) if not np.any(np.asarray(g) != 0)]
    assert len(unreached) > 50 and any(n.endswith("/kernel") for n in unreached), unreached[:5]


def test_gradients_through_matching_equal_the_fixed_index_reference(setup, batches, grads_fn, model):
    steps, state, _ = setup
    specs = clic_specs()
    inputs, targets = batches[1]

    def fixed_index_loss(tv, ntv, idx):
        with steps.ebops_leaves.enabled(True):
            outputs, _, qlosses = model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)
        losses = model_losses(permute_outputs(outputs, idx, specs), targets, specs)
        return total_loss(losses) + sum(qlosses)

    fixed_grads = jax.jit(jax.grad(fixed_index_loss))

    # the indices the step's own callback solves, computed outside it
    @jax.jit
    def solve(tv, ntv):
        outputs, _ = model.stateless_call(tv, ntv, inputs, training=True)
        return matched_loss(outputs, targets, specs)[2]

    idx = jax.tree.map(np.asarray, solve(state.tv, state.ntv))
    grads, _ = grads_fn(state, (inputs, targets), ebops_on=True)
    ref = fixed_grads(state.tv, state.ntv, idx)
    diff = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(grads, ref, strict=True))
    scale = max(float(np.abs(np.asarray(b)).max()) for b in ref)
    print(f"MEASURED matching grads: max |diff| {diff:.3e}, max |grad| {scale:.3e}")
    assert diff <= 2.8e-9 * scale, (diff, scale)  # measured 9.7e-9 at max |grad| 7.07 (1.4e-9 relative), 2x

    identity = {layer: np.tile(np.arange(v.shape[1], dtype=np.int32), (v.shape[0], 1)) for layer, v in idx.items()}
    assert any(not np.array_equal(identity[k], idx[k]) for k in idx), "the matching is the identity: control is vacuous"
    control = fixed_grads(state.tv, state.ntv, identity)
    cdiff = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(grads, control, strict=True))
    print(f"MEASURED matching grads, identity control: max |diff| {cdiff:.3e}")
    assert cdiff > 1e-3 * scale, (cdiff, scale)


def test_penalty_check_against_an_eager_reference(setup, batches, model):
    steps, state0, _ = setup
    inputs, targets = batches[0]
    tv, ntv = [np.asarray(x) for x in state0.tv], [np.asarray(x) for x in state0.ntv]
    with steps.ebops_leaves.enabled(True):
        outputs, _, qlosses = model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)  # eager
    task_ref = float(matched_loss(outputs, targets, clic_specs())[0])
    qloss_ref = float(sum(float(q) for q in qlosses))
    pen_terms, _ = steps.ebops_leaves.split_losses(qlosses)
    assert len(pen_terms) > 100 and qloss_ref > 1.0

    for ebops_on in (True, False):
        _, out = steps.train_step(copy_state(state0), batches[0], ebops_on=ebops_on)
        diff = float(out.loss) - task_ref
        err = abs(diff - qloss_ref) / qloss_ref
        print(f"MEASURED penalty check ebops_on={ebops_on}: loss - task {diff:.6f}, HGQ2 losses {qloss_ref:.6f}, rel err {err:.3e}")
        if ebops_on:
            assert err <= PENALTY_RTOL, (diff, qloss_ref, err)
            assert abs(float(out.penalty) + float(out.regularization) - qloss_ref) <= PENALTY_RTOL * qloss_ref
        else:  # control: no EBOPs terms in the loss
            assert err > 0.5, (diff, qloss_ref)


def test_beta_change_keeps_the_executable_and_scales_the_penalty(setup, batches):
    steps, state0, _ = setup
    _, out1 = steps.train_step(copy_state(state0), batches[0], ebops_on=True)
    before = steps.compile_counts()["train"]
    state2 = steps.set_beta(copy_state(state0), 7 * BETA)
    assert steps.read_beta(state2) == pytest.approx(7 * BETA, rel=1e-6)
    _, out2 = steps.train_step(state2, batches[0], ebops_on=True)
    assert steps.compile_counts()["train"] == before, "a beta change retraced the step"
    print(f"MEASURED beta x7: penalty ratio {float(out2.penalty) / float(out1.penalty):.9f}")
    assert float(out2.penalty) / float(out1.penalty) == pytest.approx(7.0, rel=2.4e-7)  # measured 6.9999992 (1.2e-7), 2x
    assert float(out2.regularization) == float(out1.regularization)
    # the read-out does not depend on beta
    assert steps.ebops(out2, 7 * BETA) == pytest.approx(steps.ebops(out1, BETA), rel=2.4e-7)


def test_periodic_ebops_skipped_step_has_no_penalty(setup, batches):
    steps, state0, _ = setup
    _, on = steps.train_step(copy_state(state0), batches[0], ebops_on=True)
    _, off = steps.train_step(copy_state(state0), batches[0], ebops_on=False)
    assert float(off.penalty) == 0.0 and float(on.penalty) > 0
    assert float(off.regularization) == pytest.approx(float(on.regularization), rel=1e-6)
    assert float(off.task_loss) == pytest.approx(float(on.task_loss), rel=1e-6), "the EBOPs switch changed the forward"
    assert steps.compile_counts()["train"] <= 2


def test_training_port_loss_falls_and_quantizers_move(setup, batches, model):
    """``tests/keras/test_training.py`` on the native step: beta 0 (task loss only), AdamW lr 1e-3, 30 steps, one batch."""
    steps, state0, _ = setup
    before_counts = steps.compile_counts()["train"]
    state = steps.set_beta(copy_state(state0), 0.0)
    names = [v.path for v in model.trainable_variables]
    before = [np.asarray(x) for x in state.tv]
    losses = []
    for _ in range(30):
        state, out = steps.train_step(state, batches[0], ebops_on=True)
        losses.append(float(out.loss))
    assert float(out.penalty) == 0.0
    print(f"MEASURED training port: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], f"training loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    moved = [n for n, a, b in zip(names, before, state.tv, strict=True) if "quantizer" in n and not np.array_equal(a, np.asarray(b))]
    assert moved, "no quantizer variable changed: quantizers are frozen"
    assert steps.compile_counts()["train"] == max(before_counts, 1)


@pytest.fixture(scope="module")
def float_model(batches):
    return build_mf("org", None, batches[0][0])


def test_accumulation_makes_one_host_call_per_micro_batch_and_one_update_per_k(float_model, batches):
    counter = Counter()
    opt = build_optimizer(float_model, optimizer="Lion", lrs_config=LRS, total_steps=100, accumulate=3)
    steps = TrainStep(float_model, opt, maskformer_loss(clic_specs(), host_fn=counter))
    state = steps.init_state()
    tv0 = [np.asarray(x) for x in state.tv]
    for i, b in enumerate(batches):
        state, _ = steps.train_step(state, b)
        changed = any(not np.array_equal(a, np.asarray(x)) for a, x in zip(tv0, state.tv, strict=True))
        assert changed == (i == 2), f"micro-step {i}: weights changed={changed}"
    assert counter.calls == 3
    assert steps.compile_counts()["train"] == 1


def small_model_and_loss():
    inputs = keras.Input((4,))
    model = keras.Model(inputs, keras.layers.Dense(3)(keras.layers.Dense(8, activation="relu")(inputs)))

    def loss_fn(outputs, targets):
        loss = jnp.mean(jnp.square(outputs - targets))
        return loss, {"mse": loss}

    return model, loss_fn


@pytest.mark.parametrize("donate", [None, True, False])
def test_state_donation_and_model_variables(donate):
    """``None``: the default, which must donate (Keras' trainer does; S5.md "G1 speed gap")."""
    model, loss_fn = small_model_and_loss()
    rng = np.random.default_rng(0)
    batch = (rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=(5, 3)).astype(np.float32))
    opt = build_optimizer(model, optimizer="AdamW", lrs_config=LRS, total_steps=100)
    steps = TrainStep(model, opt, loss_fn) if donate is None else TrainStep(model, opt, loss_fn, donate=donate)
    donate = True if donate is None else donate
    old = steps.init_state()
    new, _ = steps.train_step(old, batch)
    deleted = [x.is_deleted() for x in old.tv + old.ov]
    assert all(deleted) if donate else not any(deleted)
    # the Keras variables were copied by init_state, so they survive donation
    assert not any(v.value.is_deleted() for v in model.trainable_variables + opt.variables)
    new, _ = steps.train_step(new, batch)  # the returned state is a valid input
    steps.assign_to_model(new)
    again = steps.init_state()
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(new.tv + new.ntv + new.ov, again.tv + again.ntv + again.ov, strict=True))
    np.testing.assert_allclose(np.asarray(model(batch[0])), np.asarray(steps.predict_step(new, batch[0])), rtol=1e-6, atol=1e-7)
    steps.train_step(new, batch)  # assign_to_model copied: donating `new` must not free the Keras variables
    assert not any(v.value.is_deleted() for v in model.trainable_variables)


def test_eval_and_predict_compile_once_and_leave_the_state(setup, batches):
    steps, state, _ = setup
    ref = [np.asarray(x) for x in state.ntv]
    for b in batches:
        ev = steps.eval_step(state, b)
        steps.predict_step(state, b[0])
        assert np.isfinite(float(ev.task_loss))
    counts = steps.compile_counts()
    assert counts["eval"] == 1 and counts["predict"] == 1, counts
    assert all(np.array_equal(a, np.asarray(x)) for a, x in zip(ref, state.ntv, strict=True))


def test_steps_request_highest_matmul_precision(float_model, batches):
    def dot_precisions(matmul_precision):
        opt = build_optimizer(float_model, optimizer="AdamW", lrs_config=LRS, total_steps=100)
        steps = TrainStep(float_model, opt, maskformer_loss(clic_specs()), matmul_precision=matmul_precision)
        state = steps.init_state()
        lowered = {
            "train": steps._train.lower(state, batches[0], ebops_on=True),  # noqa: SLF001
            "eval": steps._eval.lower(state, batches[0]),  # noqa: SLF001
            "predict": steps._predict.lower(state, batches[0][0]),  # noqa: SLF001
        }
        texts = {name: low.as_text() for name, low in lowered.items()}
        return {name: [line for line in t.splitlines() if "dot_general" in line] for name, t in texts.items()}, {
            name: t.count("HIGHEST") for name, t in texts.items()
        }

    dots, _ = dot_precisions("highest")
    for name, lines in dots.items():
        loose = [line.strip()[:160] for line in lines if "precision = [HIGHEST, HIGHEST]" not in line]
        assert len(lines) > 20 and not loose, (name, len(lines), loose[:3])
    assert len(dots["train"]) > 100
    _, default = dot_precisions(None)
    assert default == {"train": 0, "eval": 0, "predict": 0}, "the control already requests HIGHEST: the check is vacuous"


def test_grad_norm_is_optional():
    model, loss_fn = small_model_and_loss()
    rng = np.random.default_rng(1)
    batch = (rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=(5, 3)).astype(np.float32))
    opt = build_optimizer(model, optimizer="AdamW", lrs_config=LRS, total_steps=100)
    with_norm = TrainStep(model, opt, loss_fn, grad_norm=True)
    grads, _ = with_norm.grads(with_norm.init_state(), batch)
    expected = float(np.sqrt(sum(np.sum(np.square(np.asarray(g, np.float64))) for g in grads)))
    _, out = with_norm.train_step(with_norm.init_state(), batch)
    assert float(out.grad_norm) == pytest.approx(expected, rel=1e-6) and expected > 0
    _, out = TrainStep(model, opt, loss_fn).train_step(with_norm.init_state(), batch)
    assert np.isnan(float(out.grad_norm))


def leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


def max_abs_diff(a, b):
    return max(float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max(initial=0.0)) for x, y in zip(leaves(a), leaves(b), strict=True))


def test_train_step_keeps_the_updated_non_trainable_state(setup, batches, model):
    """Plan 3.6: the WRAP integer-bit and ``_ebops`` assigns of the training forward come back in ``ntv``; the loop keeps them."""
    steps, state0, _ = setup
    paths = [v.path for v in model.non_trainable_variables]

    @jax.jit
    def reference_ntv(tv, ntv, inputs):
        with steps.ebops_leaves.enabled(True), jax.default_matmul_precision("highest"):
            return model.stateless_call(tv, ntv, inputs, training=True)[1]

    state = copy_state(state0)
    for step, b in enumerate(batches[:2]):
        before = copy_state(state)
        ref = reference_ntv(before.tv, before.ntv, b[0])
        state, _ = steps.train_step(state, b, ebops_on=True)
        diff = max_abs_diff(state.ntv, ref)
        changed = [p for p, a, x in zip(paths, before.ntv, state.ntv, strict=True) if not np.array_equal(np.asarray(a), np.asarray(x))]
        int_bits = [p for p in changed if p.endswith("/i")]
        print(f"MEASURED step {step}: ntv vs reference max |diff| {diff:.3e}; {len(changed)} of {len(paths)} ntv changed, {len(int_bits)} '/i'")
        assert diff == 0.0, diff  # measured 0: the step's forward is the reference program
        assert int_bits and len(changed) > len(int_bits), (len(changed), len(int_bits))

    # control: the stale ntv (dropping the updates) changes the inference outputs
    stale = state._replace(ntv=copy_state(state0).ntv)
    pdiff = max_abs_diff(steps.predict_step(state, batches[0][0]), steps.predict_step(stale, batches[0][0]))
    print(f"MEASURED predict with kept vs stale ntv: max |diff| {pdiff:.3e}")
    assert pdiff > 1e-5, pdiff


def test_eval_and_predict_run_in_inference_mode(setup, batches):
    """References are jitted too: eager vs jitted forwards differ by up to 5.7e-3 here (fixed-point rounding flips)."""
    steps, state, _ = setup
    inputs, targets = batches[1]

    def forward(training):
        @jax.jit
        def fn(tv, ntv, x):
            with steps.ebops_leaves.enabled(training), jax.default_matmul_precision("highest"):
                return steps.model.stateless_call(tv, ntv, x, training=training)[0]

        return fn(state.tv, state.ntv, inputs)

    ref, train_mode = forward(False), forward(True)
    ev = steps.eval_step(state, (inputs, targets))
    pred = steps.predict_step(state, inputs)
    d_eval, d_pred, d_ctrl = max_abs_diff(ev.outputs, ref), max_abs_diff(pred, ref), max_abs_diff(train_mode, ref)
    print(f"MEASURED eval/predict vs jitted training=False: {d_eval:.3e} / {d_pred:.3e}; training=True control {d_ctrl:.3e}")
    assert d_eval == 0.0 and d_pred == 0.0, (d_eval, d_pred)  # measured 0 (same program)
    assert d_ctrl > 1e-4, d_ctrl


def test_compile_counts_are_traces_not_dispatch_cache_entries():
    model, loss_fn = small_model_and_loss()
    rng = np.random.default_rng(2)
    batch = (rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=(5, 3)).astype(np.float32))
    opt = build_optimizer(model, optimizer="AdamW", lrs_config=LRS, total_steps=100)
    steps = TrainStep(model, opt, loss_fn)
    state, _ = steps.train_step(steps.init_state(), batch)
    state, _ = steps.train_step(state, jax.device_put(batch))
    state, _ = steps.train_step(state, batch)
    assert steps.compile_counts()["train"] == 1, steps.compile_counts()
    # control: jax.jit's dispatch cache has one entry per argument type, so it cannot serve as a compile count
    assert steps._train._cache_size() == 2  # noqa: SLF001


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generated_config(tmp_path_factory):
    out = tmp_path_factory.mktemp("cfg") / "polaris_hgq.yaml"
    env = {**os.environ, "HGQ_CONFIG_OUT": str(out)}
    subprocess.run([sys.executable, str(REPO / "polaris/03_make_config.py")], env=env, check=True, capture_output=True, timeout=120)
    return yaml.safe_load(out.read_text())


def test_maskformer_loss_from_config_reads_the_tasks_and_the_matcher(setup, batches, generated_config, monkeypatch):
    steps, state, _ = setup
    inputs, targets = batches[0]
    outputs = steps.predict_step(state, inputs)
    seen = []

    def recording_host_match(*args, n_jobs=1):
        seen.append(n_jobs)
        return host_match(*args, n_jobs=n_jobs)

    monkeypatch.setattr(matching, "host_match", recording_host_match)
    init_args = generated_config["model"]["model"]["init_args"]
    expected = float(maskformer_loss(clic_specs())(outputs, targets)[0])
    assert float(maskformer_loss_from_config(init_args)(outputs, targets)[0]) == expected
    assert seen == [1, 16], seen
    # control: without a matcher block the host matching runs in one thread
    float(maskformer_loss_from_config({k: v for k, v in init_args.items() if k != "matcher"})(outputs, targets)[0])
    assert seen == [1, 16, 1], seen


def test_assign_to_model_round_trips_the_quantized_state(setup, batches, model):
    """S6 calls ``assign_to_model`` before ``save_weights`` / export: the quantizer state (``/i``, ``_beta``, ``_ebops``) must go too."""
    steps, state0, _ = setup
    original = steps.init_state()  # restored at the end: the model is shared by the module's tests
    state, _ = steps.train_step(copy_state(state0), batches[0], ebops_on=True)
    assert len(state.ntv) > 1000
    changed_ntv = sum(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(original.ntv, state.ntv, strict=True))
    assert changed_ntv > 100, f"the step changed only {changed_ntv} ntv arrays: the ntv check would be vacuous"
    try:
        steps.assign_to_model(state)
        again = steps.init_state()
        for part in ("tv", "ntv", "ov"):
            mismatched = [
                i
                for i, (a, b) in enumerate(zip(getattr(state, part), getattr(again, part), strict=True))
                if not np.array_equal(np.asarray(a), np.asarray(b))
            ]
            assert not mismatched, f"{part}: {len(mismatched)} arrays differ after assign_to_model, e.g. {mismatched[:3]}"
        assert [np.asarray(v.value).dtype for v in model.non_trainable_variables] == [np.asarray(x).dtype for x in state.ntv]
    finally:
        steps.assign_to_model(original)


def test_eval_loss_is_task_plus_regularization(setup, batches):
    """The hybrid's val/loss: ``MPflowHGQ.aggregate_losses`` adds ``quant_losses()`` (the regularizers at training=False)."""
    steps, state0, _ = setup
    ev = steps.eval_step(state0, batches[0])
    _, out = steps.train_step(copy_state(state0), batches[0], ebops_on=True)
    reg, task = float(ev.regularization), float(ev.task_loss)
    print(f"MEASURED eval: task {task:.6f}, regularization {reg:.6f}; train-step regularization {float(out.regularization):.6f}")
    assert reg > 0.0, "no regularizer terms: the check is vacuous"
    # the regularizers read only the weights, so the training step on the same state gives the same sum
    assert reg == pytest.approx(float(out.regularization), rel=1e-6)
    assert float(ev.loss) == pytest.approx(task + reg, rel=1e-7)
