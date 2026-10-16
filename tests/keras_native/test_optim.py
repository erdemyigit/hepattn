"""S5 gates for ``keras_native.optim`` (plan KERAS_JAX_PLAN.md 3.7).

- OneCycle LR equals torch ``OneCycleLR`` at 20 sampled optimizer steps, with and without gradient accumulation (the
  Keras schedule sees micro-steps; control: without the ``steps_per_update`` division it runs k times too fast).
  torch also cycles beta1; ours keeps it fixed (documented deviation, asserted so it cannot change silently).
- OneCycle holds its value past ``total_steps`` (control: the unclamped anneal moves on).
- Weight decay skips exactly the variables with ``quantizer`` in their path, for AdamW and Lion, by state (``_use_weight_decay``)
  and by effect (a zero-gradient step only decays); control: ``var_names=["quantizer"]`` excludes none of them. The
  expected sets come from the test's own path rule and from the hybrid's groups on the same model
  (``KerasMaskFormer.trainable_parameter_groups``: counts, element totals, variable kinds), not from ``quantizer_variables``.
- AdamW and Lion steps against torch ``AdamW`` and ``lion_pytorch.Lion`` on the same gradients.
- Gradient accumulation: k micro-batches of a decomposable loss give the same weights as one k-times batch; the
  accumulated gradient is the MEAN (control: the summed gradient gives different weights), and applying each micro-batch
  separately differs.
- The factory reads the generated Polaris config (AdamW, LR 5e-5, clip 0.1, accum 4) and org ``base.yaml`` (Lion).
"""

import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import keras
import numpy as np
import pytest
import torch
import yaml

from hepattn.keras_native.optim import OneCycleLR, build_optimizer, optimizer_from_config
from hepattn.keras_native.train_step import TrainStep

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .mf_golden_utils import build_mf

REPO = Path(__file__).resolve().parents[2]
LRS = {"initial": 1e-6, "max": 8e-5, "end": 1e-6, "pct_start": 0.05, "weight_decay": 1e-4}  # org base.yaml
TOTAL = 400
SAMPLED = [0, 1, 2, 10, 18, 19, 20, 21, 22, 40, 100, 150, 200, 250, 300, 350, 397, 398, 399, 400]
LR_RTOL = 2.3e-6  # measured max rel 1.12e-6 (float32 schedule vs torch's float64), x2


def torch_onecycle(lrs: dict, total: int, steps: list[int]) -> tuple[list[float], list[float]]:
    """Torch OneCycleLR as MPflowHGQ.configure_optimizers builds it: (lr, beta1) after ``n`` scheduler steps."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=lrs["initial"], weight_decay=lrs["weight_decay"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lrs["max"],
        total_steps=total,
        div_factor=lrs["max"] / lrs["initial"],
        final_div_factor=lrs["initial"] / lrs["end"],
        pct_start=float(lrs["pct_start"]),
    )
    lrs_out, betas, n = [], [], 0
    for target in steps:
        while n < target:
            opt.step()
            sch.step()
            n += 1
        lrs_out.append(opt.param_groups[0]["lr"])
        betas.append(opt.param_groups[0]["betas"][0])
    return lrs_out, betas


def keras_schedule(lrs: dict, total: int, steps_per_update: int = 1) -> OneCycleLR:
    return OneCycleLR(
        max_lr=lrs["max"],
        total_steps=total,
        div_factor=lrs["max"] / lrs["initial"],
        final_div_factor=lrs["initial"] / lrs["end"],
        pct_start=lrs["pct_start"],
        steps_per_update=steps_per_update,
    )


def rel(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b) / np.abs(b)


@pytest.mark.parametrize("k", [1, 4])
def test_onecycle_equals_torch_at_20_steps(k):
    ref, betas = torch_onecycle(LRS, TOTAL, SAMPLED)
    sched = keras_schedule(LRS, TOTAL, steps_per_update=k)
    # Keras evaluates the schedule at its micro-step counter; the k-th micro-step of optimizer step n is k*n + k - 1
    ours = [float(sched(k * n + k - 1)) for n in SAMPLED]
    assert rel(ours, ref).max() <= LR_RTOL, list(zip(SAMPLED, ours, ref, strict=True))
    assert max(ref) == pytest.approx(LRS["max"], rel=1e-9) and min(ref) < LRS["initial"], "the samples miss the peak or the tail"
    if k > 1:  # control: the schedule without the division runs k times too fast
        fast = keras_schedule(LRS, TOTAL)
        assert rel([float(fast(k * n + k - 1)) for n in SAMPLED], ref).max() > 0.1
    # documented deviation: torch cycles beta1 (0.95 -> 0.85 -> 0.95); ours stays 0.9
    assert min(betas) == pytest.approx(0.85) and max(betas) == pytest.approx(0.95)


def test_onecycle_holds_its_value_past_total_steps():
    sched = keras_schedule(LRS, TOTAL)
    at_end = float(sched(TOTAL))
    assert [float(sched(TOTAL + 1)), float(sched(TOTAL + 50))] == [at_end, at_end]
    # control: without the clamp the cos anneal moves on (torch raises there instead)
    pct = (TOTAL + 50 - sched.phase1_end) / (sched.phase2_end - sched.phase1_end)
    unclamped = sched.min_lr + (sched.max_lr - sched.min_lr) / 2 * (np.cos(np.pi * pct) + 1)
    assert abs(unclamped - at_end) > at_end, (unclamped, at_end)  # measured 4.46e-6 vs 1.00e-6


def test_optimizer_reads_the_schedule_at_the_optimizer_step():
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(2)])
    opt = build_optimizer(model, optimizer="AdamW", lrs_config=LRS, total_steps=TOTAL, accumulate=4)
    ref, _ = torch_onecycle(LRS, TOTAL, [0, 5, 20])
    for n, lr in zip([0, 5, 20], ref, strict=True):
        opt._iterations.assign(4 * n + 3)  # noqa: SLF001  (the micro-step of the n-th update)
        assert float(opt.learning_rate) == pytest.approx(lr, rel=LR_RTOL)
        assert int(opt.iterations) == n


def test_onecycle_validation():
    with pytest.raises(ValueError, match="divides by zero"):
        keras_schedule(LRS, 20)  # pct_start * total = 1: torch's step 0 divides 0 by 0
    with pytest.raises(ValueError, match="steps_per_update"):
        keras_schedule(LRS, TOTAL, steps_per_update=0)


@pytest.fixture(scope="module")
def qmodel():
    inputs, _ = clic_dummy_batch(num_events=2)
    return build_mf("org", QUANT_SPECS["high"], inputs)


# The hybrid's groups on the same model (org test model, QUANT_SPECS["high"]): KerasMaskFormer.trainable_parameter_groups,
# measured by docs/hgq/port_log/S5_hybrid_param_groups.py (S5_evidence/hybrid_param_groups.log). Kinds are the last path
# component; the hybrid names its two torch parameters decoder.initial_queries and encoder.register_tokens.
HYBRID_GROUPS = {
    "decay": (334, 822123, {"bias": 144, "initial_queries": 1, "kernel": 180, "register_tokens": 1, "tail": 8}),
    "quantizer": (982, 15093029, {"b": 288, "f": 468, "i": 226}),
}


def group_summary(variables) -> tuple[int, int, dict]:
    kinds: dict[str, int] = {}
    for v in variables:
        kind = v.path.rsplit("/", 1)[-1]
        kinds[kind] = kinds.get(kind, 0) + 1
    return len(variables), int(sum(int(np.prod(v.shape)) for v in variables)), dict(sorted(kinds.items()))


@pytest.mark.parametrize("kind", ["AdamW", "Lion"])
def test_weight_decay_excludes_exactly_the_quantizer_paths(qmodel, kind):
    opt = build_optimizer(qmodel, optimizer=kind, lrs_config={**LRS, "skip_scheduler": True, "initial": 1e-2}, total_steps=TOTAL)
    variables = qmodel.trainable_variables
    excluded = [v for v in variables if not opt._use_weight_decay(v)]  # noqa: SLF001
    decayed = [v for v in variables if opt._use_weight_decay(v)]  # noqa: SLF001
    # the hybrid rule, written out here (trainable_parameter_groups: 'quantizer' in the path, '/beta' never trainable)
    quant = {v.path for v in variables if "quantizer" in v.path}
    assert {v.path for v in excluded} == quant
    assert not any(v.path.endswith("/beta") for v in variables), "HGQ2 beta must not be trainable"
    assert group_summary(excluded) == HYBRID_GROUPS["quantizer"], group_summary(excluded)
    assert group_summary(decayed) == HYBRID_GROUPS["decay"], group_summary(decayed)

    # control (plan 3.7): matching by name excludes nothing, because HGQ2 names its bitwidths b / i / f
    by_name = keras.optimizers.AdamW(weight_decay=1e-4)
    by_name.exclude_from_weight_decay(var_names=["quantizer"])
    by_name.build(variables)
    assert sum(not by_name._use_weight_decay(v) for v in variables) == 0  # noqa: SLF001

    # effect: one zero-gradient step only decays, w -> w * (1 - lr * wd), and leaves the quantizer variables alone
    tv = [v.value for v in variables]
    new_tv, _ = opt.stateless_apply([v.value for v in opt.variables], [jnp.zeros_like(x) for x in tv], tv)
    for var, old, new in zip(variables, tv, new_tv, strict=True):
        old, new = np.asarray(old), np.asarray(new)
        expected = old if var.path in quant else old * np.float32(1 - 1e-2 * 1e-4)
        # Keras computes w - w * wd * lr in float32: measured 1.19e-7 relative (1 ulp) from w * (1 - lr * wd); the decay
        # itself is 1e-6 relative, so 2.4e-7 still separates decayed from untouched
        np.testing.assert_allclose(new, expected, rtol=2.4e-7, atol=0, err_msg=var.path)
    assert sum(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(tv, new_tv, strict=True)) > 100


def torch_steps(kind: str, w0: np.ndarray, grads: list[np.ndarray], lr: float, wd: float, eps: float) -> np.ndarray:
    param = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float64))
    if kind == "AdamW":
        opt = torch.optim.AdamW([param], lr=lr, weight_decay=wd, eps=eps)
    else:
        from lion_pytorch import Lion  # noqa: PLC0415

        opt = Lion([param], lr=lr, weight_decay=wd)
    for g in grads:
        param.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
    return param.detach().numpy()


def keras_steps(kind: str, w0: np.ndarray, grads: list[np.ndarray], lr: float, wd: float, eps: float) -> np.ndarray:
    model = keras.Sequential([keras.Input((w0.size,)), keras.layers.Dense(1, use_bias=False)])
    model.layers[0].kernel.assign(w0[:, None])
    lrs = {**LRS, "skip_scheduler": True, "initial": lr, "weight_decay": wd}
    opt = build_optimizer(model, optimizer=kind, lrs_config=lrs, total_steps=TOTAL, adam_epsilon=eps)
    tv, ov = [model.layers[0].kernel.value], [v.value for v in opt.variables]
    for g in grads:
        tv, ov = opt.stateless_apply(ov, [jnp.asarray(g[:, None])], tv)
    return np.asarray(tv[0])[:, 0]


@pytest.mark.parametrize("kind", ["AdamW", "Lion"])
def test_update_rule_equals_torch(kind):
    """5 steps of lr 1e-3, wd 1e-2 on the same gradients ~N(0, 1e-3), float32 here vs float64 torch.

    With epsilon 0 the AdamW rules are the same: measured max abs 6.5e-7 (float32 rounding at |w| <= 1.7; tolerance
    1.3e-6, 2x). Lion: measured 4.1e-7 (tolerance 8.2e-7). The weights move by up to 5e-3. With torch's epsilon 1e-8 the
    AdamW difference is the epsilon placement (module docstring): measured 7.9e-6 (0.16% of the movement), bounded at 2x.
    """
    rng = np.random.default_rng(0)
    w0 = rng.normal(size=(64,)).astype(np.float32)
    grads = [rng.normal(scale=1e-3, size=(64,)).astype(np.float32) for _ in range(5)]
    eps, tol = (0.0, 1.3e-6) if kind == "AdamW" else (1e-8, 8.2e-7)
    ours = keras_steps(kind, w0, grads, 1e-3, 1e-2, eps)
    ref = torch_steps(kind, w0, grads, 1e-3, 1e-2, eps)
    assert np.abs(ours - ref).max() <= tol
    assert np.abs(ours - w0).max() > 1e-3, "the steps did not move the weights"
    if kind == "AdamW":
        with_eps = np.abs(keras_steps(kind, w0, grads, 1e-3, 1e-2, 1e-8) - torch_steps(kind, w0, grads, 1e-3, 1e-2, 1e-8)).max()
        assert tol < with_eps <= 1.6e-5, with_eps


def surrogate_loss(outputs, targets):
    """Mean over events of a per-event mean square of the final mask logits: decomposable over micro-batches."""
    del targets
    logit = outputs["final"]["mask"]["pflow_node_logit"]
    logit = jnp.where(jnp.abs(logit) < 1e30, logit, 0.0)
    per_event = jnp.mean(jnp.square(logit), axis=(1, 2))
    return jnp.mean(per_event), {"surrogate": jnp.mean(per_event)}


def test_accumulation_averages_k_micro_batches_like_one_k_times_batch():
    k = 2
    micro = [clic_dummy_batch(num_events=2, offset=2 * i) for i in range(k)]
    big = tuple({key: np.concatenate([m[j][key] for m in micro]) for key in micro[0][j]} for j in range(2))
    model = build_mf("org", None, micro[0][0])  # float: the forward of an event does not depend on the batch
    # epsilon 1 makes Adam's step proportional to the gradient, so a summed gradient would give a k-times larger step
    lrs = {**LRS, "skip_scheduler": True, "initial": 1e-2, "weight_decay": 0.0}

    def run(accumulate, batches, loss_fn=surrogate_loss):
        opt = build_optimizer(model, optimizer="AdamW", lrs_config=lrs, total_steps=TOTAL, accumulate=accumulate, adam_epsilon=1.0)
        steps = TrainStep(model, opt, loss_fn)
        state = steps.init_state()
        for b in batches:
            state, _ = steps.train_step(state, b, ebops_on=False)
        return [np.asarray(x) for x in state.tv], steps, state

    w0 = [np.asarray(v.value) for v in model.trainable_variables]
    accumulated, steps, state = run(k, micro)
    one_big, _, _ = run(1, [big])
    counter = [x for v, x in zip(steps.optimizer.variables, state.ov, strict=True) if v.name == "iteration"]
    assert [int(x) for x in counter] == [k], "the micro-step counter"
    assert int(steps.optimizer.iterations) == 0, "the model's optimizer variables are not touched by the stateless steps"
    moved = max(float(np.abs(a - b).max()) for a, b in zip(accumulated, w0, strict=True))
    diff = max(float(np.abs(a - b).max()) for a, b in zip(accumulated, one_big, strict=True))
    # measured: weights move 5.7e-5, accumulated vs one big batch 1.5e-8 (2.6e-4 of the move; tolerance 2x); controls:
    # summed gradient 5.7e-5, one update per micro-batch 1.1e-4 away from the accumulated weights
    assert diff <= 5.2e-4 * moved, (diff, moved)

    def summed(outputs, targets):  # control: the gradient of the sum over micro-batches (loss x k)
        loss, aux = surrogate_loss(outputs, targets)
        return k * loss, aux

    summed_w, _, _ = run(1, [big], loss_fn=summed)
    separate, _, _ = run(1, micro)  # control: one update per micro-batch
    for control in (summed_w, separate):
        cdiff = max(float(np.abs(a - b).max()) for a, b in zip(control, accumulated, strict=True))
        assert cdiff > 0.1 * moved, (cdiff, moved)


@pytest.fixture(scope="module")
def generated_config(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("cfg") / "polaris_hgq.yaml"
    env = {**os.environ, "HGQ_CONFIG_OUT": str(out)}
    subprocess.run([sys.executable, str(REPO / "polaris/03_make_config.py")], env=env, check=True, capture_output=True, timeout=120)
    return yaml.safe_load(out.read_text())


def test_factory_reads_the_generated_polaris_config_and_org_base(generated_config):
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(2)])
    cfg = generated_config
    opt = optimizer_from_config(cfg["model"], cfg["trainer"], model, total_steps=1000)
    assert isinstance(opt, keras.optimizers.AdamW)
    assert opt.global_clipnorm == pytest.approx(0.1)
    assert opt.gradient_accumulation_steps == 4
    assert opt.epsilon == pytest.approx(1e-8)
    assert opt._learning_rate.max_lr == pytest.approx(5e-5)  # noqa: SLF001
    assert opt._learning_rate.steps_per_update == 4  # noqa: SLF001

    base = yaml.safe_load((REPO / "src/hepattn/experiments/clic/configs/base.yaml").read_text())
    lion = optimizer_from_config(base["model"], base["trainer"], model, total_steps=1000)
    assert isinstance(lion, keras.optimizers.Lion)
    assert lion.gradient_accumulation_steps is None
    assert lion.weight_decay == pytest.approx(1e-4)
    with pytest.raises(NotImplementedError, match="norm"):
        optimizer_from_config(base["model"], {**base["trainer"], "gradient_clip_algorithm": "value"}, model, total_steps=1000)
