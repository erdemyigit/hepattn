"""``lrs_config.quantizer_lr_scale`` (``keras_native.optim``): the trainable quantizer variables step with LR x scale;
``lrs_config.quantizer_epsilon``: and with their own AdamW epsilon (tests at the end).

A first optimizer step from zero state moves a variable by exactly ``lr x sign(g)`` for Lion and by ``lr x g / (|g| +
eps')`` (~ lr) for AdamW, whatever |g|; so after one step the quantizer variables moved ``scale`` times as far as the
others. The step goes through ``stateless_apply`` as ``train_step`` drives it. Scale 1 builds the stock Keras classes
(the default runs are unchanged). The config reader and the loop pass the key on.
"""

import keras
import numpy as np
import pytest
import yaml

from hepattn.keras_native import config as nconfig
from hepattn.keras_native import loop
from hepattn.keras_native.optim import ScaledAdamW, ScaledLion, build_optimizer, quantizer_variables

from .conftest import REPO_ROOT

LRS = {"initial": 1e-3, "max": 1e-3, "end": 1e-3, "pct_start": 0.05, "weight_decay": 0.0, "skip_scheduler": True}


def toy_model() -> keras.Model:
    """Two Dense layers; the second's path contains ``quantizer`` (the rule of ``quantizer_variables``)."""
    x = keras.Input((3,))
    y = keras.layers.Dense(2, name="plain")(x)
    y = keras.layers.Dense(2, name="fake_quantizer")(y)
    return keras.Model(x, y)


def one_step(model, opt, grad_value: float) -> tuple[np.ndarray, np.ndarray]:
    """|change| of every quantizer and every other trainable variable after one stateless step with constant grads."""
    tv = [v.value for v in model.trainable_variables]
    grads = [keras.ops.full(v.shape, grad_value) for v in model.trainable_variables]
    new_tv, _ = opt.stateless_apply([v.value for v in opt.variables], grads, tv)
    qids = {id(v) for v in quantizer_variables(model)}
    moved = [np.abs(np.asarray(n) - np.asarray(o)).ravel() for n, o in zip(new_tv, tv, strict=True)]
    q = np.concatenate([m for m, v in zip(moved, model.trainable_variables, strict=True) if id(v) in qids])
    other = np.concatenate([m for m, v in zip(moved, model.trainable_variables, strict=True) if id(v) not in qids])
    return q, other


@pytest.mark.parametrize("name", ["Lion", "AdamW"])
@pytest.mark.parametrize("grad_value", [1e-6, 3.0])
def test_quantizer_variables_step_scale_times_further(name, grad_value):
    model = toy_model()
    assert len(quantizer_variables(model)) == 2 and len(model.trainable_variables) == 4
    opt = build_optimizer(model, optimizer=name, lrs_config={**LRS, "quantizer_lr_scale": 50.0}, total_steps=10)
    assert isinstance(opt, ScaledLion if name == "Lion" else ScaledAdamW)
    q, other = one_step(model, opt, grad_value)
    lr = LRS["max"]
    # the same step size in each group, 50x apart
    np.testing.assert_allclose(q, 50.0 * other[0], rtol=1e-3)  # |new - old| of float32 values ~0.5: ~1e-4 relative
    np.testing.assert_allclose(other, other[0], rtol=1e-3)
    # Lion: exactly lr x sign(g); AdamW: ~lr, except where Keras' epsilon (added to sqrt(v) before the bias correction)
    # is comparable to sqrt(v) = 0.032 |g|: at |g| = 1e-6 the step is 0.76 lr
    expected = lr if name == "Lion" or grad_value > 1e-3 else lr * 0.1 * grad_value / (np.sqrt(1e-3) * grad_value + 1e-8) * np.sqrt(1e-3) / 0.1
    np.testing.assert_allclose(other[0], expected, rtol=1e-3)


@pytest.mark.parametrize("name", ["Lion", "AdamW"])
def test_scale_one_is_the_stock_optimizer(name):
    model = toy_model()
    opt = build_optimizer(model, optimizer=name, lrs_config=LRS, total_steps=10)
    assert type(opt) is (keras.optimizers.Lion if name == "Lion" else keras.optimizers.AdamW)
    q, other = one_step(model, opt, 1.0)
    np.testing.assert_allclose(q, LRS["max"], rtol=1e-3)
    np.testing.assert_allclose(other, LRS["max"], rtol=1e-3)
    with pytest.raises(ValueError, match="quantizer_lr_scale"):
        build_optimizer(toy_model(), optimizer=name, lrs_config={**LRS, "quantizer_lr_scale": 0.0}, total_steps=10)


def test_config_and_loop_pass_the_scale_on(tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "tests/keras_native/s6_fixtures/polaris_normalised.yaml").read_text())
    assert nconfig.load_config(write(tmp_path / "a.yaml", cfg)).optimizer.quantizer_lr_scale == 1.0
    cfg["model"]["lrs_config"]["quantizer_lr_scale"] = 64.0
    rc = nconfig.load_config(write(tmp_path / "b.yaml", cfg))
    assert rc.optimizer.quantizer_lr_scale == 64.0
    assert loop.lrs_config(rc)["quantizer_lr_scale"] == 64.0


def write(path, cfg):
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_jitted_train_step_on_a_quantized_maskformer():
    """Through ``TrainStep`` (jit, stateless): Lion moves every bitwidth with a gradient by scale x lr, the weights by lr."""
    import jax.numpy as jnp  # noqa: PLC0415

    from hepattn.keras_native.train_step import TrainStep  # noqa: PLC0415

    from .golden.cases import QUANT_SPECS  # noqa: PLC0415
    from .golden.mf_cases import clic_dummy_batch  # noqa: PLC0415
    from .mf_golden_utils import build_mf  # noqa: PLC0415

    batch = clic_dummy_batch(num_events=2)
    model = build_mf("org", QUANT_SPECS["high"], batch[0])

    def loss_fn(outputs, targets):  # noqa: ARG001
        logit = outputs["final"]["classification"]["pflow_logit"]
        return jnp.mean(jnp.square(logit)), {}

    lr, scale = 1e-4, 10.0
    w0 = {v.path: np.asarray(v.value) for v in model.trainable_variables}
    opt = build_optimizer(
        model, optimizer="Lion", lrs_config={**LRS, "initial": lr, "max": lr, "end": lr, "quantizer_lr_scale": scale}, total_steps=10
    )
    steps = TrainStep(model, opt, loss_fn)
    state, _ = steps.train_step(steps.init_state(), batch, ebops_on=False)
    qpaths = {v.path for v in quantizer_variables(model)}
    moved = {v.path: float(np.abs(np.asarray(x) - w0[v.path]).max()) for v, x in zip(model.trainable_variables, state.tv, strict=True)}
    q_moved = [m for p, m in moved.items() if p in qpaths and m > 0]
    w_moved = [m for p, m in moved.items() if p not in qpaths and m > 0]
    assert len(q_moved) > 10 and len(w_moved) > 10, (len(q_moved), len(w_moved))
    # sign updates: exactly lr x scale / lr per element (float32 rounding of |new - old| aside)
    np.testing.assert_allclose(max(q_moved), scale * lr, rtol=1e-2)
    np.testing.assert_allclose(max(w_moved), lr, rtol=1e-2)


def moves(model, opt, grad_value: float, steps: int, jit: bool) -> tuple[np.ndarray, np.ndarray]:
    """Total |change| of the quantizer and of the other variables after ``steps`` stateless steps with constant grads."""
    import jax  # noqa: PLC0415

    apply = jax.jit(opt.stateless_apply) if jit else opt.stateless_apply
    tv0 = [np.asarray(v.value) for v in model.trainable_variables]
    tv, ov = [v.value for v in model.trainable_variables], [v.value for v in opt.variables]
    grads = [keras.ops.full(v.shape, grad_value) for v in model.trainable_variables]
    for _ in range(steps):
        tv, ov = apply(ov, grads, tv)
    qids = {id(v) for v in quantizer_variables(model)}
    moved = [(id(v) in qids, np.abs(np.asarray(n, np.float64) - o).ravel()) for v, n, o in zip(model.trainable_variables, tv, tv0, strict=True)]
    return np.concatenate([m for q, m in moved if q]), np.concatenate([m for q, m in moved if not q])


@pytest.mark.parametrize("jit", [False, True])
def test_quantizer_epsilon_is_the_adamw_epsilon_of_the_quantizer_group(jit):
    """Gradients of 1e-10 (below eps 1e-8, as the clipped bitwidth gradients of the warm start, ~1e-10): over 3 steps the
    quantizer variables move as a stock AdamW with epsilon 1e-14 moves them (~lr per step), the others as one with
    epsilon 1e-8 (~3e-4 lr per step); the second and third steps show that the epsilon is restored after each variable.
    """
    g, steps = 1e-10, 3
    lrs = {**LRS, "max": 1.0, "initial": 1.0, "end": 1.0}  # |change| ~1 per step: float32 resolves it on values ~0.5
    model = toy_model()
    grouped = build_optimizer(model, optimizer="AdamW", lrs_config={**lrs, "quantizer_epsilon": 1e-14}, total_steps=10)
    assert isinstance(grouped, ScaledAdamW)
    q, other = moves(model, grouped, g, steps, jit)
    ref_q, _ = moves(model, build_optimizer(model, optimizer="AdamW", lrs_config=lrs, total_steps=10, adam_epsilon=1e-14), g, steps, jit)
    _, ref_other = moves(model, build_optimizer(model, optimizer="AdamW", lrs_config=lrs, total_steps=10), g, steps, jit)
    np.testing.assert_allclose(q, ref_q, rtol=1e-5)
    np.testing.assert_allclose(other, ref_other, rtol=1e-3, atol=1e-7)
    # non-vacuous: the two epsilons give steps 3 orders of magnitude apart (~1 vs ~3e-4 per step at this LR of 1)
    assert q.min() > 2.5 and other.max() < 3e-3, (q.min(), other.max())


def test_quantizer_epsilon_needs_adamw_and_a_positive_value():
    with pytest.raises(ValueError, match="quantizer_epsilon"):
        build_optimizer(toy_model(), optimizer="Lion", lrs_config={**LRS, "quantizer_epsilon": 1e-12}, total_steps=10)
    with pytest.raises(ValueError, match="quantizer_epsilon"):
        build_optimizer(toy_model(), optimizer="AdamW", lrs_config={**LRS, "quantizer_epsilon": 0.0}, total_steps=10)


def test_config_and_loop_pass_the_quantizer_epsilon_on(tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "tests/keras_native/s6_fixtures/polaris_normalised.yaml").read_text())
    assert nconfig.load_config(write(tmp_path / "a.yaml", cfg)).optimizer.quantizer_epsilon is None
    cfg["model"]["lrs_config"]["quantizer_epsilon"] = 1e-12
    rc = nconfig.load_config(write(tmp_path / "b.yaml", cfg))
    assert rc.optimizer.quantizer_epsilon == 1e-12
    assert loop.lrs_config(rc)["quantizer_epsilon"] == 1e-12
