"""S2 JIT gate: each quantized layer trains under ``jax.jit(jax.value_and_grad(stateless_call ...))``.

For every golden case in the high-bitwidth mode (hybrid weights and quantizer state loaded):
- 3 calls with new inputs of the same shapes and the returned non-trainable state compile once (``_cache_size() == 1``);
- every trainable variable gets a nonzero gradient from a surrogate loss mean(out^2) + sum(layer losses);
- the EBOPs loss is present and > 0: every EBOPs-enabled leaf writes ebops > 0 into the returned state, and the
  losses sum is > 0. ``test_ebops_loss_term`` separates the EBOPs term from the other HGQ2 loss terms with an
  eval-mode control (no ebops written there).
"""

import jax
import keras
import numpy as np
import pytest

from .golden.cases import CASES
from .golden_utils import build_from_golden, inputs_of, native_leaves

MODE = "high"


def jitted_step(case: str, module, training: bool = True):
    kind = CASES[case]["kind"]

    def call(tv, ntv, inputs):
        if kind == "dense":
            args, kwargs = (inputs["x"],), {}
        elif kind == "innet":
            args, kwargs = (inputs,), {}
        elif kind == "enc":
            args, kwargs = (inputs["x"],), {"kv_mask": inputs["kv_mask"]}
        elif kind == "attn":
            args, kwargs = (inputs["q"],), {"k": inputs.get("k"), "kv_mask": inputs.get("kv_mask"), "attn_mask": inputs.get("attn_mask")}
        else:
            args, kwargs = (inputs["q"],), {"k": inputs.get("k"), "v": inputs.get("k"), "attn_mask": inputs.get("attn_mask")}
        return module.stateless_call(tv, ntv, *args, return_losses=True, training=training, **kwargs)

    def loss_fn(tv, ntv, inputs):
        out, new_ntv, losses = call(tv, ntv, inputs)
        qloss = sum(losses) if losses else 0.0
        return keras.ops.mean(keras.ops.square(out)) + qloss, (new_ntv, qloss, len(losses))

    return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


def perturbed_inputs(inputs: dict, seed: int) -> dict:
    """New float inputs of the same shapes (masks kept), so each call sees different data."""
    rng = np.random.default_rng(seed)
    return {k: (v + rng.normal(0, 0.1, v.shape).astype(v.dtype)) if v.dtype == np.float32 else v for k, v in inputs.items()}


def ebops_indices(module) -> dict[str, list[int]]:
    """Positions in ``non_trainable_variables`` of each EBOPs-enabled leaf's ebops variables (own + sublayers)."""
    position = {id(v): i for i, v in enumerate(module.non_trainable_variables)}
    out = {}
    for name, leaf in native_leaves(module).items():
        idx = [position[id(v)] for v in leaf.non_trainable_variables if v.name == "ebops"]
        if idx:
            out[name] = idx
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_layer_trains_under_jit(case):
    module, _, golden = build_from_golden(case, MODE)
    inputs = inputs_of(golden)
    tv = [v.value for v in module.trainable_variables]
    ntv = [v.value for v in module.non_trainable_variables]
    step = jitted_step(case, module)

    nonzero = np.zeros(len(tv), dtype=bool)
    for call in range(3):
        (loss, (ntv, qloss, n_losses)), grads = step(tv, ntv, perturbed_inputs(inputs, call))
        nonzero |= np.array([bool(np.any(np.asarray(g) != 0)) for g in grads])
        assert np.isfinite(float(loss))
    assert step._cache_size() == 1, f"{case}: {step._cache_size()} compilations over 3 calls"  # noqa: SLF001

    names = [v.path for v in module.trainable_variables]
    assert nonzero.all(), f"{case}: zero gradient for {[n for n, ok in zip(names, nonzero, strict=True) if not ok]}"

    ebops = ebops_indices(module)
    assert ebops, f"{case}: no EBOPs-enabled leaf"
    assert n_losses > 0 and float(qloss) > 0, f"{case}: no EBOPs loss ({n_losses} terms, sum {float(qloss)})"
    zero = {name: [int(np.asarray(ntv[i])) for i in idx] for name, idx in ebops.items() if any(int(np.asarray(ntv[i])) == 0 for i in idx)}
    assert not zero, f"{case}: ebops == 0 after a training step for {zero}"


UINT32_MAX = 2**32 - 1


@pytest.mark.parametrize("case", ["dense_ffn", "innet", "attn_cross", "enc_lin"])
def test_ebops_loss_term(case):
    """The train-mode losses carry beta x EBOPs on top of the eval-mode (non-EBOPs) terms.

    Control: with training=False HGQ2 writes no ebops and the losses are only the other regulariser terms
    (measured 0.0033 for dense_ffn, 0.246 for enc_lin). Without softmax, train - eval equals beta x sum(ebops) of
    the returned state (rel 1e-5). With softmax, the exp-table ebops saturate the uint32 variable on JAX
    (plan 3.6), so the stored sum is only a lower bound; S5 reads EBOPs out as sum(losses) / beta instead.
    """
    module, _, golden = build_from_golden(case, MODE)
    inputs = inputs_of(golden)
    tv = [v.value for v in module.trainable_variables]
    ntv = [v.value for v in module.non_trainable_variables]
    (_, (ntv_train, q_train, n_train)), _ = jitted_step(case, module, training=True)(tv, ntv, inputs)
    (_, (ntv_eval, q_eval, n_eval)), _ = jitted_step(case, module, training=False)(tv, ntv, inputs)

    idx = [i for group in ebops_indices(module).values() for i in group]
    assert all(int(np.asarray(ntv_eval[i])) == 0 for i in idx), "eval mode wrote ebops"
    stored = [int(np.asarray(ntv_train[i])) for i in idx]
    assert all(v > 0 for v in stored)
    assert n_train > n_eval, (n_train, n_eval)
    betas = {float(np.asarray(v.value)) for v in module.non_trainable_variables if v.name == "beta"}
    assert len(betas) == 1, betas
    beta = betas.pop()
    penalty = float(q_train) - float(q_eval)
    if UINT32_MAX in stored:
        assert CASES[case]["kind"] in {"attn", "enc", "lin"}, "only softmax exp tables should saturate"
        assert penalty > beta * sum(stored)
    else:
        assert penalty == pytest.approx(beta * sum(stored), rel=1e-5)
