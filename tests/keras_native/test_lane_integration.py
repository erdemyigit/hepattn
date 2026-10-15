"""Merge check: lane A's native MaskFormer outputs feed lane B's matched loss inside one jitted step.

S5 builds the train step on this path; here it is checked only that the two lanes agree on the
output/target interface (``objective.py`` docstring): the jitted ``value_and_grad`` compiles once,
makes one host matching call per step, and the matched task loss is finite and reaches every model
weight. Quantizer variables are excluded from the gradient check: some of them get gradients only
from the HGQ2 penalty (that is checked in test_mf_jit.py).
"""

import jax
import numpy as np
import pytest

from hepattn.keras_native.matching import host_match
from hepattn.keras_native.objective import clic_specs, matched_loss

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .mf_golden_utils import build_mf


def real_like(targets: dict) -> dict:
    """Dummy data gives padded nodes target 1, so mask_bce is inf (S4.md); real CLIC data has 0 there."""
    targets = dict(targets)
    node_valid = targets["node_valid"][:, None, :]
    targets["particle_node_valid"] = targets["particle_node_valid"] & node_valid
    targets["particle_incidence"] = np.where(node_valid, targets["particle_incidence"], 0.0).astype(np.float32)
    return targets


@pytest.fixture(scope="module")
def batches():
    return [clic_dummy_batch(num_events=2, offset=2 * i) for i in range(3)]


def test_native_model_outputs_train_through_the_matched_loss(batches):
    model = build_mf("lin", QUANT_SPECS["high"], batches[0][0])
    specs = clic_specs()
    host_calls = []

    def counting_host_match(*args, **kwargs):
        host_calls.append(1)
        return host_match(*args, **kwargs)

    def task_loss(tv, ntv, inputs, targets):
        outputs, ntv, _ = model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)
        total, _, _ = matched_loss(outputs, targets, specs, host_fn=counting_host_match)
        return total, ntv

    step = jax.jit(jax.value_and_grad(task_loss, has_aux=True))
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    weights = [i for i, v in enumerate(model.trainable_variables) if "quantizer" not in v.path]
    assert len(weights) > 100, "the model under test has too few weights"

    reached = np.zeros(len(tv), dtype=bool)
    losses = []
    for inputs, targets in batches:
        (loss, ntv), grads = step(tv, ntv, inputs, real_like(targets))
        losses.append(float(loss))
        for i, g in enumerate(grads):
            g = np.asarray(g)
            assert np.isfinite(g).all(), f"non-finite gradient for {model.trainable_variables[i].path}"
            reached[i] |= np.any(g != 0)

    assert step._cache_size() == 1, f"{step._cache_size()} compilations over 3 batches"  # noqa: SLF001
    assert len(host_calls) == len(batches), f"{len(host_calls)} host matching calls in {len(batches)} steps"
    assert all(np.isfinite(loss) and loss > 0 for loss in losses), losses
    assert len(set(losses)) == len(losses), f"the 3 batches gave identical losses {losses}"
    missed = [model.trainable_variables[i].path for i in weights if not reached[i]]
    assert not missed, f"{len(missed)} model weights get no gradient from the matched loss: {missed[:5]}"


def test_dummy_targets_give_the_documented_inf(batches):
    """Control for ``real_like``: without it the dummy targets make mask_bce inf, as in torch (S4.md)."""
    inputs, targets = batches[0]
    model = build_mf("lin", None, inputs)
    total, _, _ = matched_loss(model(inputs, training=False), targets, clic_specs())
    assert np.isinf(float(total))
    total, _, _ = matched_loss(model(inputs, training=False), real_like(targets), clic_specs())
    assert np.isfinite(float(total))
