"""Merge check: the native MaskFormer outputs feed the matched loss inside one jitted step.

The model is org/main ``base.yaml`` + ``linformer.yaml`` at the test width (``mf_cases`` variant ``org``; plan 5.8).
S5 builds the train step on this path; here it is checked only that model and objective agree on the
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
    """Real CLIC data has no target at padded nodes; dummy data gives them target 1 (S4.md)."""
    targets = dict(targets)
    node_valid = targets["node_valid"][:, None, :]
    targets["particle_node_valid"] = targets["particle_node_valid"] & node_valid
    targets["particle_incidence"] = np.where(node_valid, targets["particle_incidence"], 0.0).astype(np.float32)
    return targets


@pytest.fixture(scope="module")
def batches():
    return [clic_dummy_batch(num_events=2, offset=2 * i) for i in range(3)]


def test_native_model_outputs_train_through_the_matched_loss(batches):
    model = build_mf("org", QUANT_SPECS["high"], batches[0][0])
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


def test_dummy_targets_give_a_finite_loss(batches):
    """Since org a4263c1 (bug 1, R1), target 1 at padded nodes no longer makes mask_bce inf (it was inf before, S4.md);
    ``real_like`` still changes the loss (the dice target sum and the incidence targets are not masked).
    Replacement from docs/hgq/port_log/R1.md, deviation 1.
    """
    inputs, targets = batches[0]
    model = build_mf("org", None, inputs)
    dummy, _, _ = matched_loss(model(inputs, training=False), targets, clic_specs())
    real, _, _ = matched_loss(model(inputs, training=False), real_like(targets), clic_specs())
    assert np.isfinite(float(dummy))
    assert np.isfinite(float(real))
    assert float(dummy) != float(real)
