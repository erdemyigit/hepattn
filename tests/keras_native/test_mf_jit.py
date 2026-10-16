"""S3 JIT gates for the whole native MaskFormer (quantized, Linformer, CLIC test shapes).

- The jitted forward (``stateless_call``) compiles once over 3 different batches of the same static shape; in float it
  equals the eager forward.
- ``jax.jit(jax.value_and_grad(...stateless_call(..., return_losses=True)))`` compiles once over 3 batches; every
  trainable variable gets a nonzero gradient from a surrogate loss (sum of output terms + the HGQ2 losses).
- The HGQ2 regularisation terms come back from ``stateless_call(return_losses=True)``, identical inside and outside jit,
  with EBOPs > 0 in training mode only (their values against the hybrid: test_mf_golden.py).
"""

import jax
import keras
import numpy as np
import pytest

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .mf_golden_utils import build_mf, flatten

SENTINEL = 1e30

# Merge xfail (rt/integration, docs/hgq/port_log/R-merge.md); strict, so it fails once R5 changes mf_cases.py LIN_KWARGS.
MERGE_R5_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason=(
        "R5: golden/mf_cases.py LIN_KWARGS passes linformer_proj_dim, which native build_attention rejects since R2"
        " (RETARGET_ORG_MAIN.md section 5.1)"
        "; added at the rt/integration merge, see docs/hgq/port_log/R-merge.md"
    ),
)
pytestmark = MERGE_R5_XFAIL


@pytest.fixture(scope="module")
def batches():
    return [clic_dummy_batch(num_events=2, offset=2 * i)[0] for i in range(3)]


@pytest.fixture(scope="module")
def model(batches):
    return build_mf("lin", QUANT_SPECS["high"], batches[0])


def surrogate(outputs) -> jax.Array:
    """Mean square of every float output (finfo.min padding masked out), summed over layers and tasks."""
    total = 0.0
    for value in jax.tree_util.tree_leaves(outputs):
        if not keras.backend.is_float_dtype(value.dtype):
            continue
        value = keras.ops.where(keras.ops.abs(value) < SENTINEL, value, 0.0)
        total = total + keras.ops.mean(keras.ops.square(value))
    return total


def state(model):
    return [v.value for v in model.trainable_variables], [v.value for v in model.non_trainable_variables]


def test_forward_compiles_once(model, batches):
    tv, ntv = state(model)

    @jax.jit
    def forward(tv, ntv, inputs):
        out, _ = model.stateless_call(tv, ntv, inputs, training=False)
        return out

    outs = [flatten(forward(tv, ntv, b)) for b in batches]
    assert forward._cache_size() == 1, f"{forward._cache_size()} compilations over 3 batches"  # noqa: SLF001
    key = "final.mask.pflow_node_logit"
    assert not np.array_equal(outs[1][key], outs[0][key]), "the 3 batches are not different inputs"


def test_jitted_float_forward_equals_eager(batches):
    """Float: the jitted forward equals the eager one at the e2e tolerance, masks exactly.

    Quantized jitted forwards are checked against the hybrid golden files (test_mf_golden.py ``-jit`` rows).
    """
    model = build_mf("lin", None, batches[0])
    tv, ntv = state(model)
    jitted = flatten(jax.jit(lambda tv, ntv, x: model.stateless_call(tv, ntv, x, training=False)[0])(tv, ntv, batches[0]))
    eager = flatten(model(batches[0], training=False))
    assert set(jitted) == set(eager)
    for key, ref in eager.items():
        if ref.dtype == bool:
            assert np.array_equal(ref, jitted[key]), key
        else:
            assert np.all(np.abs(jitted[key].astype(np.float64) - ref) <= 5e-5 + 1e-4 * np.abs(ref)), key


def test_train_step_compiles_once_and_every_variable_gets_a_gradient(model, batches):
    tv, ntv = state(model)

    def loss_fn(tv, ntv, inputs):
        out, new_ntv, losses = model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)
        qloss = sum(losses)
        return surrogate(out) + qloss, (new_ntv, qloss, len(losses))

    step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    nonzero = np.zeros(len(tv), dtype=bool)
    for b in batches:
        (loss, (ntv, qloss, n_losses)), grads = step(tv, ntv, b)
        assert np.isfinite(float(loss))
        assert all(np.isfinite(np.asarray(g)).all() for g in grads)
        nonzero |= np.array([bool(np.any(np.asarray(g) != 0)) for g in grads])
    assert step._cache_size() == 1, f"{step._cache_size()} compilations over 3 batches"  # noqa: SLF001
    names = [v.path for v in model.trainable_variables]
    assert nonzero.all(), f"zero gradient for {[n for n, ok in zip(names, nonzero, strict=True) if not ok]}"
    assert n_losses > 0 and float(qloss) > 0


def ebops_leaves(layer) -> int:
    return sum(1 for sub in layer._flatten_layers(include_self=True) if hasattr(sub, "_ebops"))  # noqa: SLF001


def test_quant_losses_come_back_from_stateless_call(model, batches):
    """``stateless_call(return_losses=True)``: the same terms inside and outside jit, EBOPs > 0 in training only.

    Count semantics (Keras): a layer called several times in one forward keeps the terms of all its calls while it is
    nested, but a top-level call clears its earlier ones. ``MaskFormer.call`` calls the tasks a last time at top level
    ("final"), so stateless_call counts each layer's EBOPs once, as the hybrid does (test_mf_golden.py). An eager
    ``model(x, training=True)`` is one outer call and keeps them all: eager ``model.losses`` has
    sum((calls - 1) x EBOPs leaves) more terms (measured 421 vs 411). The training step must use stateless_call.
    """
    tv, ntv = state(model)
    _, ntv_train, losses_train = model.stateless_call(tv, ntv, batches[0], training=True, return_losses=True)
    _, _, losses_eval = model.stateless_call(tv, ntv, batches[0], training=False, return_losses=True)
    jitted = jax.jit(lambda tv, ntv, x: model.stateless_call(tv, ntv, x, training=True, return_losses=True)[2])(tv, ntv, batches[0])
    train = np.array([float(x) for x in losses_train])
    assert len(jitted) == train.size > 0
    np.testing.assert_allclose(np.array([float(x) for x in jitted]), train, rtol=1e-6)
    penalty = train.sum() - sum(float(x) for x in losses_eval)
    assert penalty > 0, "training-mode losses carry no EBOPs penalty"
    ebops_pos = [i for i, v in enumerate(model.non_trainable_variables) if v.name == "ebops"]
    assert ebops_pos and all(int(np.asarray(ntv_train[i])) > 0 for i in ebops_pos), "an EBOPs-enabled leaf wrote no ebops"

    model(batches[0], training=True)
    eager = len(model.losses)
    model(batches[0], training=False)
    calls = {t.name: 1 + sum(t.should_run_at_layer(i) for i in range(len(model.decoder.layers))) for t in model.tasks}
    repeated = sum((calls[t.name] - 1) * ebops_leaves(t) for t in model.tasks)
    assert repeated == 10, calls  # classification: 4 leaves x 1 extra call; mask: 3 leaves x 2 extra calls
    assert eager - train.size == repeated, (eager, train.size)
