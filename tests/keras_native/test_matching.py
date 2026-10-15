"""S4: host Hungarian matching through jax.pure_callback, the permutation, and gradients of the matched loss."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from hepattn.keras_native import matching
from hepattn.keras_native import objective as O
from hepattn.models.maskformer import MaskFormer
from hepattn.models.matcher import Matcher

from .objective_utils import LAYER_TASKS, P, random_outputs, random_targets, to_torch, torch_tasks


def _matched_total(costs: np.ndarray, valid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per problem: sum over valid targets t of cost[idx[t], t] (idx[t] = prediction placed at target t)."""
    totals = []
    for c, v, i in zip(costs, valid, idx, strict=True):
        n = int(v.sum())
        totals.append(float(np.sum(c[i[:n], np.arange(n)], dtype=np.float64)))
    return np.array(totals)


# The finite sentinel of torch Matcher._prepare_costs (matcher.py), written out independently of matching._BIG.
BIG = float(np.finfo(np.float32).max / 10)


def _random_problem(rng, batch, num=P, nonfinite=False):
    """Random costs; with ``nonfinite``, 2 NaN, 2 -inf and 2 +inf entries in the valid target columns of each event."""
    costs = rng.normal(size=(batch, num, num)).astype(np.float32)
    lengths = rng.integers(0, num + 1, size=batch)
    lengths[rng.random(batch) < 0.15] = num  # some events with every target valid
    valid = np.arange(num)[None] < lengths[:, None]
    if nonfinite:
        for b in np.flatnonzero(lengths > 0):
            pred = rng.choice(num, size=6, replace=False)
            costs[b, pred, rng.integers(0, lengths[b], size=6)] = [np.nan, np.nan, -np.inf, -np.inf, np.inf, np.inf]
    return costs, valid


def _sanitised(costs):
    return np.nan_to_num(costs, nan=BIG, posinf=BIG, neginf=-BIG)


@pytest.mark.parametrize("nonfinite", [False, True], ids=["finite", "nan-and-inf"])
def test_matched_total_cost_equals_torch_matcher_over_100_batches(nonfinite):
    """Matched totals equal torch ``Matcher`` (scipy) to rel 1e-6. With NaN/inf costs both sanitise to
    +-f32max/10 first; the totals are then taken on the sanitised costs and the indices must be identical.
    """
    rng = np.random.default_rng(0)
    jit_match = jax.jit(matching.match)
    torch_matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=True, n_jobs=4)
    worst_rel, identity_worse = 0.0, 0
    for _ in range(100):
        costs, valid = _random_problem(rng, int(rng.integers(1, 9)), nonfinite=nonfinite)
        idx = np.asarray(jit_match(costs, valid))
        ref = torch_matcher(torch.from_numpy(costs), torch.from_numpy(valid)).numpy()
        if nonfinite:
            np.testing.assert_array_equal(idx, ref)
            costs = _sanitised(costs)
        total, total_ref = _matched_total(costs, valid, idx), _matched_total(costs, valid, ref)
        rel = np.abs(total - total_ref) / np.maximum(np.abs(total_ref), 1e-12)
        worst_rel = max(worst_rel, float(rel.max()))
        # every row is a permutation of the predictions
        assert (np.sort(idx, axis=1) == np.arange(P)).all()
        # control: an identity assignment is worse wherever there is something to match
        identity = _matched_total(costs, valid, np.broadcast_to(np.arange(P), idx.shape))
        has_targets = valid.sum(1) > 1
        identity_worse += int((identity[has_targets] > total[has_targets] + 1e-3).all())
    print(f"matched total cost: max rel diff to torch Matcher {worst_rel:.2e}; identity worse in {identity_worse}/100 batches")
    assert worst_rel <= 1e-6
    # with +-f32max/10 entries the totals cannot resolve O(1) differences; the identical indices are the check there
    if not nonfinite:
        assert identity_worse == 100


def test_non_finite_costs_need_the_sanitising_step():
    """Control for the nan-and-inf row: scipy rejects NaN and -inf (the torch Matcher replaces them first);
    -inf entries become -f32max/10 and are matched, NaN and +inf become +f32max/10 and are avoided.
    """
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
    for bad in (np.nan, -np.inf):
        raw = cost.copy()
        raw[0, 1] = bad
        with pytest.raises(ValueError, match="invalid numeric entries"):
            linear_sum_assignment(raw)
    plus_inf = cost.copy()
    plus_inf[0, 0] = np.inf
    linear_sum_assignment(plus_inf)  # accepted by scipy; sanitised anyway, as in torch

    costs = np.zeros((1, 2, 2), np.float32)
    costs[0, 0, 1] = -np.inf  # prediction 0 -> target 1 is attractive
    costs[0, 1, 1] = np.nan  # prediction 1 -> target 1 is forbidden
    valid = np.ones((1, 2), bool)
    ref = Matcher(default_solver="scipy", adaptive_solver=False)(torch.from_numpy(costs), torch.from_numpy(valid)).numpy()
    np.testing.assert_array_equal(ref, [[1, 0]])  # target 0 <- prediction 1, target 1 <- prediction 0
    np.testing.assert_array_equal(matching.host_match(costs, valid), ref)


def test_unmatched_default_indices_equal_torch_matcher():
    """Same scipy call on the same array: the full index rows (matched + ascending unmatched tail) are identical."""
    rng = np.random.default_rng(1)
    costs, valid = _random_problem(rng, 16)
    valid[0] = False  # no target: pure default tail 0..P-1
    idx = np.asarray(jax.jit(matching.match)(costs, valid))
    ref = Matcher(default_solver="scipy", adaptive_solver=False)(torch.from_numpy(costs), torch.from_numpy(valid)).numpy()
    np.testing.assert_array_equal(idx, ref)
    np.testing.assert_array_equal(idx[0], np.arange(P))
    for row, v in zip(idx, valid, strict=True):
        tail = row[int(v.sum()) :]
        assert (np.diff(tail) > 0).all(), "unmatched predictions must follow in ascending order"


def test_host_match_threads_equal_sequential():
    rng = np.random.default_rng(2)
    costs, valid = _random_problem(rng, 20)
    np.testing.assert_array_equal(matching.host_match(costs, valid, n_jobs=8), matching.host_match(costs, valid, n_jobs=1))


@pytest.mark.parametrize("shape", [(3, P), (3, P, 6), (3, P, 160)])
def test_permute_equals_torch_advanced_indexing(shape):
    rng = np.random.default_rng(3)
    x = rng.normal(size=shape).astype(np.float32)
    idx = np.stack([rng.permutation(P) for _ in range(shape[0])]).astype(np.int32)
    ref = torch.from_numpy(x)[torch.arange(shape[0]).unsqueeze(1), torch.from_numpy(idx).long()].numpy()
    got = np.asarray(jax.jit(matching.permute)(x, idx))
    np.testing.assert_array_equal(got, ref)
    assert not np.array_equal(got, x)


def test_permute_outputs_equals_torch_match_and_permute():
    """Dict level: the same index per layer gives the torch ``_match_and_permute_outputs`` result bit for bit."""
    rng = np.random.default_rng(4)
    targets = random_targets(rng, 3)
    outputs = random_outputs(rng, targets["node_valid"])
    idx = {layer: np.stack([rng.permutation(P) for _ in range(3)]).astype(np.int32) for layer in outputs}

    layers = list(outputs)
    stacked = torch.from_numpy(np.concatenate([idx[layer] for layer in layers]))
    stub = SimpleNamespace(tasks=torch_tasks(), target_object="particle", matcher=lambda *_args: stacked)
    out_t = to_torch(outputs)
    zero_costs = {layer: torch.zeros(3, P, P) for layer in layers}
    MaskFormer._match_and_permute_outputs(stub, out_t, zero_costs, to_torch(targets))  # noqa: SLF001

    got = O.permute_outputs(outputs, idx, O.clic_specs())
    for layer in layers:
        for name in LAYER_TASKS[layer]:
            for key, value in out_t[layer][name].items():
                np.testing.assert_array_equal(np.asarray(got[layer][name][key]), value.numpy(), err_msg=f"{layer}.{name}.{key}")


# ---------------------------------------------------------------------------------------- jitted step

D = 16


def _toy_problem(seed=5, batch=3):
    rng = np.random.default_rng(seed)
    targets = random_targets(rng, batch, padded_node_targets=False)
    feats = {"q": rng.normal(size=(batch, P, D)).astype(np.float32), "n": rng.normal(size=(batch, 160, D)).astype(np.float32)}

    def init(k):
        return {
            "cls": (0.3 * rng.normal(size=(D, 6))).astype(np.float32) * (1 + k),
            "mask": (0.3 * rng.normal(size=(D, D))).astype(np.float32),
            "inc": (0.3 * rng.normal(size=(D, D))).astype(np.float32),
            "regr": (0.3 * rng.normal(size=(D, 5))).astype(np.float32),
        }

    params = {layer: jax.tree.map(jnp.asarray, init(k)) for k, layer in enumerate(LAYER_TASKS)}
    return params, feats, targets


def _toy_outputs(params, feats, targets):
    node_valid = targets["node_valid"]
    outputs = {}
    for layer, task_names in LAYER_TASKS.items():
        w = params[layer]
        out = {}
        if "classification" in task_names:
            logits = feats["q"] @ w["cls"]
            out["classification"] = {"pflow_logit": logits, "pflow_class_prob": jax.nn.softmax(logits, -1)}
        if "mask" in task_names:
            logit = jnp.einsum("bpd,bnd->bpn", feats["q"] @ w["mask"], feats["n"])
            out["mask"] = {"pflow_node_logit": jnp.where(node_valid[:, None, :], logit, jnp.finfo(jnp.float32).min)}
        if "incidence" in task_names:
            inc = jnp.einsum("bpd,bnd->bpn", feats["q"] @ w["inc"], feats["n"])
            out["incidence"] = {"pflow_incidence": jax.nn.softmax(inc, axis=1) * node_valid[:, None, :]}
        if "regression" in task_names:
            out["regression"] = {"pflow_regr": feats["q"] @ w["regr"], "pflow_proxy_regr": jnp.zeros((node_valid.shape[0], P, 5))}
        outputs[layer] = out
    return outputs


def _fixed_index_loss(params, feats, targets, idx):
    outputs = _toy_outputs(params, feats, targets)
    return O.total_loss(O.model_losses(O.permute_outputs(outputs, idx, O.clic_specs()), targets, O.clic_specs()))


def _assert_finite_tree(tree, what):
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        assert np.isfinite(np.asarray(leaf)).all(), f"{what}: non-finite gradient at {jax.tree_util.keystr(path)}"


def _max_tree_diff(a, b):
    """Max abs difference over all leaves; NaN-safe (Python max/min would silently drop a NaN)."""
    diff = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
    assert not np.isnan(diff), "NaN in a gradient difference"
    return diff


def test_callback_step_gradients_equal_fixed_index_reference():
    """jit(value_and_grad) with the matching inside equals the gradient at the matched indices held fixed.

    One host call per step (all layers stacked) and one compiled executable across steps. Control: the
    identity matching gives a different gradient, so the comparison discriminates.
    """
    params, feats, targets = _toy_problem()
    specs = O.clic_specs()
    calls = []

    def counting_host_fn(costs, valid):
        calls.append(costs.shape)
        return matching.host_match(costs, valid)

    def loss_fn(p):
        total, _, idx = O.matched_loss(_toy_outputs(p, feats, targets), targets, specs, host_fn=counting_host_fn)
        return total, idx

    step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    reference = jax.jit(jax.value_and_grad(_fixed_index_loss))
    worst_grad, worst_loss, min_control = 0.0, 0.0, np.inf
    n_steps = 5
    for _ in range(n_steps):
        (loss, idx), grads = step(params)
        idx = {k: np.asarray(v) for k, v in idx.items()}
        # the host matching of the eager costs gives the same indices
        eager_costs = O.model_costs(_toy_outputs(params, feats, targets), targets, specs)
        for layer, c in eager_costs.items():
            np.testing.assert_array_equal(idx[layer], matching.host_match(np.asarray(c), targets["particle_valid"]))
        ref_loss, ref_grads = reference(params, feats, targets, idx)
        _assert_finite_tree(grads, "callback step")
        _assert_finite_tree(ref_grads, "fixed-index reference")
        worst_grad = max(worst_grad, _max_tree_diff(grads, ref_grads))
        worst_loss = max(worst_loss, abs(float(loss) - float(ref_loss)))
        identity = {k: np.broadcast_to(np.arange(P, dtype=np.int32), v.shape) for k, v in idx.items()}
        ctrl_loss, ctrl_grads = reference(params, feats, targets, identity)
        _assert_finite_tree(ctrl_grads, "identity control")
        min_control = min(min_control, _max_tree_diff(grads, ctrl_grads))
        assert abs(float(ctrl_loss) - float(loss)) > 1e-2
        params = jax.tree.map(lambda w, g: w - 0.5 * g, params, grads)
    print(f"callback vs fixed-index: max |dloss| {worst_loss:.2e}, max |dgrad| {worst_grad:.2e}; identity control min |dgrad| {min_control:.2e}")
    assert worst_grad <= 1e-6
    assert worst_loss <= 1e-5
    assert min_control > 1e-3
    assert len(calls) == n_steps, f"expected one host call per step, got {len(calls)} in {n_steps} steps"
    assert all(shape == (len(LAYER_TASKS) * 3, P, P) for shape in calls), calls
    assert step._cache_size() == 1  # noqa: SLF001


def _squared_residual_loss(pred, target, object_valid_mask):
    """Control loss: like regr_l1_loss but squared, so d/dpred = 2 * residual * g (NaN * 0 = NaN when masked)."""
    diff = keras.ops.square(pred - target)
    mask = keras.ops.broadcast_to(keras.ops.expand_dims(object_valid_mask, -1), diff.shape)
    return keras.ops.sum(keras.ops.where(mask, diff, 0.0)) / keras.ops.maximum(keras.ops.sum(keras.ops.cast(mask, "float32")), 1.0)


@pytest.mark.parametrize("residual", ["l1", "squared"])
@pytest.mark.parametrize("sanitize", [True, False], ids=["nan_to_num", "without"])
def test_nan_regression_targets_and_gradients(monkeypatch, sanitize, residual):
    """NaN-padded regression targets through the full matched step.

    With nan_to_num every gradient is finite (the gate). Without it, the value stays finite (``where``), and:
    - for the production l1 loss the gradient is still finite, because JAX's abs JVP is a select
      (``jax/_src/lax/lax.py`` ``_abs_jvp_rule``), not ``sign(x) * g``;
    - for a loss whose derivative multiplies by the residual (control) the gradient is NaN, so the test
      detects a missing nan_to_num.
    """
    if not sanitize:
        monkeypatch.setattr(O, "_nan_to_zero", lambda x: x)
    if residual == "squared":
        monkeypatch.setattr(O.L, "regr_l1_loss", _squared_residual_loss)
    params, feats, targets = _toy_problem(seed=6)
    assert np.isnan(targets["particle_e"]).any()

    def loss_fn(p):
        return O.matched_loss(_toy_outputs(p, feats, targets), targets, O.clic_specs())[0]

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params)
    assert np.isfinite(float(loss))
    all_finite = all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grads))
    regr_nan = bool(np.isnan(np.asarray(grads["final"]["regr"])).any())
    if sanitize or residual == "l1":
        assert all_finite
    else:
        assert regr_nan, "control: a residual-weighted loss without nan_to_num must give NaN gradients"


def test_matched_loss_compiles_once_over_three_batches():
    specs = O.clic_specs()
    fn = jax.jit(lambda out, tgt: O.matched_loss(out, tgt, specs)[0])
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        targets = random_targets(rng, 2, padded_node_targets=False)
        targets = {k: (v.astype(np.int32) if v.dtype == np.int64 else v) for k, v in targets.items()}
        assert np.isfinite(float(fn(random_outputs(rng, targets["node_valid"]), targets)))
    assert fn._cache_size() == 1  # noqa: SLF001
