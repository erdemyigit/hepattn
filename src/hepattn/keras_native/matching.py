"""Hungarian matching on the host inside a jitted JAX step, and the output permutation.

``host_match`` is the numpy/scipy logic of ``hepattn.models.matcher.Matcher`` (scipy solver, no
``query_valid_mask``): per problem, keep the first ``n_valid`` target rows of the transposed cost, solve
with ``scipy.optimize.linear_sum_assignment``, then append the unmatched prediction indices in ascending
order (``matcher.py:96-104``). Valid targets must come first in each event, as in torch.

``match`` calls it through ``jax.pure_callback`` on ``stop_gradient(costs)`` (plan section 3.4, option A),
so one compiled step solves every layer and event with one host call. JAX-backend only.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import numpy as np
from keras import ops
from scipy.optimize import linear_sum_assignment

_POOLS: dict[int, ThreadPoolExecutor] = {}
_POOLS_LOCK = threading.Lock()


def _pool(n_jobs: int) -> ThreadPoolExecutor:
    with _POOLS_LOCK:
        if n_jobs not in _POOLS:
            _POOLS[n_jobs] = ThreadPoolExecutor(max_workers=n_jobs, thread_name_prefix="hepattn-lsap")
        return _POOLS[n_jobs]


def match_one(cost_t: np.ndarray, n_valid: int) -> np.ndarray:
    """One problem. ``cost_t`` is [T, P] (targets x predictions); returns [P] int32 prediction indices.

    Entry ``i < n_valid`` is the prediction matched to target ``i``; the rest are the unmatched
    predictions in ascending order (``matcher.match_individual``).
    """
    num_pred = cost_t.shape[1]
    _, pred_idx = linear_sum_assignment(cost_t[:n_valid])
    remaining = np.ones(num_pred, dtype=np.bool_)
    remaining[pred_idx] = False
    return np.concatenate([pred_idx.astype(np.int32), np.arange(num_pred, dtype=np.int32)[remaining]])


def host_match(costs: np.ndarray, target_valid: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """Match a stack of problems. costs [M, P, T] (prediction x target), target_valid [M, T] -> [M, P] int32."""
    costs = np.asarray(costs, dtype=np.float32)
    lengths = np.asarray(target_valid, dtype=np.bool_).sum(axis=1)
    costs_t = costs.swapaxes(1, 2)
    if n_jobs <= 1 or len(costs_t) <= 1:
        results = [match_one(costs_t[i], int(lengths[i])) for i in range(len(costs_t))]
    else:
        results = list(_pool(n_jobs).map(lambda i: match_one(costs_t[i], int(lengths[i])), range(len(costs_t))))
    idx = np.stack(results, axis=0).astype(np.int32)
    assert (idx >= 0).all(), "Matcher error!"
    return idx


def match(costs, target_valid, *, n_jobs: int = 1, host_fn=None):
    """``host_match`` (or ``host_fn``) through ``jax.pure_callback``. costs [M, P, T], target_valid [M, T] -> [M, P] int32.

    Gradients do not flow into the costs (``stop_gradient``), so the callback needs no derivative.
    """
    fn = partial(host_match, n_jobs=n_jobs) if host_fn is None else host_fn
    costs = jax.lax.stop_gradient(ops.cast(costs, "float32"))
    target_valid = jax.lax.stop_gradient(ops.cast(target_valid, "bool"))
    result = jax.ShapeDtypeStruct(costs.shape[:2], np.int32)
    return jax.pure_callback(fn, result, costs, target_valid, vmap_method="sequential")


def permute(x, idx):
    """``x[arange(B)[:, None], idx]`` for x [B, P, ...] and idx [B, P], via ``take_along_axis`` on axis 1."""
    idx = ops.cast(idx, "int32")
    for _ in range(len(x.shape) - 2):
        idx = ops.expand_dims(idx, -1)
    return ops.take_along_axis(x, idx, axis=1)
