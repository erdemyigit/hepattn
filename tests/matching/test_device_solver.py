"""Tests for the opt-in device-side LAP solver.

The acceptance criterion throughout is *equal total assignment cost* against scipy in float64,
not an identical permutation. Cost matrices with tied assignments are genuinely degenerate and
two exact solvers are free to disagree about which optimum they return.

These run on CPU so CI covers them; the ``gpu``-marked ones repeat the same checks on device to
catch anything that only shows up in the CUDA kernels.

``jv`` needs the compiled torch-linear-assignment extension, which is not a hard dependency of
this package, so the tests that solve anything skip where it is absent.
"""

import importlib.util

import numpy as np
import pytest
import scipy.optimize
import torch

from hepattn.models.device_lap import assignment_to_permutation, batched_jv
from hepattn.models.matcher import Matcher

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]

needs_jv = pytest.mark.skipif(
    importlib.util.find_spec("torch_linear_assignment") is None,
    reason="torch-linear-assignment is not installed; see setup/build_torch_linear_assignment.sh",
)


def optimal_cost(cost: np.ndarray) -> float:
    """Total cost of the optimal assignment, computed in float64."""
    row, col = scipy.optimize.linear_sum_assignment(cost)
    return float(cost[row, col].sum())


def achieved_cost(cost: np.ndarray, assignment: np.ndarray) -> float:
    """Total cost of an assignment, which must give every row a distinct column."""
    n_rows = cost.shape[0]
    assert len(set(assignment[:n_rows].tolist())) == n_rows, "assignment reuses a column"
    return float(cost[np.arange(n_rows), assignment[:n_rows]].sum())


@needs_jv
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("num_rows", "num_cols"), [(1, 1), (5, 5), (20, 20), (10, 40), (50, 150), (150, 150)])
def test_device_solver_matches_scipy(device: str, num_rows: int, num_cols: int):
    """The solver reaches the optimal cost on well-conditioned problems of every shape."""
    rng = np.random.default_rng(num_rows * 1000 + num_cols)
    batch = 8
    cost = rng.random((batch, num_rows, num_cols))
    row_valid = torch.ones(batch, num_rows, dtype=torch.bool, device=device)

    assigned = batched_jv(torch.as_tensor(cost, dtype=torch.float32, device=device), row_valid).cpu().numpy()
    for b in range(batch):
        assert achieved_cost(cost[b], assigned[b]) == pytest.approx(optimal_cost(cost[b]), abs=1e-6)


@needs_jv
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e6])
def test_device_solver_is_scale_invariant(device: str, scale: float):
    """Per-problem normalisation should make the solver indifferent to the cost scale."""
    rng = np.random.default_rng(7)
    cost = (rng.random((4, 30, 60)) - 0.5) * scale
    assigned = batched_jv(torch.as_tensor(cost, dtype=torch.float32, device=device), torch.ones(4, 30, dtype=torch.bool, device=device))

    assigned = assigned.cpu().numpy()
    for b in range(4):
        assert achieved_cost(cost[b], assigned[b]) == pytest.approx(optimal_cost(cost[b]), rel=1e-6, abs=1e-6 * abs(scale))


@needs_jv
@pytest.mark.parametrize("device", DEVICES)
def test_device_solver_handles_degenerate_costs(device: str):
    """Heavily tied costs must still produce a valid, optimal-cost assignment."""
    rng = np.random.default_rng(11)
    # Rounding to a handful of levels makes most entries exact ties.
    cost = np.round(rng.random((8, 25, 40)) * 3)
    assigned = batched_jv(torch.as_tensor(cost, dtype=torch.float32, device=device), torch.ones(8, 25, dtype=torch.bool, device=device))

    assigned = assigned.cpu().numpy()
    for b in range(8):
        assert achieved_cost(cost[b], assigned[b]) == pytest.approx(optimal_cost(cost[b]), abs=1e-6)


@needs_jv
@pytest.mark.parametrize("device", DEVICES)
def test_device_solver_respects_padding_and_forbidden_entries(device: str):
    """Padded rows stay unassigned, disallowed columns stay unused, and non-finite costs do not leak."""
    rng = np.random.default_rng(3)
    batch, num_rows, num_cols = 6, 12, 30
    cost = rng.random((batch, num_rows, num_cols))
    lengths = rng.integers(0, num_rows + 1, size=batch)
    row_valid = torch.arange(num_rows)[None, :] < torch.as_tensor(lengths)[:, None]

    col_allowed = torch.as_tensor(rng.random((batch, num_cols)) < 0.7)
    col_allowed[:, :num_rows] = True  # keep every problem feasible
    cost[:, :, 0] = np.nan  # a non-finite column must be treated as forbidden, not propagated

    assigned = (
        batched_jv(
            torch.as_tensor(cost, dtype=torch.float32, device=device),
            row_valid.to(device),
            col_allowed.to(device),
        )
        .cpu()
        .numpy()
    )

    for b in range(batch):
        n = int(lengths[b])
        assert (assigned[b, n:] == -1).all(), "a padded row was assigned"
        taken = assigned[b, :n]
        assert (taken >= 0).all()
        assert not np.isin(taken, 0).any(), "the non-finite column was assigned"
        assert col_allowed[b].numpy()[taken].all(), "a disallowed column was assigned"
        if n == 0:
            continue
        # The reference must see the same restricted problem: allowed columns only, and column
        # 0 is non-finite so it is forbidden too.
        usable = np.flatnonzero(col_allowed[b].numpy() & (np.arange(num_cols) != 0))
        assert achieved_cost(cost[b, :n], taken) == pytest.approx(optimal_cost(cost[b, :n][:, usable]), abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_device_solver_rejects_infeasible_problems(device: str):
    """More valid rows than assignable columns has no solution and must not be silently mangled."""
    with pytest.raises(ValueError, match="more valid rows than assignable columns"):
        batched_jv(
            torch.zeros(2, 5, 5, device=device),
            torch.ones(2, 5, dtype=torch.bool, device=device),
            torch.as_tensor([[True] * 3 + [False] * 2] * 2, device=device),
        )


@pytest.mark.parametrize("device", DEVICES)
def test_jv_rejects_more_rows_than_columns(device: str):
    """A batched solver assigns every row it is given, padding included, so it needs a column for each.

    Only three of the six rows are valid here, so the problem is feasible as far as the valid
    rows go; it is the padded rows that JV cannot leave on the table.
    """
    row_valid = torch.arange(6, device=device)[None, :] < 3
    with pytest.raises(ValueError, match="at least as many columns as rows"):
        batched_jv(torch.zeros(2, 6, 4, device=device), row_valid.expand(2, 6))


@pytest.mark.parametrize("device", DEVICES)
def test_assignment_to_permutation_is_a_permutation(device: str):
    """Unmatched columns fill the tail in ascending order, so the result always indexes cleanly."""
    assigned = torch.as_tensor([[2, 0, -1], [4, 1, 3]], device=device)
    n_valid = torch.as_tensor([2, 3], device=device)
    perm = assignment_to_permutation(assigned, n_valid, num_cols=5).cpu()

    torch.testing.assert_close(perm[0], torch.tensor([2, 0, 1, 3, 4]))
    torch.testing.assert_close(perm[1], torch.tensor([4, 1, 3, 0, 2]))


@needs_jv
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_preds", [16, 64])
def test_matcher_device_solver_agrees_with_host(device: str, num_preds: int):
    """The end-to-end Matcher must pick assignments of identical cost with and without the option."""
    rng = np.random.default_rng(num_preds)
    host = Matcher(default_solver="scipy", adaptive_solver=False)
    dev = Matcher(default_solver="scipy", adaptive_solver=False, device_solver="jv")

    for _ in range(10):
        batch = int(rng.integers(1, 6))
        costs = torch.as_tensor(rng.standard_normal((batch, num_preds, num_preds)), dtype=torch.float32)
        lengths = rng.integers(0, num_preds + 1, size=batch)
        object_valid = torch.arange(num_preds)[None, :] < torch.as_tensor(lengths)[:, None]

        host_idx = host(costs, object_valid).cpu().numpy()
        dev_idx = dev(costs.to(device), object_valid.to(device)).cpu().numpy()

        for b in range(batch):
            assert sorted(dev_idx[b].tolist()) == list(range(num_preds)), "not a permutation"
            n = int(lengths[b])
            if n == 0:
                continue
            cost_b = costs[b].double().numpy().T[:n]  # solver layout is [target, pred]
            assert achieved_cost(cost_b, dev_idx[b]) == pytest.approx(achieved_cost(cost_b, host_idx[b]), abs=1e-6)


def test_matcher_rejects_unknown_device_solver():
    with pytest.raises(ValueError, match="Unknown device solver"):
        Matcher(device_solver="not-a-solver")


def test_matcher_device_solver_is_off_by_default():
    """The production default must stay on the host path, so the option cannot regress anyone."""
    assert Matcher().device_solver is None
