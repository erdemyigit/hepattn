"""Batched linear assignment solved on whichever device the costs already live on.

A CLIC training step hands the matcher ~10k independent assignment problems at once (batch x
the decoder layers plus the final head), each at most a few hundred rows square. The host
solvers in :mod:`hepattn.models.matcher` need that whole stack copied device->host first, and
CPU-side parallelism over it saturates at ~16 threads. This module solves the same problems in
place on the GPU, so the copy and the host stall disappear. It is opt-in, through
``Matcher(device_solver="jv")``: on a GPU that is already saturated the solver's own kernels
cost more than the stall they remove, so it only pays when training is host-bound.

The solver is the batched Jonker-Volgenant of ``torch-linear-assignment`` (Crouse 2016, the
algorithm scipy itself uses), which is exact and strongly polynomial: its cost depends on the
shape of a problem and not on the values in it. An auction solver was tried first and dropped,
because the number of bidding rounds grows with the cost range and real matcher costs drove it
to non-convergence.

Costs are affine-normalised per problem before solving. This is exact -- every permutation
sums exactly ``n_rows`` entries, so ``(c - min) / range`` is a monotone map on total cost --
and it keeps the fp32 duals the solver works with well conditioned: the raw sentinel that
:class:`~hepattn.models.matcher.Matcher` uses for forbidden assignments on the host path
(``float32_max / 10``) would otherwise destroy them.
"""

import torch
from torch import Tensor

__all__ = ["assignment_to_permutation", "batched_jv", "require_jv"]


def _normalise(costs: Tensor, allowed: Tensor, forbidden_cost: float) -> Tensor:
    """Map each problem's allowed costs onto [0, 1] and its forbidden ones onto a sentinel.

    Args:
        costs: [batch, num_rows, num_cols] cost matrices.
        allowed: [batch, num_rows, num_cols] bool, False where a row may not take a column.
        forbidden_cost: Value to write into the disallowed entries. Must exceed the largest
            achievable total cost of a feasible assignment so that a forbidden pairing is
            never preferred to a feasible one.

    Returns:
        The normalised costs, same shape and dtype as ``costs``.
    """
    inf = torch.inf
    hi = torch.where(allowed, costs, torch.full_like(costs, -inf)).amax(dim=(1, 2))
    lo = torch.where(allowed, costs, torch.full_like(costs, inf)).amin(dim=(1, 2))

    # A problem with no allowed entry at all (e.g. no valid targets) leaves hi/lo infinite;
    # it has nothing to solve, so any finite affine map will do.
    degenerate = ~torch.isfinite(hi) | ~torch.isfinite(lo)
    lo = torch.where(degenerate, torch.zeros_like(lo), lo)
    scale = torch.where(degenerate, torch.ones_like(hi), hi - lo).clamp_min(torch.finfo(costs.dtype).tiny)

    normalised = (costs - lo[:, None, None]) / scale[:, None, None]
    return torch.where(allowed, normalised, torch.full_like(normalised, forbidden_cost))


def require_jv(device: torch.device | None = None):
    """Import the batched Jonker-Volgenant backend, or explain how to get one.

    Deliberately loud, and called at ``Matcher`` construction rather than mid-training. A
    missing build must not quietly leave the caller on the host solver, and a CPU-only build
    must not quietly solve device costs on the host -- ``batch_linear_assignment`` merely warns
    in that case, and a build without ``FORCE_CUDA`` is exactly what a login node produces.

    Args:
        device: If given and on CUDA, also require that the extension was built with CUDA.

    Returns:
        The ``batch_linear_assignment`` entry point.

    Raises:
        RuntimeError: If the package is not importable, or was built without CUDA support and
            the costs are on a CUDA device.
    """
    try:
        # The private backend, for has_cuda(): a build-time property with no public accessor.
        import torch_linear_assignment._backend as backend  # noqa: PLC0415, PLC2701  # ty: ignore[unresolved-import]
        from torch_linear_assignment import batch_linear_assignment  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise RuntimeError(
            "The 'jv' device solver needs torch-linear-assignment, which is not importable. "
            "Build it with setup/build_torch_linear_assignment.sh: it must be compiled with "
            "FORCE_CUDA=1 (setup.py gates on torch.cuda.is_available(), so a build on a login "
            "node silently produces a CPU-only extension) and TORCH_CUDA_ARCH_LIST set for the "
            "target GPU, then put on PYTHONPATH with LD_LIBRARY_PATH pointing at this "
            f"environment's lib. Original error: {exc}"
        ) from exc

    if device is not None and torch.device(device).type == "cuda" and not backend.has_cuda():
        raise RuntimeError(
            "torch-linear-assignment was built without CUDA support, so the 'jv' device solver "
            "would solve GPU costs on the host and quietly undo the point of the option. "
            "Rebuild with FORCE_CUDA=1 and TORCH_CUDA_ARCH_LIST set for this GPU."
        )
    return batch_linear_assignment


@torch.no_grad()
def batched_jv(costs: Tensor, row_valid: Tensor, col_allowed: Tensor | None = None) -> Tensor:
    """Solve a batch of rectangular linear assignment problems exactly, by Jonker-Volgenant.

    Every valid row is assigned a distinct column, minimising the total cost. There must be at
    least as many columns as rows in each problem, which the matcher guarantees because it only
    ever matches targets (rows) to a larger pool of queries (columns).

    Two preparation decisions carry the correctness of this path, and neither is obvious from
    the host solvers:

    * **Forbidden entries** go to ``num_rows + 1`` after the per-problem affine normalisation,
      and *not* to the ``float32_max / 10`` sentinel
      :meth:`~hepattn.models.matcher.Matcher._prepare_costs` hands the host solvers. A
      feasible assignment costs at most ``num_rows`` in normalised units, so ``num_rows + 1``
      is enough to make a forbidden pairing lose to every feasible one, and it keeps the fp32
      duals (``cost - u - v``) well conditioned where a 3.4e37 entry would destroy them.
    * **Padded rows** get a constant cost across every column. A batched solver assigns every
      row it is given, so padded rows cannot abstain. Constant rows contribute the same total
      whichever columns they take, so the solver parks them wherever suits the real rows and
      the optimum over the real rows is unchanged. This is the standard dummy-row reduction and
      it is exact -- padding with a large sentinel would not be, since the padded rows would
      then compete for the cheap columns.

    Args:
        costs: [batch, num_rows, num_cols] cost matrices. Non-finite entries are treated as
            forbidden assignments rather than propagating NaN.
        row_valid: [batch, num_rows] bool marking the rows that need an assignment. Padded rows
            come back as -1.
        col_allowed: Optional [batch, num_cols] bool marking the columns that may be assigned.
            Columns that are False are left for the unmatched remainder.

    Returns:
        [batch, num_rows] column assigned to each valid row, -1 for padded rows.

    Raises:
        ValueError: If the costs are not 3-dimensional, if a problem has more rows than
            columns, or if one has more valid rows than assignable columns.
    """
    if costs.ndim != 3:
        raise ValueError(f"Expected costs of shape [batch, num_rows, num_cols], got {tuple(costs.shape)}")

    batch, num_rows, num_cols = costs.shape
    device = costs.device
    dtype = costs.dtype if costs.dtype.is_floating_point else torch.float32
    costs = costs.to(dtype)

    row_valid = row_valid.to(device=device, dtype=torch.bool)
    if col_allowed is None:
        col_allowed = torch.ones(batch, num_cols, dtype=torch.bool, device=device)
    else:
        col_allowed = col_allowed.to(device=device, dtype=torch.bool)

    if bool((row_valid.sum(dim=1) > col_allowed.sum(dim=1)).any()):
        raise ValueError("Some assignment problems have more valid rows than assignable columns")
    # Every row is handed to the solver, padding included, so there must be a column for each.
    if num_rows > num_cols:
        raise ValueError(f"Batched JV needs at least as many columns as rows, got {num_rows} rows into {num_cols} columns")

    unassigned = torch.full((batch, num_rows), -1, dtype=torch.long, device=device)
    if batch == 0 or num_rows == 0 or num_cols == 0:
        return unassigned

    solve = require_jv(device)
    allowed = torch.isfinite(costs) & row_valid[:, :, None] & col_allowed[:, None, :]
    prepared = _normalise(costs, allowed, forbidden_cost=float(num_rows + 1))
    prepared = torch.where(row_valid[:, :, None], prepared, torch.zeros_like(prepared))

    matching = solve(prepared.contiguous()).to(device=device, dtype=torch.long)
    return torch.where(row_valid, matching, unassigned)


@torch.no_grad()
def assignment_to_permutation(assigned: Tensor, n_valid_rows: Tensor, num_cols: int) -> Tensor:
    """Expand a partial row->column assignment into the full column permutation the matcher wants.

    Position ``i`` of the result holds the column matched to row ``i``, for the valid rows; the
    remaining positions are filled with the unmatched columns in ascending order, so the result
    is always a permutation of ``range(num_cols)`` and can index the prediction tensors directly.

    Args:
        assigned: [batch, num_rows] column assigned to each row, -1 where unassigned.
        n_valid_rows: [batch] number of valid rows per problem, i.e. how many leading positions
            of the output are real matches.
        num_cols: Width of the permutation to produce.

    Returns:
        [batch, num_cols] int64 permutation.
    """
    batch, num_rows = assigned.shape
    device = assigned.device
    matched = assigned >= 0

    # Scatter through a trailing scratch column so that unassigned rows cannot race with real
    # writes: every valid row holds a distinct column, and everything else is aimed at `num_cols`.
    trash_col = torch.full_like(assigned, num_cols)
    used = torch.zeros(batch, num_cols + 1, dtype=torch.bool, device=device)
    used.scatter_(1, torch.where(matched, assigned, trash_col), torch.ones_like(assigned, dtype=torch.bool))

    # Unmatched columns keep their natural order and start where the matched rows leave off.
    free = ~used[:, :num_cols]
    slot = n_valid_rows[:, None] + free.long().cumsum(1) - 1
    col_idx = torch.arange(num_cols, device=device).expand(batch, num_cols)
    trash_slot = torch.full_like(slot, num_cols)

    perm = torch.zeros(batch, num_cols + 1, dtype=torch.long, device=device)
    perm.scatter_(1, torch.where(free, slot.clamp(0, num_cols), trash_slot), col_idx)
    row_idx = torch.arange(num_rows, device=device).expand(batch, num_rows)
    perm.scatter_(1, torch.where(matched, row_idx, torch.full_like(row_idx, num_cols)), assigned.clamp_min(0))
    return perm[:, :num_cols]
