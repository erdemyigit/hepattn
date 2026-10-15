import atexit
import contextlib
import time
import warnings
from multiprocessing import get_context, shared_memory
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Literal

import numpy as np
import scipy
import torch
from torch import nn

from hepattn.models.device_lap import assignment_to_permutation, batched_jv, require_jv
from hepattn.utils.import_utils import check_import_safe

# Solvers that run on whichever device the costs are already on, rather than on the host.
# Opt-in via Matcher(device_solver=...); see the module docstring of hepattn.models.device_lap.
DEVICE_SOLVERS = {
    "jv": batched_jv,
}

# Device solvers that need a compiled dependency check for it here, so a missing or CPU-only
# build fails at construction with a build recipe rather than mid-training, or worse, quietly.
DEVICE_SOLVER_CHECKS = {
    "jv": require_jv,
}

_POOL_LOCK = Lock()
_THREAD_POOLS: dict[int, ThreadPool] = {}
_PROCESS_POOLS = {}


def _get_thread_pool(n_jobs: int) -> ThreadPool:
    with _POOL_LOCK:
        pool = _THREAD_POOLS.get(n_jobs)
        if pool is None:
            pool = ThreadPool(processes=n_jobs)
            _THREAD_POOLS[n_jobs] = pool
        return pool


def _get_process_pool(n_jobs: int):
    """Get persistent multiprocessing pool using spawn method."""
    with _POOL_LOCK:
        pool = _PROCESS_POOLS.get(n_jobs)
        if pool is None:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=n_jobs)
            _PROCESS_POOLS[n_jobs] = pool
        return pool


@atexit.register
def _close_pools() -> None:
    """Clean up thread and process pools at exit."""
    for pool in list(_THREAD_POOLS.values()):
        try:
            pool.close()
            pool.join()
        except Exception:  # noqa: BLE001, S110
            pass

    for pool in list(_PROCESS_POOLS.values()):
        try:
            pool.close()
            pool.join(timeout=1.0)
        except Exception:  # noqa: BLE001,
            try:
                pool.terminate()
                pool.join(timeout=1.0)
            except Exception:  # noqa: BLE001, S110
                pass


def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx


SOLVERS = {
    "scipy": solve_scipy,
}

# Some compiled extension can cause SIGKILL errors if compiled for the wrong arch
# So we have to check they won't kill everything when we import them
if check_import_safe("lap1015"):
    import lap1015

    def solve_1015_early(cost):
        return lap1015.lap_early(cost)

    def solve_1015_late(cost):
        return lap1015.lap_late(cost)

    SOLVERS["lap1015_late"] = solve_1015_late
    # SOLVERS["lap1015_early"] = lap1015_early
else:
    warnings.warn(
        """Failed to import lap1015 solver. This could be because it is not installed,
    or because it was built targeting a different architecture than supported on the current machine.
    Rebuilding the package on the current machine may fix this.""",
        ImportWarning,
        stacklevel=2,
    )


def _lap1015_releases_gil() -> bool:
    """Whether the imported lap1015 extension was built with the GIL released around the solve.

    Builds from this repo's vendored ``src/lap1015`` set ``releases_gil`` on the module;
    older or upstream builds have no such attribute, in which case threaded matching
    serialises on the GIL.
    """
    module = globals().get("lap1015")
    if module is None:
        return False
    if getattr(module, "releases_gil", False):
        return True
    # Fall back to the compiled extension, in case an older __init__.py does not re-export it.
    return bool(getattr(getattr(module, "_core", None), "releases_gil", False))


def match_individual(solver_fn, cost: np.ndarray, default_idx: np.ndarray) -> np.ndarray:
    # No valid targets: skip the solver — lap1015 returns uninitialised memory for
    # empty cost matrices, and the identity permutation is correct for every solver.
    if cost.shape[0] == 0:
        return default_idx.copy()

    pred_idx = np.asarray(solver_fn(cost), dtype=np.int32)

    if solver_fn is SOLVERS["scipy"]:
        remaining = np.ones(default_idx.shape[0], dtype=np.bool_)
        remaining[pred_idx] = False
        pred_idx = np.concatenate([pred_idx, default_idx[remaining]])
    else:
        # Non-scipy solvers must return a full permutation; fall back to the reference
        # scipy solver for this event if they return anything else (out-of-range or
        # duplicate indices would silently corrupt the loss, or assert on-device).
        n = default_idx.shape[0]
        valid = pred_idx.shape[0] == n and pred_idx.min() >= 0 and pred_idx.max() < n and np.bincount(pred_idx, minlength=n).max() == 1
        if not valid:
            warnings.warn("LAP solver returned an invalid permutation; falling back to scipy for this event.", stacklevel=2)
            return match_individual(SOLVERS["scipy"], cost, default_idx)

    return pred_idx


def match_parallel(solver_fn, costs_t: np.ndarray, lengths_np: np.ndarray, pred_dim: int, n_jobs: int = 8) -> torch.Tensor:
    """Thread-based parallel matching across batch."""
    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs
    default_idx = np.arange(pred_dim, dtype=np.int32)

    if n_jobs <= 1 or batch_size <= 1:
        results = [match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx) for i in range(batch_size)]
        return torch.from_numpy(np.stack(results, axis=0))

    def _run(i: int) -> np.ndarray:
        return match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx)

    pool = _get_thread_pool(n_jobs)
    results = pool.map(_run, range(batch_size), chunksize=chunk_size)
    return torch.from_numpy(np.stack(results, axis=0))


def _mp_match_task(args: tuple[str, str, tuple[int, int, int], str, int, int, int]) -> np.ndarray:
    solver_name, shm_name, shape, dtype_str, i, length, pred_dim = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        costs_t = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        default_idx = np.arange(pred_dim, dtype=np.int32)
        cost = costs_t[i][:length]
        return match_individual(SOLVERS[solver_name], cost, default_idx)
    finally:
        shm.close()


def match_multiprocess(
    solver_name: str,
    costs_t: np.ndarray,
    lengths_np: np.ndarray,
    pred_dim: int,
    n_jobs: int = 8,
) -> torch.Tensor:
    """Multiprocess matching using shared memory to bypass GIL.

    Raises:
        ValueError: If solver_name is not in the available SOLVERS.
    """
    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {list(SOLVERS.keys())}")

    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs

    shm = shared_memory.SharedMemory(create=True, size=costs_t.nbytes)
    try:
        shm_arr = np.ndarray(costs_t.shape, dtype=costs_t.dtype, buffer=shm.buf)
        shm_arr[...] = costs_t

        tasks = [(solver_name, shm.name, costs_t.shape, costs_t.dtype.str, i, int(lengths_np[i]), pred_dim) for i in range(batch_size)]

        pool = _get_process_pool(n_jobs)
        results = pool.map(_mp_match_task, tasks, chunksize=chunk_size)
        return torch.from_numpy(np.stack(results, axis=0))
    finally:
        try:
            shm.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()


class Matcher(nn.Module):
    def __init__(
        self,
        default_solver: str = "scipy",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 1000,
        parallel_solver: bool = False,
        parallel_backend: Literal["thread", "process"] = "thread",
        n_jobs: int = 8,
        device_solver: str | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        """ Used to match predictions to targets based on a given cost matrix.

        Parameters
        ----------
        default_solver : str
            The default solving algorithm to use.
        adaptive_solver : bool
            If true, then after every adaptive_check_interval calls of the solver,
            each solver algorithm is timed and used to determine the fastest solver, which
            is then set as the current solver.
        adaptive_check_interval : bool
            Interval for checking which solver is the fastest.
        parallel_solver : bool
            If true, then the solver will use a parallel implementation to speed up the matching.
        parallel_backend : str
            Parallel backend when parallel_solver is True. One of: 'thread', 'process'.
        n_jobs: int
            Number of jobs to use for parallel matching. Only used if parallel_solver is True.
        device_solver : str | None
            If set, solve on whichever device the costs already live on instead of copying
            them to the host, which removes the device-to-host transfer and the host stall
            that goes with it. Currently 'jv' (exact Jonker-Volgenant, needs the compiled
            torch-linear-assignment extension; see hepattn.models.device_lap). Defaults to
            None, i.e. the host solvers above. Worth turning on only when training is
            host-bound: on a GPU that is already saturated the solver's own kernels cost more
            than the stall they remove.
        verbose : bool
            If true, extra information on solver timing is printed.
        """
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        if parallel_backend not in {"thread", "process"}:
            raise ValueError(f"parallel_backend must be 'thread' or 'process', got: {parallel_backend}")
        if device_solver is not None:
            if device_solver not in DEVICE_SOLVERS:
                raise ValueError(f"Unknown device solver: {device_solver}. Available device solvers: {list(DEVICE_SOLVERS.keys())}")
            check = DEVICE_SOLVER_CHECKS.get(device_solver)
            if check is not None:
                check()
        if default_solver.startswith("lap1015") and parallel_solver and parallel_backend == "thread" and not _lap1015_releases_gil():
            warnings.warn(
                f"The installed lap1015 extension does not release the GIL while solving, so the '{default_solver}' solver "
                "cannot run in parallel: threaded matching will serialise and be roughly 2x slower than the 'scipy' solver. "
                "Rebuild the extension from this repo's vendored source in src/lap1015 (e.g. by reinstalling hepattn from "
                "source), or set default_solver: scipy in the config.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.solver = default_solver
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.parallel_solver = parallel_solver
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs
        self.device_solver = device_solver
        self.step = 0
        self.verbose = verbose
        self._pinned_buffer = None

    def _prepare_costs(self, costs, object_valid_mask=None, query_valid_mask=None):
        """Turn a [batch, num_pred, num_true] cost tensor into the host array the solvers want.

        Everything here (sanitising, masking padded queries, transposing to solver layout,
        cropping to the largest event) happens on whichever device the costs live on, so on
        GPU the single device->host copy already lands contiguous and in final shape. Doing
        the transpose/masking host-side instead costs two extra full-size numpy copies of a
        tensor that is ~1 GB per step at CLIC batch sizes.

        Returns:
            Tuple of the [batch, max_true, num_pred] host array and the per-event target counts.
        """
        costs = costs.detach().to(torch.float32)

        # Replace non-finite costs (e.g. from -inf padded mask logits) with the finite
        # sentinel used for invalid queries below: scipy treats inf as a forbidden
        # assignment and a huge finite cost identically, while lap1015 has undefined
        # behaviour on non-finite input.
        big = float(np.finfo(np.float32).max / 10)
        costs = torch.nan_to_num(costs, nan=big, posinf=big, neginf=-big)

        # If we have invalid/padded queries, set their costs to a high value
        # so they won't be matched to valid targets.
        if query_valid_mask is not None:
            invalid_query_mask = ~query_valid_mask.detach().bool().to(costs.device)
            costs = costs.masked_fill(invalid_query_mask.unsqueeze(-1), big)

        if object_valid_mask is None:
            lengths_np = np.full(costs.shape[0], costs.shape[2], dtype=np.int32)
        else:
            lengths = object_valid_mask.detach().bool().sum(dim=1)
            lengths_np = lengths.cpu().numpy().astype(np.int32, copy=False)

        # Transpose to solver layout [batch, true, pred] and drop the target rows past the
        # largest event: match_individual only ever reads cost[: lengths[k]], so the padded
        # rows are pure transfer and solve overhead.
        max_len = int(lengths_np.max()) if lengths_np.size else 0
        costs_t = costs.transpose(1, 2)[:, :max_len].contiguous()

        if not costs_t.is_cuda:
            return costs_t.numpy(), lengths_np

        return self._stage_to_host(costs_t), lengths_np

    def _stage_to_host(self, costs_t: torch.Tensor) -> np.ndarray:
        """Copy a prepared device cost tensor to the host, staged through a pinned buffer.

        A device->pageable memcpy of the cost tensor is several times slower than
        device->pinned, and profiling showed it dominating the matcher cost. The buffer is
        cached and grow-only, so allocation (expensive for pinned memory) happens rarely.

        Kept as its own method because it is the single transfer the device solver exists to
        remove, which makes it the natural thing to put a timer around.
        """
        n = costs_t.numel()
        if self._pinned_buffer is None or self._pinned_buffer.numel() < n:
            self._pinned_buffer = torch.empty(n, dtype=torch.float32, pin_memory=True)
        staged = self._pinned_buffer[:n].view(costs_t.shape)
        staged.copy_(costs_t)
        return staged.numpy()

    def _match_on_device(self, costs, object_valid_mask=None, query_valid_mask=None) -> torch.Tensor:
        """Match without ever leaving the device the costs are on.

        This is the same preparation as :meth:`_prepare_costs` minus everything that only
        exists to serve a host solver: there is no sentinel fill (the device solver treats
        non-finite and disallowed entries as forbidden directly), no pinned staging buffer and
        no ``.numpy()``. The one host round trip left is reading ``max(num_valid_targets)`` to
        crop the padded target rows, which is four bytes against the ~1 GB the host path moves.

        Returns:
            [batch, num_pred] permutation tensor, on the costs' device.
        """
        costs = costs.detach().to(torch.float32)
        batch, num_pred, num_true = costs.shape

        if object_valid_mask is None:
            lengths = torch.full((batch,), num_true, dtype=torch.long, device=costs.device)
        else:
            lengths = object_valid_mask.detach().bool().to(costs.device).sum(dim=1)

        max_len = int(lengths.max()) if batch else 0
        costs_t = costs.transpose(1, 2)[:, :max_len].contiguous()

        col_allowed = None if query_valid_mask is None else query_valid_mask.detach().bool().to(costs.device)
        row_valid = torch.arange(max_len, device=costs.device)[None, :] < lengths[:, None]

        # The solver is exact and cannot come back short, so there is nothing to check and no
        # reason to stall the host on the result -- which is the point of the device path.
        assigned = DEVICE_SOLVERS[self.device_solver](costs_t, row_valid, col_allowed)
        return assignment_to_permutation(assigned, lengths, num_pred)

    def _solve(self, costs_t: np.ndarray, lengths_np: np.ndarray, pred_dim: int) -> torch.Tensor:
        """Run the LAP solver over a prepared [batch, max_true, num_pred] host array."""
        if self.parallel_solver:
            if self.parallel_backend == "thread":
                return match_parallel(SOLVERS[self.solver], costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)
            return match_multiprocess(self.solver, costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)

        # Sequential matching
        default_idx = np.arange(pred_dim, dtype=np.int32)
        idxs = [match_individual(SOLVERS[self.solver], costs_t[k][: lengths_np[k]], default_idx) for k in range(len(costs_t))]

        return torch.from_numpy(np.stack(idxs))

    def compute_matching(self, costs, object_valid_mask=None, query_valid_mask=None):
        if not isinstance(costs, torch.Tensor):
            costs = torch.from_numpy(np.asarray(costs))

        if self.device_solver is not None:
            return self._match_on_device(costs, object_valid_mask, query_valid_mask)

        costs_t, lengths_np = self._prepare_costs(costs, object_valid_mask, query_valid_mask)

        return self._solve(costs_t, lengths_np, costs.shape[1])

    @torch.no_grad()
    def forward(self, costs, object_valid_mask=None, query_valid_mask=None):
        # The device path bypasses solver adaptation: the host solvers it would be timed
        # against are on the other side of the transfer this exists to avoid.
        if self.device_solver is not None:
            pred_idxs = self._match_on_device(costs, object_valid_mask, query_valid_mask)
            self.step += 1
            return pred_idxs

        pred_dim = costs.shape[1]
        costs_t, lengths_np = self._prepare_costs(costs, object_valid_mask, query_valid_mask)

        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs_t, lengths_np, pred_dim)

        pred_idxs = self._solve(costs_t, lengths_np, pred_dim)
        self.step += 1

        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs

    def adapt_solver(self, costs_t, lengths_np, pred_dim):
        solver_times = {}

        if self.verbose:
            print("\nAdaptive LAP Solver: Starting solver check...")

        for solver in SOLVERS:
            self.solver = solver
            start_time = time.time()
            self._solve(costs_t, lengths_np, pred_dim)
            solver_times[solver] = time.time() - start_time

            if self.verbose:
                print(f"Adaptive LAP Solver: Evaluated {solver}, took {solver_times[solver]:.2f}s")

        fastest_solver = min(solver_times, key=solver_times.get)

        if self.verbose:
            if fastest_solver != self.solver:
                print(f"Adaptive LAP Solver: Switching from {self.solver} solver to {fastest_solver} solver\n")
            else:
                print(f"Adaptive LAP Solver: Sticking with {self.solver} solver\n")

        self.solver = fastest_solver
