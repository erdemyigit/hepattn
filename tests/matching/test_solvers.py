import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import lap1015


def test_lap1015_releases_gil():
    """The installed extension must be built from this repo's `src/lap1015`, not an upstream one.

    Only this repo's build wraps the solve in `py::gil_scoped_release`. Without that,
    `Matcher(parallel_solver=True, n_jobs=16)` is a lie -- sixteen threads queue behind the GIL
    and matching runs single-threaded, which is slower than `scipy` and silently invalidates
    every throughput measurement taken with it. The failure mode is invisible at run time apart
    from one `RuntimeWarning` that scrolls past in a training log, so it is asserted here.
    """
    assert lap1015.releases_gil, (
        "the installed lap1015 extension does not release the GIL, so threaded matching "
        "serialises. Rebuild it with `pixi reinstall hepattn` -- a plain `pixi install` skips "
        "an already-installed package and will not pick up changes to src/lap1015/src/main.cpp. "
        "See the 'lap1015 Extension' section of README.md."
    )


@pytest.mark.parametrize("size", range(100, 2500, 50))
def test_lap1015(size):
    """`lap_late` is the only lap1015 entry point hepattn exposes, so it is the one pinned here.

    `lap_early` (the OpenMP variant) is deliberately absent from `SOLVERS` in
    `hepattn.models.matcher` and is covered separately below, because this build of it does not
    work at all.
    """
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    _, col_idx_scipy = linear_sum_assignment(cost)

    assert all(col_idx_scipy == lap1015.lap_late(cost))

    cost = np.random.default_rng().random((size, size)) * 1e5
    _, col_idx_scipy = linear_sum_assignment(cost)
    out_late = lap1015.lap_late(cost)

    # add col indices that are not in the output
    col_idx = np.arange(cost.shape[1])
    col_idx_scipy = np.concatenate([col_idx_scipy, col_idx[~np.isin(col_idx, col_idx_scipy)]])

    assert np.all(col_idx_scipy == out_late)


@pytest.mark.xfail(
    reason="the OpenMP path of the vendored lap1015 returns an all -1 (unassigned) solution at "
    "every problem size tried, 3x3 through 1000x1000. This is why 'lap1015_early' is commented "
    "out of SOLVERS in hepattn.models.matcher and nothing can select it. Kept as xfail so that "
    "a build which ever fixes it reports XPASS rather than passing silently.",
    strict=False,
)
@pytest.mark.parametrize("size", [3, 100])
def test_lap1015_early_omp_path(size):
    if size == 3:
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=np.float32)
    else:
        cost = (np.random.default_rng(0).random((size, size)) * 1e5).astype(np.float32)
    _, col_idx_scipy = linear_sum_assignment(cost)

    assert np.all(col_idx_scipy == lap1015.lap_early(cost))
