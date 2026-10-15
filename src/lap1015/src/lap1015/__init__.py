from __future__ import annotations

from functools import partial

from . import _core
from ._core import linear_sum_assignment

lap_late = partial(linear_sum_assignment, omp=False, eps=True)
lap_early = partial(linear_sum_assignment, omp=True, eps=True)

# True only for extensions built from this source tree, which release the GIL while
# solving. Older builds do not define the flag, so getattr keeps them importable.
releases_gil: bool = bool(getattr(_core, "releases_gil", False))

__all__ = ["lap_early", "lap_late", "linear_sum_assignment", "releases_gil"]
