"""Opt-in wall-time profiler of the training driver's phases (docs/hgq/port_log/PROF.md).

The driver (``loop.Trainer``), the batch conversion (``data.HostBatches``) and the checkpoint writer open named phases
with ``current().phase(name)``. With no profiler active, ``current()`` is ``NULL``, whose ``phase`` returns one shared
``contextlib.nullcontext`` and whose ``wait`` returns its argument: no clock read, no device sync, no other effect.

``PhaseProfiler`` (activated with ``with profiler.active():``) records every phase of the thread that activated it:
its path (nested phases join with ``/``: ``epoch_end/val/forward``), start, duration and exclusive time (duration minus
the time of its child phases), and the data position (micro-batch, optimizer step, epoch) the driver sets with
``at``. Phases of other threads (the matching callback, a prefetch thread) are not recorded as phases; the matching
callback is timed as a concurrent counter (``time_host_fn``). ``summary()`` attributes the wall time between
``start()`` and ``stop()`` to the phases: the exclusive times of all phases plus ``unattributed`` (the Python
bookkeeping between phases) sum to the wall time exactly, so the check that matters is that ``unattributed`` is small
(``PROFILE_TOLERANCE`` of the wall time); a phase that the driver forgets to open shows up there.

What a phase measures under JAX's asynchronous dispatch:

- ``async`` mode (default): no sync is added anywhere the loop would not block anyway. ``train/step`` is the time to
  DISPATCH the train step; it grows only when the device queue is full (back-pressure: the device is the
  bottleneck). Time the loop spends blocked on device results is in the phases where it already blocks
  (``train/log/device_wait``: ``_log_step``'s ``block_until_ready``) or where the profiler adds a ``wait``. It adds
  five: ``*/ckpt*/device_wait`` (before each checkpoint save), ``epoch_end/drain``, ``train_end/drain/wait``,
  ``train/log/metrics_device`` and ``epoch_end/val/device`` (one per validation batch). Each blocks on values that the
  next statement reads on the host anyway (the checkpoint copies the whole state to the host; the host metrics read
  the predictions; validation and the epoch checkpoint read results computed from the state), so they move a wait
  into a named phase without changing what overlaps with what, except: the first validation batch is no longer loaded
  while the device drains (at most once per epoch), and the wait on a short validation batch comes before the dispatch
  of its ``tail`` program, not after it (once per validation).
- ``sync`` mode: additionally ``train/device`` blocks on every train step's loss right after the dispatch, and
  ``data/device_put`` blocks on the transferred batch. Every train step then runs alone, so ``train/step +
  train/device`` is the step's device time inside the real loop and the host phases no longer overlap with it; the
  wall time grows by the overlap that ``async`` mode had. Compare the two modes (and a run without the profiler) to
  get the perturbation.

One-time costs and the steady window: ``compile_s`` (every ``+compile`` phase and ``train/first_step_wait``), the first
train batch (loader worker start) and the first validation batch (validation loader start) happen once per process.
``steady_wall_s`` is the wall time without them and ``steady_micro_batches`` the micro-batches without the first
(its device time is inside ``train/first_step_wait``); ``s_per_micro_batch_steady`` is their ratio. The steady window
still holds the work outside the train loop (``is_outside_loop``: setup, epoch start, validation, the epoch-end epoch
and ``last`` checkpoints, the train end, summary; a full checkpoint is 1.1 GB at production dims). The loop window
(``loop_wall_s``) is the steady window without their non-compile exclusive time (``outside_loop_s``): the
micro-batches, the log rows and step checkpoints at their cadence, ``epoch_end/drain`` (device time of steps already
queued) and the Python between phases. ``s_per_micro_batch_loop`` = ``loop_wall_s / steady_micro_batches`` is the
rate to compare with a device step; ``s_per_micro_batch_steady`` is the end-to-end rate without one-time costs.

The device-idle fraction is measured only in ``sync`` mode, over the steady window: ``1 - busy / steady_wall_s``, with
``busy`` the exclusive time of the non-compile phases from a device dispatch to its result (``DEVICE_PHASES``: train
step and its wait, the log row's and validation's forward and waits, the checkpoint waits and read-outs, the drains).
In ``sync`` mode every dispatch of device work is followed by its wait inside one of these phases, so ``busy`` is an
upper bound on the device's busy time and the idle fraction a lower bound. Batch transfers (``data/device_put*``) are
not in ``busy``: the compute units are idle while the loop waits for a transfer. In ``async`` mode the idle fraction
is estimated as ``1 - steady_micro_batches * step_ref_s / steady_wall_s`` when a reference step time is given
(``summary(step_ref_s=...)``, e.g. the median ``train/step + train/device`` of a ``sync`` run or the G3 step). The
``*_loop`` keys hold the same fractions over the loop window, with ``busy`` restricted to the phases inside it
(``device_busy_loop_s``); the ``*_wall`` keys over the whole wall (one-time costs included: not the steady loop).
"""

import contextlib
import csv
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax

PROFILE_TOLERANCE = 0.01  # largest unattributed fraction of the wall time the tests accept
_NULL_CONTEXT = contextlib.nullcontext()
COMPILE_SUFFIX = "+compile"
COMPILE_GROUP = "compile"  # summary()["groups_s"]: every compile phase (any path), apart from the steady-state groups
# leaf names of the phases from a device dispatch to its result (module docstring); compile phases are excluded
DEVICE_PHASES = frozenset({
    "step", "device", "device_wait", "device_get", "metrics_forward", "metrics_device", "bits", "forward", "tail", "meta",
    "host_copy", "drain", "wait",
})  # fmt: skip
EVENT_FIELDS = ("seq", "path", "start_s", "dur_s", "excl_s", "micro_step", "global_step", "epoch")


class NullProfiler:
    """The inactive profiler: every call is a no-op (no clock read, no sync)."""

    enabled = False
    sync = False

    def phase(self, name: str):
        return _NULL_CONTEXT

    def wait(self, tree, name: str = "device_wait"):
        return tree

    def block(self, tree, name: str = "device"):
        return tree

    def at(self, micro_step: int, global_step: int, epoch: int) -> None:
        pass


NULL = NullProfiler()
_ACTIVE: list = [NULL]


def current():
    """The active profiler of the calling thread's process, or ``NULL``; ``NULL`` for any thread but the profiler's."""
    prof = _ACTIVE[-1]
    if prof is NULL or threading.get_ident() != prof.thread:
        return NULL
    return prof


class _Phase:
    __slots__ = ("child", "name", "prof", "start")

    def __init__(self, prof: "PhaseProfiler", name: str) -> None:
        self.prof, self.name = prof, name

    def __enter__(self):
        self.child = 0.0
        self.prof._stack.append(self)  # noqa: SLF001
        self.start = self.prof.clock()
        return self

    def __exit__(self, *exc) -> None:
        end = self.prof.clock()
        prof = self.prof
        stack = prof._stack  # noqa: SLF001
        stack.pop()
        dur = end - self.start
        if stack:
            stack[-1].child += dur
        path = "/".join([p.name for p in stack] + [self.name])
        prof.events.append((len(prof.events), path, self.start - prof.t0, dur, dur - self.child, *prof.position))


class PhaseProfiler:
    """Records the driver's phases (module docstring).

    Args:
        sync: ``sync`` mode (block on every train step and device transfer) instead of ``async``.
        clock: The clock (seconds).
    """

    enabled = True

    def __init__(self, *, sync: bool = False, clock: Callable[[], float] = time.perf_counter) -> None:
        self.sync = sync
        self.clock = clock
        self.thread = threading.get_ident()
        self.events: list[tuple] = []
        self.position = (0, 0, 0)
        self.concurrent: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stack: list[_Phase] = []
        self.t0 = self.t1 = None

    # ------------------------------------------------------------------------------------------------ recording

    @contextlib.contextmanager
    def active(self) -> Iterator["PhaseProfiler"]:
        """Make this the profiler that ``current()`` returns (in the thread that created it).

        Yields:
            This profiler.
        """
        _ACTIVE.append(self)
        try:
            yield self
        finally:
            _ACTIVE.remove(self)

    def start(self) -> None:
        self.t0 = self.clock()

    def stop(self) -> None:
        self.t1 = self.clock()

    def at(self, micro_step: int, global_step: int, epoch: int) -> None:
        self.position = (micro_step, global_step, epoch)

    def phase(self, name: str) -> _Phase:
        return _Phase(self, name)

    def wait(self, tree, name: str = "device_wait"):
        """Block on ``tree`` in phase ``name`` (both modes; only where the loop blocks on ``tree`` next anyway)."""
        with self.phase(name):
            return jax.block_until_ready(tree)

    def block(self, tree, name: str = "device"):
        """``sync`` mode: block on ``tree`` in phase ``name``; ``async`` mode: no-op."""
        if not self.sync:
            return tree
        return self.wait(tree, name)

    def mark_compile(self) -> None:
        """Relabel the phase that ended last as ``<path>+compile`` (a jitted call that traced: trace, compile or
        persistent-cache load), so steady-state phases exclude it.
        """
        e = self.events[-1]
        self.events[-1] = (e[0], e[1] + COMPILE_SUFFIX, *e[2:])

    def add_concurrent(self, name: str, seconds: float) -> None:
        with self._lock:
            self.concurrent[name].append(seconds)

    def time_host_fn(self, fn: Callable, name: str = "match_callback") -> Callable:
        """``fn`` with its wall time per call recorded under ``concurrent[name]`` (any thread); same results."""

        def timed(*args, **kwargs):
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add_concurrent(name, time.perf_counter() - t)

        return timed

    # ------------------------------------------------------------------------------------------------ results

    def write_events(self, path: str | Path) -> Path:
        """One row per phase: ``EVENT_FIELDS`` (times in seconds from ``start()``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(EVENT_FIELDS)
            writer.writerows(self.events)
        return path

    def summary(self, *, step_ref_s: float | None = None) -> dict[str, Any]:
        """Wall time attributed to phases (exclusive times), per-phase statistics, concurrent counters, device idle.

        Raises:
            RuntimeError: If ``start`` / ``stop`` were not called or a phase is still open.
        """
        if self.t0 is None or self.t1 is None:
            raise RuntimeError("summary() needs start() and stop()")
        if self._stack:
            raise RuntimeError(f"phases still open: {[p.name for p in self._stack]}")
        wall = self.t1 - self.t0
        by_path: dict[str, list[tuple]] = defaultdict(list)
        for e in self.events:
            by_path[e[1]].append(e)
        micro = len(by_path.get("train/step", ())) + len(by_path.get("train/step" + COMPILE_SUFFIX, ()))
        phases = {}
        for path, rows in sorted(by_path.items()):
            durs = [r[3] for r in rows]
            excl = sum(r[4] for r in rows)
            phases[path] = {
                "n": len(rows),
                "total_s": sum(durs),
                "excl_s": excl,
                "excl_frac": excl / wall if wall else float("nan"),
                "median_ms": 1e3 * statistics.median(durs),
                "max_ms": 1e3 * max(durs),
                "excl_per_micro_ms": 1e3 * excl / micro if micro else float("nan"),
            }
        attributed = sum(p["excl_s"] for p in phases.values())
        groups: dict[str, float] = defaultdict(float)
        for path, p in phases.items():
            groups[COMPILE_GROUP if is_compile(path) else group_of(path)] += p["excl_s"]
        out: dict[str, Any] = {
            "mode": "sync" if self.sync else "async",
            "wall_s": wall,
            "attributed_s": attributed,
            "unattributed_s": wall - attributed,
            "unattributed_frac": (wall - attributed) / wall if wall else float("nan"),
            "tolerance_frac": PROFILE_TOLERANCE,
            "micro_batches": micro,
            "s_per_micro_batch": wall / micro if micro else float("nan"),
            "groups_s": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
            "phases": phases,
            "concurrent": {
                name: {"n": len(v), "total_s": sum(v), "median_ms": 1e3 * statistics.median(v), "max_ms": 1e3 * max(v)}
                for name, v in self.concurrent.items()
                if v
            },
        }
        compile_s = sum(p["excl_s"] for path, p in phases.items() if is_compile(path))
        first_train = by_path["data"][0][3] if by_path.get("data") else 0.0
        first_val = by_path["epoch_end/val/data"][0][3] if by_path.get("epoch_end/val/data") else 0.0
        steady_wall = wall - compile_s - first_train - first_val
        steady_micro = micro - len(by_path.get("train/first_step_wait", ()))
        out["compile_s"] = compile_s
        out["one_time_s"] = {"compile": compile_s, "first_train_batch": first_train, "first_val_batch": first_val}
        out["steady_wall_s"] = steady_wall
        out["steady_micro_batches"] = steady_micro
        out["s_per_micro_batch_excl_compile"] = (wall - compile_s) / micro if micro else float("nan")
        out["s_per_micro_batch_steady"] = steady_wall / steady_micro if steady_micro > 0 else float("nan")
        # the first validation batch is inside an outside phase (epoch_end/val/data), so it is not subtracted again
        outside = sum(p["excl_s"] for path, p in phases.items() if not is_compile(path) and is_outside_loop(path))
        loop_wall = wall - compile_s - first_train - outside
        out["outside_loop_s"] = outside
        out["loop_wall_s"] = loop_wall
        out["s_per_micro_batch_loop"] = loop_wall / steady_micro if steady_micro > 0 else float("nan")
        out["steady"] = self._steady(by_path)

        def idle(busy_s: float, window_s: float) -> float:
            return 1 - busy_s / window_s if window_s > 0 else float("nan")

        if self.sync:
            # a compile phase's leaf ends in COMPILE_SUFFIX, so it never matches DEVICE_PHASES
            device = {path: p["excl_s"] for path, p in phases.items() if path.rsplit("/", 1)[-1] in DEVICE_PHASES}
            busy = sum(device.values())
            busy_loop = sum(s for path, s in device.items() if not is_outside_loop(path))
            out["device_busy_s"] = busy
            out["device_busy_loop_s"] = busy_loop
            out["device_idle_frac"] = idle(busy, steady_wall)
            out["device_idle_frac_loop"] = idle(busy_loop, loop_wall)
            out["device_idle_frac_wall"] = idle(busy, wall)
        if step_ref_s is not None and micro:
            out["step_ref_s"] = step_ref_s
            out["device_idle_frac_est"] = idle(steady_micro * step_ref_s, steady_wall)
            out["device_idle_frac_est_loop"] = idle(steady_micro * step_ref_s, loop_wall)
            out["device_idle_frac_est_wall"] = idle(micro * step_ref_s, wall)
        return out

    def _steady(self, by_path: dict[str, list[tuple]]) -> dict[str, Any]:
        """Per-micro-batch medians of the micro-batch phases after the run's first optimizer step (no compile)."""
        steps = by_path.get("train/step", []) + by_path.get("train/step" + COMPILE_SUFFIX, [])
        if not steps:
            return {}
        first = min(e[6] for e in steps)
        rows = {}
        for path in ("data", "train/step", "train/device", "train/beta"):
            durs = [e[3] for e in by_path.get(path, []) if e[6] > first]
            if durs:
                rows[path] = {"n": len(durs), "median_ms": 1e3 * statistics.median(durs), "mean_ms": 1e3 * statistics.fmean(durs)}
        return rows


def is_compile(path: str) -> bool:
    """A phase that traced / compiled (or loaded from the cache), or the wait for the first (compiling) step."""
    return path.endswith(COMPILE_SUFFIX) or path == "train/first_step_wait"


OUTSIDE_LOOP = frozenset({"setup", "epoch_start", "epoch_end", "train_end", "summary"})  # first path segments
LOOP_DRAIN = "epoch_end/drain"  # device time of the steps already queued: loop time, though it runs at the epoch end


def is_outside_loop(path: str) -> bool:
    """A phase outside the loop window: setup, epoch start, epoch end (validation, the epoch and ``last`` checkpoints;
    not the drain), train end, summary.
    """
    return path.split("/", 1)[0] in OUTSIDE_LOOP and path != LOOP_DRAIN


def group_of(path: str) -> str:
    """The attribution group of a phase path: the first segment, with the train sub-phases split out."""
    head, _, rest = path.partition("/")
    if head == "train" and rest:
        return f"train/{rest.split('/', 1)[0]}"
    if head == "epoch_end" and rest:
        return f"epoch_end/{rest.split('/', 1)[0]}"
    return head


def format_table(summary: dict, *, top: int = 40) -> str:
    """A plain-text table of the phases (exclusive time, share of the wall time, per-micro-batch cost)."""
    lines = [
        f"mode {summary['mode']}  wall {summary['wall_s']:.3f} s  micro-batches {summary['micro_batches']}  "
        f"{1e3 * summary['s_per_micro_batch']:.1f} ms/micro  unattributed {summary['unattributed_s']:.4f} s "
        f"({100 * summary['unattributed_frac']:.2f}%)  steady {1e3 * summary['s_per_micro_batch_steady']:.1f} ms/micro  "
        f"loop {1e3 * summary['s_per_micro_batch_loop']:.1f} ms/micro",
        f"{'phase':48s} {'n':>6s} {'excl_s':>10s} {'%wall':>7s} {'ms/micro':>9s} {'median_ms':>10s}",
    ]
    rows = sorted(summary["phases"].items(), key=lambda kv: -kv[1]["excl_s"])[:top]
    lines += [
        f"{path:48s} {p['n']:6d} {p['excl_s']:10.3f} {100 * p['excl_frac']:7.2f} {p['excl_per_micro_ms']:9.2f} {p['median_ms']:10.2f}"
        for path, p in rows
    ]
    return "\n".join(lines)


class JaxTrace:
    """``jax.profiler`` trace of micro-batches ``[first, last)`` (counted from the run start, resumes included).

    The trace stops after blocking on the state, so the window's device work is inside the trace (one sync).
    """

    def __init__(self, directory: str | Path, first: int, last: int) -> None:
        if not 0 <= first < last:
            raise ValueError(f"trace window [{first}, {last}) is empty")
        self.directory, self.first, self.last = str(directory), first, last
        self.running = self.done = False

    def before(self, micro_step: int) -> None:
        if not self.running and not self.done and micro_step >= self.first:
            jax.profiler.start_trace(self.directory)
            self.running = True

    def after(self, micro_step: int, state) -> None:
        """Call with the number of micro-batches done; stops the trace once it reaches ``last``."""
        if self.running and micro_step >= self.last:
            self.stop(state)

    def stop(self, state=None) -> None:
        if self.running:
            if state is not None:
                jax.block_until_ready(state)
            jax.profiler.stop_trace()
            self.running, self.done = False, True


def parse_window(text: str) -> tuple[int, int]:
    """``"a:b"`` -> ``(a, b)``.

    Raises:
        ValueError: If ``text`` is not two integers separated by a colon.
    """
    a, sep, b = text.partition(":")
    if not sep:
        raise ValueError(f"trace window {text!r}: expected FIRST:LAST micro-batches")
    return int(a), int(b)
