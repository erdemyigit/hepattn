"""The opt-in phase profiler of the driver (``keras_native.profile``, docs/hgq/port_log/PROF.md).

Tiny generated Polaris config on CLIC dummy events, 2 epochs of 8 micro-batches (accumulation 2: 8 optimizer steps),
a log row per optimizer step, a step checkpoint every 2 steps, validation every epoch. Runs: no profiler (``off``),
``async`` profiler, ``sync`` profiler with a ``jax.profiler`` trace, prefetch 2, and the controls.

- profiler off / async / sync and prefetch: the final state and every logged loss / metric are bitwise equal;
  control: a prefetch that swaps two batches changes the state (the comparison discriminates a data-order change);
- the exclusive phase times sum to the wall time: unattributed < ``PROFILE_TOLERANCE`` (1%) of the wall and >= 0, and
  the profiler's wall equals the wall the test measures around ``fit``; control: a profiler that does not record
  ``train/step`` leaves more than the tolerance unattributed;
- added syncs: with the profiler off, ``jax.block_until_ready`` runs exactly as often as before the profiler existed
  (the first step + one per logged step); async mode adds only the waits at points that block anyway; sync mode adds
  one per train step after the first and one per device transfer;
- the phase counts follow the cadence (micro-batches, logs, checkpoints, validation batches); the driver sets each
  micro-batch's position (micro-batch, optimizer step, epoch) on its data and train phases; the trace window [3, 6)
  starts after 3 dispatched micro-batches and stops after the 6th;
- exclusive-time accounting on a fake clock; the derived numbers on a fake-clock run (compile time incl. the first-step
  wait, the steady window without the first train / validation batch, the loop window without the work outside the
  train loop (setup, validation, the epoch-end ``last`` checkpoint, the train end; the epoch drain stays), the steady
  medians after the first optimizer step, the sync-mode device-idle fractions incl. drain, log, checkpoint and
  validation device phases, the estimates from a reference step); prefetch order, error propagation and thread
  shutdown;
- the ``fit`` command writes the phase CSV / JSON and the trace and passes ``--profile-step-ref-ms``;
  ``--profile-sync`` makes a sync profiler.
"""

import csv
import threading
import time

import jax
import numpy as np
import pytest

from hepattn.experiments.clic.main_jax import clic_data
from hepattn.keras_native import cli, data, loop, profile

from .s6b_utils import TINY, sources, tiny_config

EPOCHS, MICRO_PER_EPOCH, K = 2, 8, 2
MICRO, STEPS = EPOCHS * MICRO_PER_EPOCH, EPOCHS * MICRO_PER_EPOCH // K
CKPT_EVERY = 2


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return loop.enable_compilation_cache(tmp_path_factory.mktemp("jax_cache"))


@pytest.fixture(scope="module")
def rc():
    return tiny_config(ckpt_every_n_steps=CKPT_EVERY, ckpt_save_top_k=-1)


class BlockCounter:
    """Counts ``jax.block_until_ready`` calls (loop and profile call it through the ``jax`` module)."""

    def __init__(self, monkeypatch) -> None:
        self.n = 0
        real = jax.block_until_ready

        def counted(x):
            self.n += 1
            return real(x)

        monkeypatch.setattr(jax, "block_until_ready", counted)


def run(rc, tmp_path_factory, name: str, *, profiler=None, trace=None, prefetch: int = 0, count: bool = False) -> dict:
    train, val = sources(rc)
    s = loop.LoopSettings.from_run_config(rc, tmp_path_factory.mktemp(name), max_epochs=EPOCHS, prefetch=prefetch)
    t = loop.Trainer(rc, s, sample=next(iter(train.epoch(0))), train_batches=train.num_batches, profiler=profiler, trace=trace)
    blocks = None
    with pytest.MonkeyPatch.context() as mp:
        if count:
            blocks = BlockCounter(mp)
        t0 = time.perf_counter()
        t.fit(train, val)
        wall = time.perf_counter() - t0
    return {"trainer": t, "wall": wall, "blocks": None if blocks is None else blocks.n, "state": state(t), "logs": logs(t)}


def state(t) -> dict:
    keys = loop.ckpt.state_keys(t.steps.model, t.steps.optimizer)
    host = loop.host_state(t.state)
    return {f"{g}/{k}": v for g, values in zip(loop.ckpt.GROUPS, host, strict=True) for k, v in zip(keys[g], values, strict=True)}


def logs(t) -> list[dict]:
    """Every logged row (steps.csv, metrics.csv) without the wall-clock columns."""
    rows = []
    for name in ("steps.csv", "metrics.csv"):
        with (t.run_dir / name).open(newline="") as f:
            rows += [{k: v for k, v in r.items() if not k.startswith("time/")} for r in csv.DictReader(f)]
    return rows


def bitwise_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a)


@pytest.fixture(scope="module")
def runs(rc, tmp_path_factory, cache):  # noqa: ARG001
    out = {"off": run(rc, tmp_path_factory, "off", count=True)}
    out["async"] = run(rc, tmp_path_factory, "async", profiler=profile.PhaseProfiler(), count=True)
    trace_dir = tmp_path_factory.mktemp("trace")
    sync, calls = profile.PhaseProfiler(sync=True), []

    def recorded(kind: str, real):
        def call(*args, **kwargs):  # the micro-batches dispatched when the trace starts / stops
            calls.append((kind, sum(e[1].startswith("train/step") for e in sync.events)))
            return real(*args, **kwargs)

        return call

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax.profiler, "start_trace", recorded("start", jax.profiler.start_trace))
        mp.setattr(jax.profiler, "stop_trace", recorded("stop", jax.profiler.stop_trace))
        out["sync"] = run(rc, tmp_path_factory, "sync", profiler=sync, trace=profile.JaxTrace(trace_dir, 3, 6), count=True)
    out["sync"] |= {"trace_dir": trace_dir, "trace_calls": calls}
    out["prefetch"] = run(rc, tmp_path_factory, "prefetch", prefetch=2)
    return out


def test_profiler_and_prefetch_do_not_change_the_run(runs):
    ref = runs["off"]
    assert ref["trainer"].global_step == STEPS
    assert len(ref["logs"]) == STEPS + EPOCHS  # non-vacuous: every step and epoch row is compared
    for name in ("async", "sync", "prefetch"):
        assert bitwise_equal(ref["state"], runs[name]["state"]), name
        assert ref["logs"] == runs[name]["logs"], name


def test_control_a_prefetch_that_reorders_two_batches_changes_the_run(rc, tmp_path_factory, runs, monkeypatch):
    real = data.prefetch

    def swap_first_two(iterable, depth):
        def swapped():
            it = iter(iterable)
            first = next(it)
            yield next(it)
            yield first
            yield from it

        return real(swapped(), depth)

    monkeypatch.setattr(loop, "prefetch", swap_first_two)
    mutated = run(rc, tmp_path_factory, "swap", prefetch=2)
    assert not bitwise_equal(runs["off"]["state"], mutated["state"])


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_phases_sum_to_the_wall_time(runs, mode):
    r = runs[mode]
    s = r["trainer"].prof.summary()
    assert s["mode"] == mode
    assert s["unattributed_s"] >= 0
    assert s["unattributed_frac"] < profile.PROFILE_TOLERANCE, profile.format_table(s)
    # (unattributed is wall - attributed by definition; the bound above and the dropped-phase control are the checks)
    # the groups partition the attributed time; every compile phase is in group "compile", none in a steady group
    assert abs(sum(s["groups_s"].values()) - s["attributed_s"]) < 1e-9
    assert s["groups_s"][profile.COMPILE_GROUP] == pytest.approx(s["compile_s"]) and s["compile_s"] > 0
    assert not any(profile.COMPILE_SUFFIX in g for g in s["groups_s"])
    # the profiler's wall is the fit's wall (the test's clock wraps fit: the profiler's must be inside and close)
    assert s["wall_s"] <= r["wall"] < s["wall_s"] + 0.05 + 0.01 * s["wall_s"]
    assert s["micro_batches"] == MICRO


def test_phase_counts_follow_the_cadence(runs):
    phases = runs["async"]["trainer"].prof.summary()["phases"]

    def n(path: str) -> int:
        return phases.get(path, {}).get("n", 0) + phases.get(path + profile.COMPILE_SUFFIX, {}).get("n", 0)

    val_batches = -(-10 // 4)  # num_val 10, batch 4 (padded)
    assert n("train/step") == MICRO
    assert n("data/numpy") == n("data/device_put") == MICRO
    # an epoch that does not end the run asks its loader once more (exhausted) before it stops
    assert n("data") == n("data/loader") == MICRO + EPOCHS - 1
    assert n("train/log") == n("train/log/device_wait") == n("train/log/bits") == STEPS
    assert n("train/ckpt") == n("train/ckpt/npz_write") == n("train/ckpt/sha256") == STEPS // CKPT_EVERY
    assert n("epoch_end") == n("epoch_end/val") == n("epoch_end/ckpt_epoch") == EPOCHS
    assert n("epoch_end/val/forward") == n("epoch_end/val/metrics") == EPOCHS * val_batches
    assert n("epoch_end/val/tail") == EPOCHS  # 10 = 2 * 4 + 2: one short batch per validation
    assert phases["train/step" + profile.COMPILE_SUFFIX]["n"] == 2  # PeriodicEBOPs: one program per ebops_on value
    assert "train/device" not in phases  # async: no per-step block
    sync = runs["sync"]["trainer"].prof.summary()
    assert sync["phases"]["train/device"]["n"] == MICRO - 1  # the first step blocks in train/first_step_wait
    assert 0 <= sync["device_idle_frac"] < 1 and sync["steady_micro_batches"] == MICRO - 1
    assert runs["async"]["trainer"].prof.summary()["concurrent"]["match_callback"]["n"] >= MICRO


def test_driver_sets_the_data_position(runs):
    """Every train micro-batch's phases carry its position: micro-batch i (from 1), optimizer step (i - 1) // K."""
    prof = runs["async"]["trainer"].prof
    expect = [(i, (i - 1) // K, (i - 1) // MICRO_PER_EPOCH) for i in range(1, MICRO + 1)]
    steps = [e for e in prof.events if e[1] in {"train/step", "train/step" + profile.COMPILE_SUFFIX}]
    assert [e[5:] for e in steps] == expect
    # the data phase that loads a micro-batch (the last one before its step) carries the same position
    data = [e for e in prof.events if e[1] == "data"]
    assert [[d for d in data if d[0] < e[0]][-1][5:] for e in steps] == expect  # events are in seq order
    # PeriodicEBOPs: optimizer step 0 compiles the ebops-on program (micro-batch 1), step 1 the ebops-off one (K + 1)
    compiled = [e[5] for e in steps if e[1].endswith(profile.COMPILE_SUFFIX)]
    assert compiled == [1, K + 1]
    # steady medians: the micro-batches after optimizer step 0, without the compiling micro-batch K + 1
    assert prof.summary()["steady"]["train/step"]["n"] == MICRO - K - 1


def test_syncs_added_by_each_mode(runs):
    off, on, sync = (runs[m]["blocks"] for m in ("off", "async", "sync"))
    # the driver without the profiler: the first step (compile) and one block per logged step, as before profile.py
    assert off == 1 + STEPS
    ckpt_saves = STEPS // CKPT_EVERY + 2 * EPOCHS + 1  # step, (epoch, last) per epoch, the train-end last (a no-op write)
    val_batches = EPOCHS * -(-10 // 4)
    added = ckpt_saves + EPOCHS + 1 + STEPS + val_batches  # ckpt waits, epoch drains, final drain, log metrics, val
    assert on == off + added
    transfers = MICRO + val_batches
    traced = 1  # JaxTrace.stop blocks on the state
    assert sync == on + (MICRO - 1) + transfers + traced


def test_sync_run_wrote_a_trace(runs):
    files = list(runs["sync"]["trace_dir"].rglob("*.xplane.pb"))
    assert files, sorted(runs["sync"]["trace_dir"].rglob("*"))
    assert runs["sync"]["trainer"].trace.done
    # window [3, 6): the trace starts after 3 micro-batches were dispatched and stops after the 6th
    assert runs["sync"]["trace_calls"] == [("start", 3), ("stop", 6)]


def test_control_a_phase_left_out_breaks_the_sum(rc, tmp_path_factory, runs):  # noqa: ARG001  (after the cache fill)
    class Drops(profile.PhaseProfiler):
        def phase(self, name):
            return profile.NULL.phase(name) if name == "train/step" else super().phase(name)

    r = run(rc, tmp_path_factory, "drop", profiler=Drops())
    s = r["trainer"].prof.summary()
    assert s["unattributed_frac"] > profile.PROFILE_TOLERANCE


def test_exclusive_times_on_a_fake_clock():
    ticks = iter(float(t) for t in range(100))
    p = profile.PhaseProfiler(clock=lambda: next(ticks))
    with p.active():
        p.start()  # 0
        with p.phase("a"):  # 1
            with p.phase("b"):  # 2
                pass  # 3
            with p.phase("c"):  # 4
                p.at(1, 0, 0)
            # 5
        # 6
        with p.phase("d"):  # 7
            pass  # 8
        p.stop()  # 9
    s = p.summary()
    got = {k: (v["total_s"], v["excl_s"]) for k, v in s["phases"].items()}
    assert got == {"a": (5.0, 3.0), "a/b": (1.0, 1.0), "a/c": (1.0, 1.0), "d": (1.0, 1.0)}
    assert (s["wall_s"], s["unattributed_s"]) == (9.0, 3.0)  # 0-1, 6-7, 8-9 are outside every phase
    assert profile.current() is profile.NULL


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def synthetic_run(*, sync: bool) -> profile.PhaseProfiler:
    """4 micro-batches (accumulation 2) on a fake clock, the phases as the driver opens them; quarter seconds (exact).

    One epoch: setup, epoch start, the micro-batches (the first ``data`` holds the loader start in ``data/loader``),
    a log row and a step checkpoint, the epoch end (drain, validation of a full and a short batch with a compiling
    first forward, the ``last`` checkpoint), the train end (a no-op ``last`` save, the drain) and the summary.
    """
    clock = FakeClock()
    p = profile.PhaseProfiler(sync=sync, clock=clock)

    def ph(name: str, seconds: float, *, compiled: bool = False) -> None:
        with p.phase(name):
            clock.t += seconds
        if compiled:
            p.mark_compile()

    with p.active():
        p.start()
        ph("setup", 1.0), ph("epoch_start", 0.25)
        p.at(1, 0, 0)  # optimizer step 0: loader start, the compiling first step, its wait
        with p.phase("data"):
            ph("loader", 4.5)
            clock.t += 0.5
        ph("train/step", 10.0, compiled=True), ph("train/first_step_wait", 3.0)
        p.at(2, 0, 0)
        ph("data", 0.5), ph("train/step", 0.25), ph("train/device", 1.0)
        p.at(3, 1, 0)  # optimizer step 1: the second train program compiles
        ph("data", 0.25), ph("train/step", 7.0, compiled=True), ph("train/device", 1.0)
        p.at(4, 1, 0)
        ph("data", 0.75), ph("train/step", 0.5), ph("train/device", 1.0)
        with p.phase("train/log"):
            ph("bits", 0.25), ph("metrics_forward", 0.5), ph("metrics_device", 0.5), ph("csv", 0.25)
        with p.phase("train/ckpt"):
            ph("device_wait", 0.25), ph("meta", 0.25), ph("host_copy", 0.5), ph("npz_write", 2.0)
        with p.phase("epoch_end"):
            ph("drain", 0.5)
            with p.phase("val"):
                with p.phase("data"):
                    ph("loader", 1.5)
                    clock.t += 0.5
                ph("forward", 0.5, compiled=True), ph("device", 4.0), ph("device_get", 0.25)
                ph("data", 0.5), ph("forward", 0.25), ph("device", 1.0), ph("tail", 0.25), ph("device_get", 0.25)
            with p.phase("ckpt_last"):
                ph("device_wait", 0.25), ph("meta", 0.25), ph("host_copy", 0.5), ph("npz_write", 2.0), ph("sha256", 0.5)
        with p.phase("train_end/ckpt_last"):
            ph("device_wait", 0.25), ph("meta", 0.25)
        with p.phase("train_end/drain"):
            ph("wait", 0.25)
        ph("summary", 0.25)
        clock.t += 0.25  # outside every phase
        p.stop()
    return p


def test_derived_numbers_on_a_fake_clock():
    s = synthetic_run(sync=True).summary(step_ref_s=1.0)
    micro_batches = (5 + 10 + 3) + (0.5 + 0.25 + 1) + (0.25 + 7 + 1) + (0.75 + 0.5 + 1)
    log, step_ckpt, drain = 1.5, 3.0, 0.5
    val, ckpt_last, train_end = (2.0 + 0.5 + 4 + 0.25) + (0.5 + 0.25 + 1 + 0.25 + 0.25), 3.5, 0.5 + 0.25
    wall = 1.0 + 0.25 + micro_batches + log + step_ckpt + drain + val + ckpt_last + train_end + 0.25 + 0.25
    assert (s["wall_s"], s["unattributed_s"], s["micro_batches"]) == (wall, 0.25, 4)
    # one-time costs: both compiling steps, the first step's wait and validation's compiling forward; the first train
    # batch (duration: its loader start is a child phase) and the first validation batch
    assert s["compile_s"] == s["groups_s"][profile.COMPILE_GROUP] == 10 + 3 + 7 + 0.5
    assert s["one_time_s"] == {"compile": 20.5, "first_train_batch": 5.0, "first_val_batch": 2.0}
    steady_wall = wall - 20.5 - 5 - 2
    assert (s["steady_wall_s"], s["steady_micro_batches"]) == (steady_wall, 3)  # the first micro-batch is in the wait
    assert s["s_per_micro_batch"] == wall / 4
    assert s["s_per_micro_batch_excl_compile"] == (wall - 20.5) / 4
    assert s["s_per_micro_batch_steady"] == steady_wall / 3
    # outside the loop window (non-compile): setup, epoch start, validation (its first batch included, its compiling
    # forward not), the epoch-end last checkpoint, the train end, the summary; epoch_end/drain stays in the loop
    outside = 1.0 + 0.25 + (val - 0.5) + ckpt_last + train_end + 0.25
    loop_wall = wall - 20.5 - 5 - outside
    assert loop_wall == (0.5 + 0.25 + 1) + (0.25 + 1) + (0.75 + 0.5 + 1) + log + step_ckpt + drain + 0.25  # by hand
    assert (s["outside_loop_s"], s["loop_wall_s"]) == (outside, loop_wall)
    assert s["s_per_micro_batch_loop"] == loop_wall / 3
    # medians of the micro-batches after the first optimizer step (step 1: micro-batches 3 and 4; 3 compiled)
    assert s["steady"] == {
        "data": {"n": 2, "median_ms": 500.0, "mean_ms": 500.0},
        "train/step": {"n": 1, "median_ms": 500.0, "mean_ms": 500.0},
        "train/device": {"n": 2, "median_ms": 1000.0, "mean_ms": 1000.0},
    }
    # busy: train dispatch + wait, the log row's bits / forward / wait, the step checkpoint's wait / read-outs, the
    # epoch drain; outside the loop: validation's forward / wait / tail / read-outs, the last checkpoints' wait /
    # read-outs, the train-end drain. Not the compiles, the loaders, the npz writes, the hashes or the csv
    busy_loop = (0.25 + 0.5) + 3 * 1.0 + (0.25 + 0.5 + 0.5) + (0.25 + 0.25 + 0.5) + 0.5
    busy_outside = (4.0 + 0.25 + 0.25 + 1.0 + 0.25 + 0.25) + (0.25 + 0.25 + 0.5) + (0.25 + 0.25) + 0.25
    busy = busy_loop + busy_outside
    assert (s["device_busy_s"], s["device_busy_loop_s"]) == (busy, busy_loop)
    assert s["device_idle_frac"] == pytest.approx(1 - busy / steady_wall, rel=1e-15)
    assert s["device_idle_frac_loop"] == pytest.approx(1 - busy_loop / loop_wall, rel=1e-15)
    assert s["device_idle_frac_wall"] == pytest.approx(1 - busy / wall, rel=1e-15)
    assert s["device_idle_frac_est"] == pytest.approx(1 - 3 * 1.0 / steady_wall, rel=1e-15)
    assert s["device_idle_frac_est_loop"] == pytest.approx(1 - 3 * 1.0 / loop_wall, rel=1e-15)
    assert s["device_idle_frac_est_wall"] == pytest.approx(1 - 4 * 1.0 / wall, rel=1e-15)
    a = synthetic_run(sync=False).summary()
    assert a["mode"] == "async" and not {"device_busy_s", "device_idle_frac_loop", "device_idle_frac_est"} & a.keys()
    assert (a["s_per_micro_batch_steady"], a["s_per_micro_batch_loop"]) == (s["s_per_micro_batch_steady"], s["s_per_micro_batch_loop"])


def test_cli_profiler_flags():
    parse = cli.build_parser().parse_args
    base = ["fit", "--config", str(TINY), "--run-dir", "unused"]
    assert cli.profilers(parse(base)) == (None, None)
    prof, trace = cli.profilers(parse([*base, "--profile-phases", "p.csv"]))
    assert prof.sync is False and trace is None
    assert cli.profilers(parse([*base, "--profile-phases", "p.csv", "--profile-sync"]))[0].sync is True
    _, trace = cli.profilers(parse([*base, "--jax-trace", "t", "--jax-trace-window", "3:7"]))
    assert (trace.first, trace.last) == (3, 7)


def test_null_profiler_is_inert():
    assert profile.current() is profile.NULL
    x = object()
    assert profile.NULL.wait(x) is x and profile.NULL.block(x) is x
    assert profile.NULL.phase("a") is profile.NULL.phase("b")
    p = profile.PhaseProfiler()
    with p.active():
        seen = []
        th = threading.Thread(target=lambda: seen.append(profile.current()))
        th.start()
        th.join()
        assert seen == [profile.NULL]  # other threads (prefetch, matching callback) never record phases
        assert profile.current() is p


def test_prefetch_order_errors_and_shutdown():
    assert list(data.prefetch(range(50), 3)) == list(range(50))

    def failing():
        yield from range(3)
        raise KeyError("boom")

    got = []
    with pytest.raises(KeyError, match="boom"):
        got.extend(data.prefetch(failing(), 2))
    assert got == [0, 1, 2]
    produced = []

    def endless():
        i = 0
        while True:
            produced.append(i)
            yield i
            i += 1

    gen = data.prefetch(endless(), 2)
    assert [next(gen) for _ in range(5)] == list(range(5))
    gen.close()
    assert not any(t.name == "hepattn-prefetch" and t.is_alive() for t in threading.enumerate())
    stopped = len(produced)
    time.sleep(0.2)
    assert len(produced) == stopped <= 5 + 2 + 2  # read ahead at most the queue and one item in hand
    with pytest.raises(ValueError, match="depth"):
        next(data.prefetch(range(3), 0))


def test_fit_command_writes_the_profile(tmp_path, cache):  # noqa: ARG001
    out = tmp_path / "phases.csv"
    argv = ["fit", "--config", str(TINY), "--run-dir", str(tmp_path / "run"), "--dummy-data", "--max-epochs", "1", "--compile-cache", ""]
    argv += ["--profile-phases", str(out), "--prefetch", "1", "--jax-trace", str(tmp_path / "trace"), "--jax-trace-window", "1:2"]
    argv += ["--profile-step-ref-ms", "250"]
    args = cli.build_parser().parse_args(argv)
    summary = cli.fit(args, clic_data)
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == profile.EVENT_FIELDS
    assert {r["path"] for r in rows} >= {"data", "train/log", "epoch_end/val/forward"}
    assert "data/loader" not in {r["path"] for r in rows}  # prefetch: the loader runs in the prefetch thread
    report = summary["profile"]
    assert report["mode"] == "async"
    assert report["micro_batches"] == MICRO_PER_EPOCH and report["unattributed_frac"] < profile.PROFILE_TOLERANCE
    # --profile-step-ref-ms: the async idle estimates from the reference step
    assert report["step_ref_s"] == 0.25
    assert report["device_idle_frac_est_loop"] == pytest.approx(1 - report["steady_micro_batches"] * 0.25 / report["loop_wall_s"])
    assert (tmp_path / "phases.csv.json").is_file()
    assert list((tmp_path / "trace").rglob("*.xplane.pb"))
    assert summary["settings"]["prefetch"] == 1
