"""polaris/18_loop_profile.py arithmetic that runs without a GPU (docs/hgq/port_log/PROF.md).

- ``smoke-timing`` (hypothesis H0): run B trains only the micro-batches after its resume point, the one-time costs are
  everything outside ``fit`` plus the first (compiling) step, and the log windows come from ``steps.csv``;
- an unprofiled variant reports only its fit rate (no compile-free number, no overhead over G3); a profiled one's
  overhead over G3 uses the loop rate (no compiles, no first batches, no work outside the train loop), the steady
  overhead the steady rate; the log windows after the first are all the windows but the first;
- the perturbation is the ratio of the median log windows after the first of ``pure_log`` and ``pure_noprof`` (not the
  whole-fit ``fit_s`` ratio), valid only with ``MIN_WINDOWS`` windows in each; ``report`` merges partial JSONs;
- ``run_fit`` builds each variant's command and ``XLA_FLAGS`` (the H7 pairs differ only in the determinism flag; the
  perturbation pair only in ``--profile-phases``), passes the period overrides and deletes the checkpoints after the
  fit; ``run_loader`` never gets the determinism flag;
- the budget: a variant starts only if the time since the job start plus the longest variant so far (loaders too) fits.
"""

import csv
import importlib.util
import json
import pathlib
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lp():
    spec = importlib.util.spec_from_file_location("loop_profile_script", REPO / "polaris/18_loop_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_smoke(tmp_path: pathlib.Path) -> pathlib.Path:
    run = {"wall_s": 600.0, "fit_s": 480.0, "first_step_s": 218.0, "build_s": 110.0, "micro_step": 1188}
    resumed = {"wall_s": 300.0, "fit_s": 270.0, "first_step_s": 61.0, "build_s": 20.0, "micro_step": 1188}
    smoke = {"args": {"run_root": str(tmp_path), "resume_step": 150, "accum": 4}, "run_a": run, "run_b": resumed}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(smoke))
    (tmp_path / "run_a").mkdir()
    with (tmp_path / "run_a/steps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["step", "time/ms_per_micro_batch"])
        w.writeheader()
        # asymmetric after the first: dropping one more window changes the median, the mean and the max
        w.writerows({"step": 10 * (i + 1), "time/ms_per_micro_batch": v} for i, v in enumerate([5000.0, 100.0, 200.0, 600.0]))
    return path


def test_smoke_timing_splits_one_time_costs(lp, tmp_path):
    t = lp.smoke_timing(write_smoke(tmp_path))
    a, b = t["run_a"], t["run_b"]
    assert a["micro_batches"] == 1188
    assert b["micro_batches"] == 1188 - 150 * 4  # run B trains from its resume point only
    assert b["s_per_micro_wall"] == pytest.approx(300.0 / 588)
    assert a["outside_fit_s"] == pytest.approx(120.0)
    assert a["one_time_s_per_micro"] == pytest.approx((120.0 + 218.0) / 1188)
    assert a["s_per_micro_fit_excl_first_step"] == pytest.approx((480.0 - 218.0) / 1187)
    assert a["s_per_micro_wall"] == pytest.approx(a["one_time_s_per_micro"] + (480.0 - 218.0) / 1188)
    assert a["log_windows"] == {
        "n": 4,
        "first_ms": 5000.0,
        "median_ms": 400.0,
        "median_ms_after_first": 200.0,
        "mean_ms_after_first": 300.0,
        "max_ms_after_first": 600.0,
    }
    assert "log_windows" not in b  # no steps.csv for run B here


def profile_summary(**kw) -> dict:
    out = {"s_per_micro_batch": 2.5, "s_per_micro_batch_excl_compile": 0.5, "s_per_micro_batch_steady": 0.25, "s_per_micro_batch_loop": 0.2}
    out |= {"outside_loop_s": 0.8}
    out |= {"one_time_s": {"compile": 32.0, "first_train_batch": 4.0, "first_val_batch": 0.0}, "groups_s": {"train/step": 4.0}}
    return out | {"unattributed_frac": 0.0, "steady": {}, "concurrent": {}} | kw


def test_unprofiled_variant_and_steady_overhead(lp):
    noprof = lp.derive({"result": {"micro_step": 16, "fit_s": 32.0, "cache_events": {"cache_misses": 0}}}, 161.6)
    assert noprof["s_per_micro"] == 2.0 and noprof["cache_misses"] == 0
    assert all(noprof[k] is None for k in ("s_per_micro_excl_compile", "s_per_micro_steady", "s_per_micro_loop", "overhead_vs_g3_ms"))
    assert noprof["overhead_steady_vs_g3_ms"] is None
    profiled = lp.derive({"result": {"micro_step": 16, "fit_s": 40.0}, "profile": profile_summary()}, 161.6)
    # the overhead uses the loop rate (0.2 s: no validation, epoch-end checkpoint, setup), not the steady rate (0.25 s)
    # or the rate without compiles only (0.5 s); the steady overhead is kept for the end-to-end comparison
    assert profiled["overhead_vs_g3_ms"] == pytest.approx(200.0 - 161.6)
    assert profiled["overhead_steady_vs_g3_ms"] == pytest.approx(250.0 - 161.6)
    assert (profiled["s_per_micro_excl_compile"], profiled["s_per_micro_steady"], profiled["s_per_micro_loop"]) == (0.5, 0.25, 0.2)
    assert profiled["outside_loop_s"] == 0.8 and "device_idle_frac_loop" not in profiled
    synced = lp.derive({"result": {"micro_step": 16, "fit_s": 40.0}, "profile": profile_summary(device_idle_frac_loop=0.125)}, 161.6)
    assert synced["device_idle_frac_loop"] == 0.125
    assert profiled["groups_ms_per_micro"] == {"train/step": 250.0}
    assert profiled["cache_misses"] is None
    text = lp.table({"variants": {"pure_noprof": {"derived": noprof}, "pure": {"derived": profiled}}})
    assert "     0  no profiler: fit_s 32.0" in text  # cache misses column: 0
    assert "   0.250    0.200       38.4" in text  # steady, loop, loop - G3


def windows(median_after_first: float, n: int) -> dict:
    return {"n": n, "first_ms": 9000.0, "median_ms_after_first": median_after_first}


def test_perturbation_compares_steady_log_windows(lp):
    on = {"result": {"fit_s": 100.0}, "log_windows": windows(210.0, 10)}
    off = {"result": {"fit_s": 50.0}, "log_windows": windows(200.0, 10)}
    result = {"variants": {"pure_log": on, "pure_noprof": off}}
    p = lp.perturbation(result)
    # the steady ratio (1.05), not the whole-fit ratio (2.0: one-time costs of different size in the two runs)
    assert p["valid"] and p["ratio"] == pytest.approx(210.0 / 200.0) and p["fit_s_ratio"] == 2.0
    assert p["windows_after_first"] == [9, 9]
    result["perturbation"] = p
    assert "= 1.050" in lp.table(result) and "NOT VALID" not in lp.table(result)
    # too few windows after the first in one run: not valid
    off["log_windows"] = windows(200.0, lp.MIN_WINDOWS)
    p = lp.perturbation(result)
    assert not p["valid"] and p["ratio"] is None and p["windows_after_first"] == [9, lp.MIN_WINDOWS - 1]
    result["perturbation"] = p
    assert "NOT VALID" in lp.table(result)
    del off["log_windows"]
    assert not lp.perturbation(result)["valid"]
    # the old pair (pure without log rows) does not give a perturbation
    assert lp.perturbation({"variants": {"pure": on, "pure_noprof": off}}) is None


def test_report_merges_partial_results(lp, tmp_path):
    # job 1 was killed after pure_log (no table, no perturbation); job 2 re-ran pure_log and ran pure_noprof
    a = {"variants": {"pure_log": {"result": {"fit_s": 100.0}, "log_windows": windows(210.0, 10)}}, "skipped": ["pure_noprof", "pure_sync"]}
    b = {"variants": {"pure_log": {"result": {"fit_s": 90.0}, "log_windows": windows(220.0, 10)}}, "skipped": ["pure_sync"]}
    b["variants"]["pure_noprof"] = {"result": {"fit_s": 50.0}, "log_windows": windows(200.0, 10)}
    paths = []
    for name, r in (("a", a), ("b", b)):
        paths.append(tmp_path / f"{name}.json")
        paths[-1].write_text(json.dumps(r))
    merged = lp.report([str(x) for x in paths])
    assert merged["perturbation"]["ratio"] == pytest.approx(220.0 / 200.0)  # the later file's pure_log
    assert merged["sources"] == {"pure_log": str(paths[1]), "pure_noprof": str(paths[1])}
    assert merged["skipped"] == ["pure_sync"] and "skipped (budget): pure_sync" in merged["table"]  # pure_noprof ran in b
    assert lp.report([str(paths[0])])["perturbation"] is None


def fake_driver(calls: list, steps_ms: list[float]):
    """``subprocess.run`` of a fit: records the command and XLA_FLAGS, writes a checkpoint and steps.csv, prints RESULT."""

    def run(cmd, env, **_):
        calls.append((cmd, env["XLA_FLAGS"]))
        run_dir = pathlib.Path(cmd[cmd.index("--run-dir") + 1])
        (run_dir / "ckpts").mkdir(parents=True)
        (run_dir / "ckpts/last.npz").write_bytes(b"x" * 1000)
        if steps_ms:
            with (run_dir / "steps.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, ["step", "time/ms_per_micro_batch"])
                w.writeheader()
                w.writerows({"step": i, "time/ms_per_micro_batch": v} for i, v in enumerate(steps_ms))
        result = {"micro_step": 320, "fit_s": 100.0}
        return types.SimpleNamespace(returncode=0, stdout="noise\nRESULT " + json.dumps(result) + "\n", stderr="")

    return run


def test_run_fit_builds_each_variant(lp, tmp_path, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--base_flag=1")
    calls = []
    monkeypatch.setattr(lp, "subprocess", types.SimpleNamespace(run=fake_driver(calls, [900.0, 100.0, 200.0, 600.0])))
    args = lp.parse_args(["--run-root", str(tmp_path), "--python", "py", "--compile-cache", "cache"])
    det = "--base_flag=1 --xla_gpu_deterministic_ops=true"
    res = {}
    for name in lp.FIT_VARIANTS:
        res[name] = lp.run_fit(args, pathlib.Path("cfg.yaml"), tmp_path, name)
    got = {}
    for cmd, flags in calls:
        name = pathlib.Path(cmd[cmd.index("--run-dir") + 1]).name

        def opt(key, cmd=cmd):
            return cmd[cmd.index(key) + 1] if key in cmd else None

        got[name] = {
            "flags": flags,
            "profiled": "--profile-phases" in cmd,
            "sync": "--profile-sync" in cmd,
            "prefetch": opt("--prefetch"),
            "trace": "--jax-trace" in cmd,
            "log": opt("--log-every-n-steps"),
            "ckpt": opt("--ckpt-every-n-steps"),
            "keep": opt("--ckpt-save-top-k"),
            "val": opt("--check-val-every-n-epoch") == "1",
        }
    never = str(lp.NEVER)
    plain = {"flags": "--base_flag=1", "profiled": True, "sync": False, "prefetch": None, "trace": False}
    bare = plain | {"log": never, "ckpt": never, "keep": "1", "val": False}
    assert got == {
        "smoke": plain | {"flags": det, "log": "10", "ckpt": "50", "keep": "-1", "val": True},
        "smoke_nodet": plain | {"log": "10", "ckpt": "50", "keep": "-1", "val": True},
        "prod": plain | {"log": "200", "ckpt": "1000", "keep": "1", "val": True},
        "pure": bare,
        "pure_det": bare | {"flags": det},
        "pure_log": bare | {"log": "8"},
        "pure_noprof": bare | {"log": "8", "profiled": False},
        "pure_sync": bare | {"sync": True},
        "pure_prefetch": bare | {"prefetch": "2"},
        "pure_trace": bare | {"trace": True},
    }
    # the checkpoints are deleted after each fit (their bytes recorded); the log windows are read first
    assert all(r["ckpt_bytes_deleted"] == 1000 for r in res.values())
    assert not list(tmp_path.glob("*/ckpts"))
    assert res["pure_log"]["log_windows"]["median_ms_after_first"] == 200.0
    # the overrides reach the commands: perturbation log period, smoke checkpoint / log periods
    calls.clear()
    base = ["--run-root", str(tmp_path / "o"), "--python", "py"]
    over = lp.parse_args([*base, "--pert-log-every", "3", "--smoke-ckpt-every", "7", "--smoke-log-every", "5"])
    for name in ("pure_log", "pure_noprof", "smoke", "smoke_nodet", "prod"):
        lp.run_fit(over, pathlib.Path("cfg.yaml"), tmp_path / "o", name)
    periods = {}
    for c, _ in calls:
        periods[pathlib.Path(c[c.index("--run-dir") + 1]).name] = (c[c.index("--log-every-n-steps") + 1], c[c.index("--ckpt-every-n-steps") + 1])
    never = str(lp.NEVER)
    assert periods == {"pure_log": ("3", never), "pure_noprof": ("3", never), "smoke": ("5", "7"), "smoke_nodet": ("5", "7"), "prod": ("200", "1000")}
    keep = lp.parse_args(["--run-root", str(tmp_path / "k"), "--python", "py", "--keep-ckpts"])
    assert "ckpt_bytes_deleted" not in lp.run_fit(keep, pathlib.Path("cfg.yaml"), tmp_path / "k", "pure")
    assert (tmp_path / "k/pure/ckpts/last.npz").is_file()


def test_budget_counts_from_the_job_start_and_the_longest_variant(lp, tmp_path, monkeypatch):
    now = [150.0]  # the job started at 0; the prelude took 150 s
    durations = {"smoke": 240.0, "smoke_nodet": 60.0, "pure": 60.0}

    def fake_fit(args, config, root, name):  # noqa: ARG001
        now[0] += durations[name]
        return {"result": {"micro_step": 320, "fit_s": durations[name]}}

    monkeypatch.setattr(lp, "make_config", lambda _args, root: root / "config.yaml")
    monkeypatch.setattr(lp, "run_fit", fake_fit)
    args = lp.parse_args(["--run-root", str(tmp_path), "--variants", "smoke,smoke_nodet,pure", "--budget-min", "10", "--job-start", "0"])
    lp.main(args, clock=lambda: now[0])
    result = json.loads((tmp_path / "loop_profile.json").read_text())
    # smoke ends at 390 s; 390 + 240 (the longest so far) > 600: smoke_nodet and pure do not start. Counting from
    # this script's start (0 + 240 + 240 <= 600) or ignoring the longest variant (390 <= 600) would start smoke_nodet.
    assert list(result["variants"]) == ["smoke"] and result["skipped"] == ["smoke_nodet", "pure"]
    assert lp.fits_budget(300.0, 300.0, 10) and not lp.fits_budget(300.0, 300.5, 10)


def test_budget_counts_loader_variants(lp, tmp_path, monkeypatch):
    now = [150.0]

    def fake_loader(args, config, root, workers):  # noqa: ARG001
        now[0] += 240.0
        return {"median_ms": 1.0, "first_batch_s": 10.0}

    monkeypatch.setattr(lp, "make_config", lambda _args, root: root / "config.yaml")
    monkeypatch.setattr(lp, "run_loader", fake_loader)
    monkeypatch.setattr(lp, "run_fit", lambda *_: pytest.fail("the budget should skip the fit"))
    args = lp.parse_args(["--run-root", str(tmp_path), "--variants", "loader8,pure", "--budget-min", "10", "--job-start", "0"])
    lp.main(args, clock=lambda: now[0])
    result = json.loads((tmp_path / "loop_profile.json").read_text())
    # loader8 ends at 390 s and is the longest so far: 390 + 240 > 600, so pure does not start
    assert list(result["variants"]) == ["loader8"] and result["skipped"] == ["pure"]


def test_loader_runs_without_the_determinism_flag(lp, tmp_path, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--base_flag=1")
    calls = []

    def run(cmd, env, **_):
        calls.append((cmd, env["XLA_FLAGS"]))
        out = pathlib.Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"workers": int(cmd[cmd.index("--workers") + 1]), "median_ms": 1.5}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lp, "subprocess", types.SimpleNamespace(run=run))
    args = lp.parse_args(["--run-root", str(tmp_path), "--python", "py", "--loader-batches", "5"])
    assert lp.run_loader(args, pathlib.Path("cfg.yaml"), tmp_path, 8) == {"workers": 8, "median_ms": 1.5}
    ((cmd, flags),) = calls
    assert flags == "--base_flag=1"  # the loader is host work: the run's own flags, never --det-flags
    assert cmd[cmd.index("--batches") + 1] == "5" and cmd[cmd.index("--num-train") + 1] == str(6 * 32)
