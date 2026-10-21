"""polaris/18_loop_profile.py arithmetic that runs without a GPU (docs/hgq/port_log/PROF.md).

- ``smoke-timing`` (hypothesis H0): run B trains only the micro-batches after its resume point, the one-time costs are
  everything outside ``fit`` plus the first (compiling) step, and the log windows come from ``steps.csv``;
- an unprofiled variant reports only its fit rate (no compile-free number, no overhead over G3), and the perturbation
  ratio compares the ``fit_s`` of ``pure`` and ``pure_noprof``; the table prints both kinds of rows.
"""

import csv
import importlib.util
import json
import pathlib

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
        w.writerows({"step": 10 * (i + 1), "time/ms_per_micro_batch": v} for i, v in enumerate([5000.0, 200.0, 210.0, 190.0]))
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
    assert a["log_windows"] == {"n": 4, "first_ms": 5000.0, "median_ms": 205.0, "mean_ms_after_first": 200.0, "max_ms_after_first": 210.0}
    assert "log_windows" not in b  # no steps.csv for run B here


def test_unprofiled_variant_and_perturbation(lp):
    noprof = lp.derive({"result": {"micro_step": 16, "fit_s": 32.0}}, 161.6)
    assert noprof["s_per_micro"] == 2.0
    assert noprof["s_per_micro_excl_compile"] is None and noprof["overhead_vs_g3_ms"] is None
    prof = {"s_per_micro_batch": 2.5, "s_per_micro_batch_excl_compile": 0.5, "groups_s": {"train/step": 4.0}, "unattributed_frac": 0.0}
    prof |= {"steady": {}, "concurrent": {}}
    profiled = lp.derive({"result": {"micro_step": 16, "fit_s": 40.0}, "profile": prof}, 161.6)
    assert profiled["overhead_vs_g3_ms"] == pytest.approx(500.0 - 161.6)
    assert profiled["groups_ms_per_micro"] == {"train/step": 250.0}
    warm = {"cache_events": {"cache_misses": 0}}
    pure = {"result": {"fit_s": 40.0, **warm}, "derived": profiled}
    result = {"variants": {"pure": pure, "pure_noprof": {"result": {"fit_s": 32.0, **warm}, "derived": noprof}}}
    result["perturbation"] = lp.perturbation(result)
    assert result["perturbation"]["ratio"] == pytest.approx(40.0 / 32.0) and result["perturbation"]["valid"]
    text = lp.table(result)
    assert "no profiler: fit_s 32.0" in text and "= 1.250" in text and "NOT VALID" not in text
    # a cold compile in one of the two runs makes the ratio meaningless
    result["variants"]["pure"]["result"]["cache_events"] = {"cache_misses": 2}
    result["perturbation"] = lp.perturbation(result)
    assert not result["perturbation"]["valid"] and "NOT VALID" in lp.table(result)
    assert lp.perturbation({"variants": {"pure": result["variants"]["pure"]}}) is None
