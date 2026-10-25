"""Gate G4 scripts without a GPU (polaris/19_g4_compare.py, polaris/19_g4_hybrid.py).

- the verdict band: its coverage formula (also by Monte Carlo), the strict [min, max] result, the degenerate case;
- ``verdict``: PASS / FAIL / INVALID (LR or forward precondition) / INCOMPLETE / INCONCLUSIVE, the EBOPs gate (inside the
  band, or no farther from the setpoint than the farther hybrid), a resumed run's rewritten rows (the later row wins);
- the G4 config: K / epoch length / checkpoint interval / BetaPID warm-up written in, the production LR, clip,
  accumulation and fp32 kept; hybrid checkpoints never on an epoch's last step;
- the hybrid data order: ``epoch_order`` equals the native ``EpochSampler`` (JAX env only: ``native_data`` imports
  ``keras_native``), and the trainer position (``data_position``) is the next micro-batch of the epoch.
"""

import importlib.util
import math
import pathlib
import sys
import types

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def g4():
    return load("g4_compare", "polaris/19_g4_compare.py")


@pytest.fixture(scope="module")
def hyb():
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    sys.modules.setdefault("comet_ml", types.ModuleType("comet_ml"))  # the JAX env has no comet_ml; only imported
    return load("g4_hybrid", "polaris/19_g4_hybrid.py")


# --------------------------------------------------------------------------------------------- band
def test_coverage_formula_and_monte_carlo(g4):
    assert g4.coverage(0.5) == pytest.approx(1 / 3, abs=1e-12)  # the literal [min, max] rule
    assert g4.coverage(1.5) == pytest.approx(2 / 3, abs=1e-12)  # [min - D, max + D]
    assert g4.coverage(g4.C_DEFAULT) == pytest.approx(0.90, abs=0.001)
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 400_000))
    m, d = 0.5 * (x[0] + x[1]), np.abs(x[0] - x[1])
    for c in (0.5, 1.5, g4.C_DEFAULT):
        assert np.mean(np.abs(x[2] - m) <= c * d) == pytest.approx(g4.coverage(c), abs=0.003), c


def test_band_strict_inside_and_degenerate(g4):
    b = g4.band(10.0, 9.0, 11.0, 1.5)  # inside [min, max]
    assert b["strict"] and b["inside"] and b["mean"] == 10.0 and b["spread"] == 2.0 and b["band"] == [7.0, 13.0]
    b = g4.band(12.5, 9.0, 11.0, 1.5)  # outside [min, max], inside the band
    assert not b["strict"] and b["inside"] and b["z"] == pytest.approx(1.25)
    assert not g4.band(13.5, 9.0, 11.0, 1.5)["inside"]
    assert g4.band(13.0, 9.0, 11.0, 1.5)["inside"]  # the edge counts
    b = g4.band(10.0, 10.0, 10.0, 5.0)
    assert b["degenerate"] and not b["inside"]


# --------------------------------------------------------------------------------------------- verdict
K, ACCUM, TARGET = 100, 4, 1.5e11


def series(val: dict[int, float], *, ebops: float = 1.6e11, lr_scale: float = 1.0, ms: float = 100.0) -> dict:
    metrics = {s: {"step": s, "val/loss": v, "val/ebops": ebops, "train/quant_beta": 1e-10} for s, v in val.items()}
    steps = {
        s: {"step": s, "train/loss": 20.0 - s / 50, "train/lr": lr_scale * 1e-5 * (1 + s), "time/ms_per_micro_batch": ms}
        for s in range(10, K + 1, 10)
    }
    segs = [{"start_step": 0, "end_step": K, "wall_s": ms / 1e3 * K * ACCUM}]
    return {"metrics": metrics, "steps": steps, "segments": segs}


# float-mode forward check, measured at production size on the CPU (G4.md "Forward check"): hybrid vs native
FLOAT_MEASURED = {
    "final.mask.pflow_node_logit": {"max_abs": 4.292e-05, "ref_max_abs": 18.94},
    "final.regression.pflow_proxy_regr": {"max_abs": 4.883e-04, "ref_max_abs": 2480.0},
    "layer_0.attn_mask": {"flips": 0, "n": 768000},
}
# controls, measured the same way: one decoder-layer-3 kernel x 1.01 (worst output 1.1e-2 of its scale) and one
# encoder kernel transposed (20730 layer-0 mask flips, outputs at 100% of their scale)
FLOAT_SCALED_KERNEL = {**FLOAT_MEASURED, "final.regression.pflow_proxy_regr": {"max_abs": 0.0112 * 2480.0, "ref_max_abs": 2480.0}}
FLOAT_FLIPS = {**FLOAT_MEASURED, "layer_0.attn_mask": {"flips": 20730, "n": 768000}}
GOOD_FORWARD = {"platforms": {"gpu": {"float": FLOAT_MEASURED}, "cpu": {"float": FLOAT_MEASURED}}}


def arms(jax_val: float = 15.05, *, a: float = 15.0, b: float = 15.2, **jax_kwargs) -> dict:
    return {
        "hybrid-a": series({50: 16.0, K: a}, ms=1400.0),
        "hybrid-b": series({50: 16.1, K: b}, ms=1400.0),
        "jax": series({50: 16.05, K: jax_val}, ms=160.0, **jax_kwargs),
    }


def test_verdict_pass_and_fail(g4):
    v = g4.verdict(arms(15.05), k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)
    assert v["status"] == "PASS" and v["strict_min_max_at_k"]
    assert v["preconditions"] == {"lr_steps_compared": 10, "lr_max_rel_diff": 0.0, "lr_ok": True, "forward_ok": True}
    assert v["speedup_median_window"] == pytest.approx(1400 / 160)
    assert [t["step"] for t in v["val_loss"]["trajectory"]] == [50, K]
    # outside [min, max] but inside mean +- 5.47 |D| = 15.1 +- 1.094: PASS with the default band, FAIL with the strict one
    v = g4.verdict(arms(16.0), k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)
    assert v["status"] == "PASS" and not v["strict_min_max_at_k"]
    assert g4.verdict(arms(16.0), k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD, c=0.5)["status"] == "FAIL"
    assert g4.verdict(arms(16.3), k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)["status"] == "FAIL"


def test_verdict_preconditions(g4):
    v = g4.verdict(arms(15.05, lr_scale=1.001), k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)
    assert v["status"] == "INVALID" and not v["preconditions"]["lr_ok"]
    assert v["preconditions"]["lr_max_rel_diff"] == pytest.approx(1e-3)
    bad = {"platforms": {"gpu": {"float": FLOAT_MEASURED}, "cpu": {"float": FLOAT_SCALED_KERNEL}}}  # one platform fails
    assert g4.verdict(arms(15.05), k=K, accum=ACCUM, target=TARGET, forward=bad)["status"] == "INVALID"
    old_format = {"errors": FLOAT_MEASURED, "ok": True}  # the quantized-only check of init f6930a6: re-run the forward check
    assert g4.verdict(arms(15.05), k=K, accum=ACCUM, target=TARGET, forward=old_format)["status"] == "INVALID"
    assert g4.verdict(arms(15.05), k=K, accum=ACCUM, target=TARGET, forward=None)["status"] == "INVALID"


def test_float_gate(g4):
    assert g4.float_ok(FLOAT_MEASURED)
    assert not g4.float_ok(FLOAT_SCALED_KERNEL)
    assert not g4.float_ok(FLOAT_FLIPS)
    assert not g4.float_ok({"layer_0.attn_mask": {"flips": 0, "n": 768000}})  # nothing float compared
    assert g4.forward_ok(GOOD_FORWARD) and not g4.forward_ok(None) and not g4.forward_ok({"platforms": {}})


def test_forward_errors(g4):
    np = pytest.importorskip("numpy")
    ref = {"m": np.array([1.0, -2.0, 1e31, 4.0], np.float32), "b": np.array([True, False, True])}
    got = {"m": np.array([1.5, -2.0, -5.0, 3.0], np.float32), "b": np.array([True, True, True])}
    e = g4.forward_errors(ref, got)
    assert e["m"] == {"max_abs": 1.0, "q50_abs": 0.5, "ref_max_abs": 4.0, "frac_gt_1pct": pytest.approx(2 / 3)}  # 1e31 padding slot left out
    assert e["b"] == {"flips": 1, "n": 3}
    with pytest.raises(KeyError, match="output keys differ"):
        g4.forward_errors(ref, {"m": got["m"]})


def test_verdict_incomplete_and_inconclusive(g4):
    a = arms(15.05)
    del a["jax"]["metrics"][K]
    v = g4.verdict(a, k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)
    assert v["status"] == "INCOMPLETE" and v["missing"] == ["jax"]
    a = arms(15.05)
    del a["hybrid-b"]
    assert g4.verdict(a, k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)["missing"] == ["hybrid-b"]
    assert g4.verdict(arms(15.0, a=15.0, b=15.0), k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)["status"] == "INCONCLUSIVE"


def test_verdict_ebops_gate(g4):
    def with_ebops(jax_e: float, a_e: float = 1.60e11, b_e: float = 1.61e11) -> dict:  # band 0.0294 +- 0.0148 in log10
        a = arms(15.05)
        for name, e in (("hybrid-a", a_e), ("hybrid-b", b_e), ("jax", jax_e)):
            for row in a[name]["metrics"].values():
                row["val/ebops"] = e
        return g4.verdict(a, k=K, accum=ACCUM, target=TARGET, forward=GOOD_FORWARD)

    inside = with_ebops(1.605e11)
    assert inside["status"] == "PASS" and inside["ebops"]["at_k"]["inside"]
    assert inside["ebops"]["at_k"]["value"] == pytest.approx(math.log10(1.605e11 / TARGET))
    # outside the narrow band but closer to the setpoint than the farther hybrid: tracking is not worse -> PASS
    closer = with_ebops(1.52e11)
    assert not closer["ebops"]["at_k"]["inside"] and closer["ebops"]["no_farther_than_worst_hybrid"] and closer["status"] == "PASS"
    far = with_ebops(3.0e11)  # far above both and outside the band
    assert not far["ebops"]["ok"] and far["status"] == "FAIL"
    below = with_ebops(0.5e11)  # overshoots below the setpoint by more than the hybrids are above it
    assert not below["ebops"]["ok"]
    assert g4.verdict(arms(15.05), k=K, accum=ACCUM, target=None, forward=GOOD_FORWARD)["status"] == "FAIL"
    # hybrids with the same EBOPs (discrete bit steps): the floor keeps a 0.01 dex band; JAX 0.0054 dex away and farther
    # from the setpoint passes only through the floor, 0.02 dex away fails
    floor = with_ebops(1.62e11, 1.6e11, 1.6e11)
    assert floor["ebops"]["at_k"]["inside"] and not floor["ebops"]["no_farther_than_worst_hybrid"] and floor["status"] == "PASS"
    assert not with_ebops(1.675e11, 1.6e11, 1.6e11)["ebops"]["ok"]
    assert g4.band(1.0, 1.0, 1.0, 5.47)["degenerate"] and not g4.band(1.0, 1.0, 1.0, 5.47, floor=0.01)["degenerate"]


def test_resumed_rows_later_wins(g4, tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,step,val/loss\n0,500,20.0\n1,1000,18.0\n1,1000,17.5\n2,1500,\n")
    rows = g4.by_step(g4.read_rows(path))
    assert rows[1000]["val/loss"] == 17.5 and math.isnan(rows[1500]["val/loss"]) and sorted(rows) == [500, 1000, 1500]
    assert g4.read_rows(tmp_path / "missing.csv") == []


def test_throughput_per_segment(g4):
    t = g4.throughput(
        {
            "steps": {10: {"time/ms_per_micro_batch": 150.0}, 20: {"time/ms_per_micro_batch": 9000.0}, 30: {"time/ms_per_micro_batch": 170.0}},
            "segments": [{"start_step": 0, "end_step": 30, "wall_s": 24.0}, {"start_step": 30, "end_step": 30, "wall_s": 5.0}],
        },
        4,
    )
    assert t["median_ms_per_micro_batch"] == 170.0 and t["wall_s_total"] == 29.0
    assert [s["s_per_micro_batch"] for s in t["segments"]] == [pytest.approx(24.0 / 120)]


# --------------------------------------------------------------------------------------------- config
def test_checkpoint_steps_never_on_an_epoch_end(g4):
    assert g4.check_ckpt_steps(5000, 500, 96) == []
    assert g4.check_ckpt_steps(5000, 500, 100) == [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    assert g4.check_ckpt_steps(1000, 500, 1000) == [1000]  # K itself counts
    args = g4.parse_args(["init", "--root", "/r", "--tag", "t", "--ckpt-every", "250"])
    with pytest.raises(SystemExit, match="epoch ends"):
        g4.settings_from(args)
    s = g4.settings_from(g4.parse_args(["init", "--root", "/r", "--tag", "t"]))
    assert (s["k"], s["epoch_steps"], s["ckpt_every"], s["jax_ckpt_every"], s["num_train"], s["seed_a"] != s["seed_b"]) == (
        5000,
        500,
        96,
        200,
        -1,
        True,
    )


def test_g4_config(g4, tmp_path):
    s = g4.settings_from(g4.parse_args(["init", "--root", str(tmp_path), "--tag", "t"]))
    cfg = yaml.safe_load(g4.make_config(s, tmp_path).read_text())
    t, d = cfg["trainer"], cfg["data"]
    assert (t["max_epochs"], t["limit_train_batches"], t["log_every_n_steps"], t["check_val_every_n_epoch"]) == (10, 2000, 10, 1)
    assert t["max_epochs"] * t["limit_train_batches"] // t["accumulate_grad_batches"] == 5000  # K
    # the production settings G4 must keep (the LR / clip accumulation fix of 2026-10-14, fp32, one device)
    assert (t["accumulate_grad_batches"], t["gradient_clip_val"], str(t["precision"]), t["devices"]) == (4, 0.1, "32", 1)
    assert cfg["model"]["lrs_config"]["max"] == 5e-5 and d["batch_size"] == 32
    assert (d["num_train"], d["num_val"], d["num_workers"]) == (-1, 5000, 4)
    cbs = {c["class_path"].rsplit(".", 1)[-1]: c.get("init_args", {}) for c in t["callbacks"]}
    assert cbs["ModelCheckpoint"]["every_n_train_steps"] == 96 and cbs["ModelCheckpoint"]["save_top_k"] == 1
    assert cbs["BetaPID"]["warmup_epochs"] == 2 and g4.target_ebops(tmp_path / "config.yaml") == cbs["BetaPID"]["target_ebops"]
    assert "PeriodicEBOPs" in cbs and "MetricsCsvWriter" in cbs


def test_arm_hybrid_config_paths_and_seed(g4, tmp_path):
    s = g4.settings_from(g4.parse_args(["init", "--root", str(tmp_path), "--tag", "t"]))
    config = g4.make_config(s, tmp_path)
    arm = tmp_path / "hybrid-b"
    arm.mkdir()
    cfg = yaml.safe_load(g4.arm_hybrid_config({"config": str(config)}, arm, 7).read_text())
    cbs = {c["class_path"].rsplit(".", 1)[-1]: c.get("init_args", {}) for c in cfg["trainer"]["callbacks"]}
    assert cbs["ModelCheckpoint"]["dirpath"] == str(arm / "ckpts") and cbs["MetricsCsvWriter"]["path"] == str(arm / "metrics.csv")
    assert cfg["data"]["data_seed"] == 7
    assert "data_seed" not in yaml.safe_load(config.read_text())["data"]  # the JAX config stays the plain one


# --------------------------------------------------------------------------------------------- data order
def test_data_position(hyb):
    assert hyb.data_position(0, 0, 4, 2000) == 0
    assert hyb.data_position(96, 0, 4, 2000) == 384
    assert hyb.data_position(500, 0, 4, 2000) == 2000  # the epoch's last step: its end hooks are due
    assert hyb.data_position(500, 1, 4, 2000) == 0
    assert hyb.data_position(576, 1, 4, 2000) == 304
    with pytest.raises(ValueError, match="not in epoch"):
        hyb.data_position(96, 1, 4, 2000)


def test_hybrid_order_is_the_native_order(hyb):
    native_data = pytest.importorskip("hepattn.experiments.clic.native_data", reason="native_data needs the JAX env")
    for seed, epoch, start, n, b in ((42, 0, 0, 1000, 4), (42, 3, 7, 1000, 4), (1_000_042, 1, 0, 997, 32), (5, 2, 30, 997, 32)):
        native = list(native_data.EpochSampler(n, batch_size=b, shuffle=True, seed=seed, epoch=epoch, start_batch=start, drop_last=True))
        assert hyb.epoch_order(n, seed, epoch, b, start) == native, (seed, epoch, start)
    assert hyb.epoch_order(1000, 42, 0, 4) != hyb.epoch_order(1000, 1_000_042, 0, 4)


def test_trainer_sampler_reads_the_position(hyb):
    pos = [(0, 0)]
    sampler = hyb.TrainerEpochSampler(1000, batch_size=4, seed=42, position=lambda: pos[0])
    sampler.set_epoch(5)  # Lightning's call: ignored
    assert list(sampler) == hyb.epoch_order(1000, 42, 0, 4)
    pos[0] = (2, 10)
    assert list(sampler) == hyb.epoch_order(1000, 42, 2, 4, 10) and len(sampler) == 1000 - 40
    assert sampler.history == [(0, 0), (2, 10)]


def test_run_refuses_an_init_whose_forward_check_fails(g4, tmp_path):
    tag = tmp_path / "t"
    tag.mkdir()
    manifest = {
        "complete": True,
        "settings": {"k": 10, "tiny": True},
        "seeds": {"a": 1, "b": 2},
        "forward_check": {"errors": FLOAT_MEASURED, "ok": False},
    }
    (tag / "manifest.json").write_text(__import__("json").dumps(manifest))
    with pytest.raises(SystemExit, match="forward check does not pass"):
        g4.main(["run", "--root", str(tmp_path), "--tag", "t", "--arm", "jax"])
    manifest["forward_check"] = {"platforms": {"cpu": {"float": FLOAT_SCALED_KERNEL}}}
    (tag / "manifest.json").write_text(__import__("json").dumps(manifest))
    with pytest.raises(SystemExit, match="forward check does not pass"):
        g4.main(["run", "--root", str(tmp_path), "--tag", "t", "--arm", "hybrid-a"])
