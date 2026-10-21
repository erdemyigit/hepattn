"""S7 gate: ``main_jax fit --init-from`` a converted Lightning checkpoint (``loop.Trainer.warm_start``).

For each source (the hybrid HGQ2 checkpoint into its own config; the org float checkpoint into the same quantized config,
a QAT warm start), the converted ``.npz`` is the start of a one-step ``main_jax fit`` on dummy events (the entry
point's ``main``, in this process, with a persistent compilation cache so the later trainers reuse the compiles).
The logged step-1 losses (task loss, total with the EBOPs penalty) must be finite and bitwise equal to a trainer
built here whose model the converter fills in memory (no file): the warm start loads exactly the converted weights.
Control: a trainer from the config's own initialisation logs another loss.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from hepattn.experiments.clic import convert_to_native as C
from hepattn.experiments.clic import main_jax
from hepattn.keras_native import loop
from hepattn.keras_native.config import load_config

FIX = Path(__file__).parent / "s7_fixtures"
ORG = FIX / "org_float"
HYB = FIX / "hybrid"
CONFIG = HYB / "config.yaml"  # the quantized target of both conversions
LOGGED = ("train/loss", "train/total_loss")
# OneCycle needs pct_start * total_steps > 1 (optim.py): 4 steps = the fixture's epoch; step 1 is the converted weights' loss
STEPS = 4


def logged(run_dir: Path) -> list[dict]:
    rows = loop.read_csv(run_dir / "steps.csv")
    assert [r["step"] for r in rows] == [float(s) for s in range(1, STEPS + 1)], rows
    return rows


def direct_run(run_dir: Path, port) -> dict:
    """One optimizer step of a trainer built as ``cli.fit`` builds it, its model filled by ``port(model)`` (or not)."""
    rc = load_config([CONFIG])
    settings = loop.LoopSettings.from_run_config(rc, run_dir, max_steps=STEPS, matmul_precision="highest")
    train = main_jax.clic_data(rc, "train", seed=settings.seed, dummy_data=True)
    trainer = loop.Trainer(rc, settings, sample=next(iter(train.epoch(0, 0))), train_batches=train.num_batches)
    if port is not None:
        port(trainer.steps.model)
        trainer.state = trainer.steps.init_state()
        trainer.beta = trainer.steps.read_beta(trainer.state)
    trainer.fit(train, None)
    return logged(run_dir)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("convert_fit")
    cache = base / "jax_cache"
    hyb_npz, org_npz = base / "hybrid.npz", base / "org_qat.npz"
    assert C.main(["--config", str(HYB / "config.yaml"), "--ckpt", str(HYB / "model.ckpt"), "--out", str(hyb_npz)]) == 0
    assert (
        C.main(["--config", str(ORG / "config.yaml"), "--ckpt", str(ORG / "model.ckpt"), "--target-config", str(CONFIG), "--out", str(org_npz)]) == 0
    )
    out = {}
    for name, npz in (("hybrid", hyb_npz), ("org", org_npz)):
        argv = ["fit", "--config", str(CONFIG), "--dummy-data", "--max-steps", str(STEPS), "--compile-cache", str(cache)]
        assert main_jax.main([*argv, "--run-dir", str(base / f"cli_{name}"), "--init-from", str(npz)]) == 0
        out[f"cli_{name}"] = logged(base / f"cli_{name}")
    hyb_state, _ = C.read_lightning_state(HYB / "model.ckpt")
    out["direct_hybrid"] = direct_run(base / "direct_hybrid", lambda m: C.port_hybrid_state(hyb_state, m))

    def port_org(m):
        tmodel = C.instantiate_torch_model(C.source_model_section([ORG / "config.yaml"]))
        state, _ = C.read_lightning_state(ORG / "model.ckpt")
        tmodel.load_state_dict(state, strict=True)
        C.port_torch_model(tmodel, m)

    out["direct_org"] = direct_run(base / "direct_org", port_org)
    out["fresh"] = direct_run(base / "fresh", None)
    out["summary_hybrid"] = (base / "cli_hybrid/summary.json").read_text()
    return out


@pytest.mark.parametrize("source", ["hybrid", "org"])
def test_warm_start_equals_a_direct_run_from_the_same_weights(runs, source):
    """Step 1 is the converted weights' loss; steps 2-4 show the optimizer state starts fresh in both."""
    for step, (cli, direct) in enumerate(zip(runs[f"cli_{source}"], runs[f"direct_{source}"], strict=True), start=1):
        for key in LOGGED:
            assert math.isfinite(cli[key]), (step, key, cli[key])
            assert cli[key] == direct[key], f"{source} step {step} {key}: --init-from {cli[key]!r} vs direct {direct[key]!r}"


@pytest.mark.parametrize("source", ["hybrid", "org"])
def test_warm_start_is_not_the_fresh_initialisation(runs, source):
    """Control: the config's own initialisation gives another step-1 loss (so the equality above is about the weights)."""
    cli, fresh = runs[f"cli_{source}"][0], runs["fresh"][0]
    for key in LOGGED:
        assert cli[key] != fresh[key], key
    assert not np.isclose(cli["train/loss"], fresh["train/loss"], rtol=1e-3), (cli["train/loss"], fresh["train/loss"])


def test_summary_records_the_warm_start(runs):
    assert '"warm_started_from": "' in runs["summary_hybrid"] and "hybrid.npz" in runs["summary_hybrid"]


def fresh_trainer(run_dir: Path, config: Path = CONFIG) -> loop.Trainer:
    rc = load_config([config])
    settings = loop.LoopSettings.from_run_config(rc, run_dir, max_steps=STEPS, matmul_precision="highest")
    train = main_jax.clic_data(rc, "train", seed=settings.seed, dummy_data=True)
    return loop.Trainer(rc, settings, sample=next(iter(train.epoch(0, 0))), train_batches=train.num_batches)


@pytest.fixture(scope="module")
def hybrid_npz(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("warm") / "hybrid.npz"
    assert C.main(["--config", str(HYB / "config.yaml"), "--ckpt", str(HYB / "model.ckpt"), "--out", str(out)]) == 0
    return out


def test_warm_start_reads_the_checkpoint_beta(hybrid_npz, tmp_path):
    """``trainer.beta`` (what ``_set_beta`` compares against) is the checkpoint's beta, not the fresh model's."""
    from hepattn.keras_native import checkpoint as nckpt  # noqa: PLC0415

    trainer = fresh_trainer(tmp_path / "run")
    fresh_beta = trainer.beta
    _, ntv, _ = nckpt.load_checkpoint(hybrid_npz, trainer.steps.model, None, expect_config_hash=trainer.rc.model_hash)[0]
    stored = trainer.steps.ebops_leaves.read_beta(ntv)
    assert stored != fresh_beta, (stored, fresh_beta)  # measured: 1e-7 (BetaPID after the fixture's epoch) vs 0
    trainer.warm_start(hybrid_npz)
    assert trainer.beta == trainer.steps.read_beta(trainer.state) == stored


def test_warm_start_needs_a_fresh_trainer(hybrid_npz, tmp_path):
    trainer = fresh_trainer(tmp_path / "run")
    trainer.global_step = 1
    with pytest.raises(ValueError, match="warm_start needs a fresh trainer"):
        trainer.warm_start(hybrid_npz)
    trainer.global_step, trainer.micro_step = 0, 1
    with pytest.raises(ValueError, match="warm_start needs a fresh trainer"):
        trainer.warm_start(hybrid_npz)
    trainer.micro_step, trainer.resumed_from = 0, tmp_path / "ckpts" / "last.npz"  # a restored trainer, steps 0
    with pytest.raises(ValueError, match="warm_start needs a fresh trainer"):
        trainer.warm_start(hybrid_npz)


def test_warm_start_refuses_another_model_config(hybrid_npz, tmp_path):
    """The target config with another loss weight (no variable changes shape) is another model: ``--init-from`` refuses
    the converted checkpoint by its config hash. Control: without the hash check the same file loads.
    """
    import yaml  # noqa: PLC0415

    from hepattn.keras_native import checkpoint as nckpt  # noqa: PLC0415

    text = CONFIG.read_text()
    assert text.count("loss_weight: 10.0") == 1
    other = tmp_path / "other.yaml"
    other.write_text(text.replace("loss_weight: 10.0", "loss_weight: 5.0"))
    assert yaml.safe_load(other.read_text())["model"] != yaml.safe_load(text)["model"]
    trainer = fresh_trainer(tmp_path / "run", other)
    assert trainer.rc.model_hash != load_config([CONFIG]).model_hash
    with pytest.raises(nckpt.CheckpointError, match="written for config"):
        trainer.warm_start(hybrid_npz)
    nckpt.load_checkpoint(hybrid_npz, trainer.steps.model, None, expect_config_hash=None)  # control: shapes match


def test_resume_and_init_from_are_exclusive(tmp_path, capsys):
    argv = ["fit", "--config", str(CONFIG), "--run-dir", str(tmp_path / "r"), "--dummy-data", "--resume", "last", "--init-from", "x.npz"]
    with pytest.raises(SystemExit) as err:
        main_jax.main(argv)
    assert err.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
