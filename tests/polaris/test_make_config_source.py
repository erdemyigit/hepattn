"""polaris/03_make_config.py with HGQ_SOURCE_CONFIG (the model of a saved float run, for a QAT warm start) and the
optimizer knobs.

The source fixture is the LightningCLI-normalised org config (``s6_fixtures/org_normalised.yaml``: base.yaml +
linformer.yaml as a run saves it), with the run-specific keys a real saved config has added (a resume ``ckpt_path``,
a GPU ``device_solver``, the logger's directory, ``default_root_dir``). The same model generated from the raw YAMLs
(the default mode) is the oracle: both configs must give the same native model hash.
"""

import math
import os
import pathlib
import subprocess
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
MAKE_CONFIG = REPO / "polaris/03_make_config.py"
SAVED = REPO / "tests/keras_native/s6_fixtures/org_normalised.yaml"
SETPOINTS = {"HGQ_MODEL_EBOPS": "4.0e10", "HGQ_TARGET_EBOPS": "2.4e10", "HGQ_INIT_BETA": "1.0e-10"}
EVERY = 8  # HGQ_EBOPS_EVERY default: the generated betas are x 8


def run(tmp_path, name: str, **env) -> subprocess.CompletedProcess:
    out = tmp_path / f"{name}.yaml"
    full = {k: v for k, v in os.environ.items() if not k.startswith("HGQ_")}
    full.update({"HGQ_CONFIG_OUT": str(out), "DATA_DIR": "/data/clic", "CKPT_DIR": str(tmp_path / "ckpts"), **env})
    return subprocess.run([sys.executable, str(MAKE_CONFIG)], env=full, capture_output=True, text=True, timeout=120)


def generate(tmp_path, name: str, **env) -> dict:
    proc = run(tmp_path, name, **env)
    assert proc.returncode == 0, proc.stderr
    return yaml.safe_load((tmp_path / f"{name}.yaml").read_text())


@pytest.fixture
def saved_run(tmp_path) -> pathlib.Path:
    cfg = yaml.safe_load(SAVED.read_text())
    cfg["ckpt_path"] = "/blue/somewhere/ckpts/epoch=036-val_loss=5.22902.ckpt"
    cfg["trainer"]["default_root_dir"] = "/blue/somewhere/logs/run"
    cfg["trainer"]["logger"]["init_args"]["offline_directory"] = "/blue/somewhere/logs/run"
    cfg["model"]["model"]["init_args"]["matcher"]["init_args"]["device_solver"] = "jv"
    cfg["model"]["lrs_config"]["cycle_momentum"] = False
    cfg["name"] = "saved_run"
    path = tmp_path / "saved_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_source_mode_needs_its_own_setpoints(tmp_path, saved_run):
    for drop in SETPOINTS:
        env = {k: v for k, v in SETPOINTS.items() if k != drop}
        proc = run(tmp_path, f"no_{drop}", HGQ_SOURCE_CONFIG=str(saved_run), **env)
        assert proc.returncode != 0 and drop in proc.stderr, (drop, proc.stderr)
    proc = run(tmp_path, "attn", HGQ_SOURCE_CONFIG=str(saved_run), HGQ_ATTN="linformer", **SETPOINTS)
    assert proc.returncode != 0 and "HGQ_ATTN" in proc.stderr


def test_saved_run_gives_the_same_native_model_as_the_raw_yamls(tmp_path, saved_run):
    """Oracle: the default mode generates base.yaml + linformer.yaml; the fixture is that model as a run saves it."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    common = {"HGQ_DEVICES": "1", "HGQ_PRECISION": "32"}
    from_raw = generate(tmp_path, "raw", **common)
    from_saved = generate(tmp_path, "saved", HGQ_SOURCE_CONFIG=str(saved_run), **SETPOINTS, **common)
    rc_raw, rc_saved = load_config([tmp_path / "raw.yaml"]), load_config([tmp_path / "saved.yaml"])
    assert rc_saved.model_hash == rc_raw.model_hash
    assert from_saved["model"]["model"]["class_path"] == "hepattn.keras.maskformer.KerasMaskFormer"
    assert from_saved["model"]["model"]["init_args"]["quant"] == from_raw["model"]["model"]["init_args"]["quant"]
    # control: a model change in the source changes the hash (the comparison above can fail)
    cfg = yaml.safe_load(saved_run.read_text())
    cfg["model"]["model"]["init_args"]["encoder"]["init_args"]["num_register_tokens"] = 4
    changed = tmp_path / "changed_source.yaml"
    changed.write_text(yaml.safe_dump(cfg))
    generate(tmp_path, "changed", HGQ_SOURCE_CONFIG=str(changed), **SETPOINTS, **common)
    assert load_config([tmp_path / "changed.yaml"]).model_hash != rc_raw.model_hash


def test_run_specific_keys_are_dropped(tmp_path, saved_run):
    cfg = generate(tmp_path, "saved", HGQ_SOURCE_CONFIG=str(saved_run), **SETPOINTS)
    assert "ckpt_path" not in cfg
    assert "default_root_dir" not in cfg["trainer"]
    assert "offline_directory" not in cfg["trainer"]["logger"]["init_args"]
    assert cfg["trainer"]["logger"]["init_args"]["online"] is False
    matcher = cfg["model"]["model"]["init_args"]["matcher"]["init_args"]
    assert "device_solver" not in matcher and matcher["default_solver"] == "scipy"
    assert cfg["name"] == "saved_run_HGQ_polaris"
    assert cfg["data"]["train_path"] == "/data/clic/train_clic_fix.root"
    # the source's optimizer settings other than the overridden ones are kept (Lion source, AdamW default)
    assert cfg["model"]["optimizer"] == "AdamW" and cfg["model"]["lrs_config"]["max"] == pytest.approx(5e-5)
    assert cfg["model"]["lrs_config"]["weight_decay"] == pytest.approx(1e-4)


def test_source_setpoints_and_the_headroom_rule(tmp_path, saved_run):
    pid = next(
        c
        for c in generate(tmp_path, "s", HGQ_SOURCE_CONFIG=str(saved_run), **SETPOINTS)["trainer"]["callbacks"]
        if c["class_path"].endswith("BetaPID")
    )["init_args"]
    init, ceiling, floor = pid["init_beta"] / EVERY, pid["max_beta"] / EVERY, pid["min_beta"] / EVERY
    assert (pid["target_ebops"], init) == pytest.approx((2.4e10, 1.0e-10), rel=1e-12)
    decades = math.log10(1.0e-7 / 2.6e-10)  # the same rule as the default setpoints (test_beta_setpoints.py)
    assert math.log10(ceiling / init) == pytest.approx(decades, abs=1e-9)
    assert math.log10(init / floor) == pytest.approx(decades, abs=1e-9)
    pid = next(
        c
        for c in generate(tmp_path, "o", HGQ_SOURCE_CONFIG=str(saved_run), HGQ_MAX_BETA="5e-9", **SETPOINTS)["trainer"]["callbacks"]
        if c["class_path"].endswith("BetaPID")
    )["init_args"]
    assert pid["max_beta"] == pytest.approx(5e-9 * EVERY, rel=1e-12)


def test_optimizer_knobs(tmp_path):
    cfg = generate(tmp_path, "lion", HGQ_OPTIMIZER="Lion", HGQ_LR_MAX="6.2e-4", HGQ_WEIGHT_DECAY="0.1")
    assert (cfg["model"]["optimizer"], cfg["model"]["lrs_config"]["max"], cfg["model"]["lrs_config"]["weight_decay"]) == ("Lion", 6.2e-4, 0.1)
    default = generate(tmp_path, "default")
    # base.yaml's `weight_decay: 1e-4` is a YAML 1.1 string; the native and torch loaders apply float()
    assert (default["model"]["optimizer"], default["model"]["lrs_config"]["max"], float(default["model"]["lrs_config"]["weight_decay"])) == (
        "AdamW",
        5e-5,
        1e-4,
    )
    proc = run(tmp_path, "bad", HGQ_OPTIMIZER="SGD")
    assert proc.returncode != 0 and "HGQ_OPTIMIZER" in proc.stderr


def test_weight_b0_knob(tmp_path):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    default, twelve = generate(tmp_path, "b8"), generate(tmp_path, "b12", HGQ_WEIGHT_B0="12")
    assert default["model"]["model"]["init_args"]["quant"]["weight"] == {"default_q_type": "kbi", "b0": 8, "i0": 2}
    assert twelve["model"]["model"]["init_args"]["quant"]["weight"]["b0"] == 12
    assert load_config([tmp_path / "b8.yaml"]).model_hash != load_config([tmp_path / "b12.yaml"]).model_hash


def test_datalane_overflow_knob(tmp_path):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    default, sat = generate(tmp_path, "wrap"), generate(tmp_path, "sat", HGQ_DATALANE_OVERFLOW="sat")
    assert default["model"]["model"]["init_args"]["quant"]["datalane"] == {"default_q_type": "kif", "i0": 4, "f0": 8}
    assert sat["model"]["model"]["init_args"]["quant"]["datalane"]["overflow_mode"] == "SAT"
    rc_wrap, rc_sat = load_config([tmp_path / "wrap.yaml"]), load_config([tmp_path / "sat.yaml"])
    assert rc_wrap.model_hash != rc_sat.model_hash
    proc = run(tmp_path, "bad", HGQ_DATALANE_OVERFLOW="CLAMP")
    assert proc.returncode != 0 and "HGQ_DATALANE_OVERFLOW" in proc.stderr
