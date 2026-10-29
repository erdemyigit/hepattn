"""polaris/22_warmstart.py arm configs (ws2): the arm's optimizer keys, BetaPID arguments and PeriodicEBOPs cadence are
written over the init's config, other BetaPID arguments are kept, and a ``"model": "sat"`` arm trains the init's SAT
model (its config and init), which an init without ``--sat-cal-batches`` does not have.
"""

import importlib.util
import pathlib

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ws():
    spec = importlib.util.spec_from_file_location("ws22", REPO / "polaris/22_warmstart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path, name: str, overflow: str | None) -> pathlib.Path:
    datalane = {"default_q_type": "kif", "i0": 4, "f0": 8, **({"overflow_mode": overflow} if overflow else {})}
    cfg = {
        "model": {
            "optimizer": "Lion",
            "lrs_config": {"initial": 1e-6, "max": 6.2e-4, "end": 1e-6, "pct_start": 0.05, "weight_decay": 0.1},
            "model": {"init_args": {"quant": {"datalane": datalane}}},
        },
        "trainer": {
            "callbacks": [
                {
                    "class_path": "hepattn.keras.callbacks.BetaPID",
                    "init_args": {"target_ebops": 2.9e10, "init_beta": 4.8e-10, "i": 0.5, "max_beta": 1.8e-7},
                },
                {"class_path": "hepattn.keras.callbacks.PeriodicEBOPs", "init_args": {"every_n_steps": 8}},
            ]
        },
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def manifest(tmp_path):
    return {
        "config": str(config(tmp_path, "wrap", None)),
        "native_init": str(tmp_path / "init.npz"),
        "sat": {"config": str(config(tmp_path, "sat", "SAT")), "native_init": str(tmp_path / "init_sat.npz"), "ok": True},
    }


def written(ws, manifest, arm, tmp_path) -> dict:
    out = tmp_path / arm
    out.mkdir()
    return yaml.safe_load(ws.arm_config(manifest, arm, out).read_text())


def every(cfg) -> int:
    return next(c for c in cfg["trainer"]["callbacks"] if c["class_path"].endswith("PeriodicEBOPs"))["init_args"]["every_n_steps"]


def pid_args(cfg) -> dict:
    return next(c for c in cfg["trainer"]["callbacks"] if c["class_path"].endswith("BetaPID"))["init_args"]


def test_ws2_arms_write_the_optimizer_and_pid_settings(ws, manifest, tmp_path):
    a, c = written(ws, manifest, "ws2-a", tmp_path), written(ws, manifest, "ws2-c", tmp_path)
    for cfg in (a, c):
        lrs = cfg["model"]["lrs_config"]
        assert cfg["model"]["optimizer"] == "AdamW"
        assert (lrs["max"], lrs["weight_decay"], lrs["quantizer_lr_scale"], lrs["quantizer_epsilon"]) == (5e-5, 1e-4, 20.0, 1e-12)
        assert "overflow_mode" not in cfg["model"]["model"]["init_args"]["quant"]["datalane"]
    assert pid_args(a) == {"target_ebops": 2.9e10, "init_beta": 8.9e-10, "i": 0.5, "max_beta": 1.8e-7}
    assert pid_args(c) == {"target_ebops": 2.9e10, "init_beta": 8.9e-10, "i": 2.0, "max_beta": 1.8e-7}
    assert pid_args(written(ws, manifest, "ws2-b", tmp_path))["init_beta"] == 2.7e-9
    # every step: the same time-averaged pull as ws2-a (logged beta / every)
    e1 = written(ws, manifest, "ws2-e1", tmp_path)
    assert every(e1) == 1 and every(a) == 8
    assert pid_args(e1)["init_beta"] / every(e1) == pytest.approx(pid_args(a)["init_beta"] / every(a), rel=1e-12)
    # the ws1 arms are unchanged: no epsilon key, the init's BetaPID
    q20 = written(ws, manifest, "adamw-q20", tmp_path)
    assert "quantizer_epsilon" not in q20["model"]["lrs_config"] and pid_args(q20)["init_beta"] == 4.8e-10


def test_the_sat_arm_trains_the_sat_model(ws, manifest, tmp_path):
    sat = written(ws, manifest, "ws2-sat", tmp_path)
    assert sat["model"]["model"]["init_args"]["quant"]["datalane"]["overflow_mode"] == "SAT"
    assert pid_args(sat)["init_beta"] == 8.9e-10
    assert ws.arm_model(manifest, "ws2-sat")["native_init"].endswith("init_sat.npz")
    assert ws.arm_model(manifest, "ws2-a")["native_init"].endswith("/init.npz")
    del manifest["sat"]
    with pytest.raises(SystemExit, match="sat-cal-batches"):
        ws.arm_model(manifest, "ws2-sat")
