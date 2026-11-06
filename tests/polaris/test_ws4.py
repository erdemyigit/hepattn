"""ws4 config knobs and helpers (polaris/03_make_config.py, 22_warmstart.py, 23_ws_diag.py datalane).

- ``HGQ_DATALANE_AXIS=last``: the datalane quantizers get ``heterogeneous_axis [-1]``; on the built model a QDense input
  quantizer of a (B, N, C) input has bitwidths (1, 1, C) (one per channel) instead of HGQ2's default (1, N, C). Unset:
  the old config, key for key;
- ``HGQ_NORM=DyT`` and ``HGQ_ATTN_SCALE_FRAC_BITS``: every encoder / decoder norm, and the attention kwargs of both;
- ``floatify``: the float twin (no quant spec, no beta / EBOPs callbacks) loads as a float model; ``best_arm``;
- the datalane audit on the s7 hybrid model with last-axis bits: the only quantizers with a heterogeneous token axis are
  the masked-Linformer attend cores' E-projection and p.v inputs (per key token), every QDense input is per channel.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
MAKE_CONFIG = REPO / "polaris/03_make_config.py"
SAVED = REPO / "tests/keras_native/s6_fixtures/org_normalised.yaml"
HYBRID = REPO / "tests/keras_native/s7_fixtures/hybrid/config.yaml"
SETPOINTS = {"HGQ_MODEL_EBOPS": "4.0e10", "HGQ_TARGET_EBOPS": "2.4e10", "HGQ_INIT_BETA": "1.0e-10", "HGQ_DEVICES": "1", "HGQ_PRECISION": "32"}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), REPO / "polaris" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(tmp_path, name: str, **env) -> dict:
    out = tmp_path / f"{name}.yaml"
    full = {k: v for k, v in os.environ.items() if not k.startswith("HGQ_")}
    full.update({"HGQ_CONFIG_OUT": str(out), "DATA_DIR": "/data/clic", "CKPT_DIR": str(tmp_path / "ckpts"), "HGQ_SOURCE_CONFIG": str(SAVED)})
    proc = subprocess.run([sys.executable, str(MAKE_CONFIG)], env={**full, **SETPOINTS, **env}, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return yaml.safe_load(out.read_text())


def net(cfg: dict) -> dict:
    return cfg["model"]["model"]["init_args"]


def test_knobs_write_the_ws4_model(tmp_path):
    old = generate(tmp_path, "old")
    new = generate(tmp_path, "new", HGQ_DATALANE_AXIS="last", HGQ_NORM="DyT", HGQ_ATTN_SCALE_FRAC_BITS="12")
    assert net(new)["quant"]["datalane"] == {**net(old)["quant"]["datalane"], "heterogeneous_axis": [-1], "homogeneous_axis": None}
    assert "heterogeneous_axis" not in net(old)["quant"]["datalane"]
    layer = net(new)["decoder"]["init_args"]["decoder_layer_config"]
    assert net(new)["encoder"]["norm"] == layer["norm"] == "DyT"
    assert net(new)["encoder"]["attn_kwargs"]["scale_frac_bits"] == layer["attn_kwargs"]["scale_frac_bits"] == 12
    # nothing else moved
    for cfg in (old, new):
        n = net(cfg)
        n["quant"]["datalane"] = {k: v for k, v in n["quant"]["datalane"].items() if k not in {"heterogeneous_axis", "homogeneous_axis"}}
        for part in (n["encoder"], n["decoder"]["init_args"]["decoder_layer_config"]):
            part["norm"] = None
            part["attn_kwargs"].pop("scale_frac_bits", None)
    assert old == new
    with pytest.raises(AssertionError):
        generate(tmp_path, "bad", HGQ_DATALANE_AXIS="first")


def test_last_axis_gives_one_bitwidth_per_channel(tmp_path):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from hgq.layers import QDense  # noqa: PLC0415

    from hepattn.experiments.clic.convert_to_native import sample_batch  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    shapes = {}
    for name, env in (("default", {}), ("last", {"HGQ_DATALANE_AXIS": "last"})):
        cfg = yaml.safe_load(HYBRID.read_text())
        if env:
            net(cfg)["quant"]["datalane"].update(heterogeneous_axis=[-1], homogeneous_axis=None)
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        rc = load_config([path])
        model = rc.build_model()
        inputs, _ = sample_batch(rc, 2)
        model(inputs, training=False)
        leaves = model._flatten_layers(include_self=False, recursive=True)  # noqa: SLF001
        layer = next(x for x in leaves if isinstance(x, QDense) and "encoder_l0_attn_q_proj" in x.name)
        shapes[name] = tuple(layer.iq.quantizer.kif[2].shape)
    n = shapes["default"][1]
    assert shapes["default"] == (1, n, 16) and n > 1
    assert shapes["last"] == (1, 1, 16)


def test_floatify_and_best_arm(tmp_path):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    ws = load("22_warmstart.py")
    cfg = yaml.safe_load(HYBRID.read_text())
    assert any(c["class_path"].endswith("BetaPID") for c in cfg["trainer"]["callbacks"])
    flat = ws.floatify(cfg)
    assert "quant" not in net(flat)
    assert not any(c.get("class_path", "").endswith(ws.FLOAT_DROPPED_CALLBACKS) for c in flat["trainer"]["callbacks"])
    path = tmp_path / "float.yaml"
    path.write_text(yaml.safe_dump(flat))
    rc = load_config([path])
    assert rc.quant.spec is None and rc.quant.beta_pid is None
    # best_arm: the DONE arm with the lowest last val/loss
    for arm, losses, done in (("a", [5.0, 4.2], True), ("b", [5.0, 4.1], True), ("c", [3.0], False)):
        d = tmp_path / "runs" / arm
        d.mkdir(parents=True)
        (d / "metrics.csv").write_text("step,val/loss\n" + "".join(f"{i},{v}\n" for i, v in enumerate(losses)))
        if done:
            (d / "DONE").write_text("1\n")
    assert ws.best_arm(tmp_path / "runs") == "b"
    assert ws.ARMS["ws4-ft-a"]["max"] == 1e-4 and "pid" not in ws.ARMS["ws4-ft-a"]
    assert ws.ARMS["ws4-t8"] == ws.ARMS["ws3-t8"]


def test_datalane_audit_flags_only_token_axes(tmp_path):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from hepattn.experiments.clic.convert_to_native import sample_batch  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    diag = load("23_ws_diag.py")
    cfg = yaml.safe_load(HYBRID.read_text())
    net(cfg)["quant"]["datalane"].update(heterogeneous_axis=[-1], homogeneous_axis=None)
    path = tmp_path / "last.yaml"
    path.write_text(yaml.safe_dump(cfg))
    rc = load_config([path])
    model = rc.build_model()
    inputs, _ = sample_batch(rc, 2)
    model(inputs, training=False)
    rows = diag.datalane_audit(model)
    token = {(r["leaf"].split("_attend_")[-1], r["operand"]) for r in rows if r["heterogeneous_token_axes"]}
    assert token == {("E_proj", 0), ("pv", 0)}
    assert all("_attend_" in r["leaf"] for r in rows if r["heterogeneous_token_axes"])
    dense = [r for r in rows if r["leaf_type"] == "QDense"]
    assert dense and all(r["last_axis_role"] == "channel" and r["heterogeneous_axes"] == (2,) and not r["flag"] for r in dense)
    # default axes (control): many token axes are heterogeneous
    cfg = yaml.safe_load(HYBRID.read_text())
    path.write_text(yaml.safe_dump(cfg))
    rc = load_config([path])
    model = rc.build_model()
    model(inputs, training=False)
    assert sum(bool(r["heterogeneous_token_axes"]) for r in diag.datalane_audit(model)) > len(token) * 4


def test_native_init_hash_needs_the_same_model(tmp_path):
    """``init --native-init``: the checkpoint's model hash must equal the init config's (a longer float fine-tune of a run)."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import json  # noqa: PLC0415

    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    ws = load("22_warmstart.py")
    config = tmp_path / "float.yaml"
    config.write_text(yaml.safe_dump(ws.floatify(yaml.safe_load(HYBRID.read_text()))))
    want = load_config([config]).model_hash
    ckpt = tmp_path / "step-000000024.npz"
    for meta_hash in (want, "0" * 64):
        ckpt.with_suffix(".json").write_text(json.dumps({"meta": {"config_hash": meta_hash}}))
        if meta_hash == want:
            assert ws.native_init_hash(config, ckpt) == want
        else:
            with pytest.raises(SystemExit, match="written for model"):
                ws.native_init_hash(config, ckpt)
    # the QAT twin of the same network is another model
    qat = tmp_path / "qat.yaml"
    qat.write_text(HYBRID.read_text())
    ckpt.with_suffix(".json").write_text(json.dumps({"meta": {"config_hash": want}}))
    with pytest.raises(SystemExit):
        ws.native_init_hash(qat, ckpt)
    args = ws.parse_args(["init", "--root", "r", "--tag", "t", "--source-config", "c", "--data-dir", "d", "--ckpt", "x",
                          "--model-ebops", "1", "--target-ebops", "1", "--init-beta", "1", "--native-init", str(ckpt)])  # fmt: skip
    assert args.native_init == str(ckpt)
