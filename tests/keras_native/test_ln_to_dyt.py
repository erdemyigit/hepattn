"""LayerNorm -> DyT warm start and float -> QAT transfer (``hepattn.experiments.clic.convert_to_native``).

Fixture: the org float Lightning run of ``test_convert.py`` (``s7_fixtures/org_float``: FastLayerNorm, hybrid_norm, the
layout of the small CLIC reference model). Gates:
- ``ln_to_dyt``: the same checkpoint converted into the LayerNorm target and into the DyT target gives the same value,
  bit for bit, for every variable of the LayerNorm target (every non-norm tensor: FastLayerNorm has none); the other
  variables of the DyT target are exactly the DyT variables, each at the torch DyT init, one DyT per torch LayerNorm.
  Control: one ulp on one ported variable is found. Without ``ln_to_dyt`` (or for an affine LayerNorm) it raises;
- torch DyT -> native DyT: the converted float forward equals the torch model with DyT norms off their init;
- ``port_native_state``: a float model's state copied into its QAT twin bit for bit, the quantizer state at the spec's
  init; a missing destination key or an unwritten destination weight raises.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hepattn.experiments.clic import convert_to_native as C
from hepattn.keras_native import checkpoint as nckpt
from hepattn.keras_native.config import load_config
from hepattn.keras_native.norm import DyT

FIX = Path(__file__).parent / "s7_fixtures"
ORG = FIX / "org_float"
HYB = FIX / "hybrid"
SENTINEL = 1e30


def with_dyt(src: Path, out: Path) -> Path:
    cfg = yaml.safe_load(src.read_text())
    init = cfg["model"]["model"]["init_args"]
    enc = init["encoder"]["init_args"] if "init_args" in init["encoder"] else init["encoder"]
    enc["norm"] = "DyT"
    dec = init["decoder"]["init_args"] if "init_args" in init["decoder"] else init["decoder"]
    dec["decoder_layer_config"]["norm"] = "DyT"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def keyed(model) -> dict[str, np.ndarray]:
    keys = nckpt.variable_keys(model)
    return {keys[id(v)]: np.asarray(v.value) for v in model.variables}


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    dyt_cfg = with_dyt(ORG / "config.yaml", tmp_path_factory.mktemp("dyt") / "config.yaml")
    _, ln_model, _ = C.convert([ORG / "config.yaml"], ORG / "model.ckpt", verify=False)
    _, dyt_model, report = C.convert([ORG / "config.yaml"], ORG / "model.ckpt", target_config=[dyt_cfg], ln_to_dyt=True)
    return {"ln": keyed(ln_model), "dyt": keyed(dyt_model), "dyt_model": dyt_model, "report": report, "dyt_cfg": dyt_cfg}


def ln_to_dyt_differences(ln: dict, dyt: dict) -> list[str]:
    """Keys of the LayerNorm target whose value is missing from, or differs in, the DyT target."""
    return [k for k, v in ln.items() if k not in dyt or dyt[k].dtype != v.dtype or not np.array_equal(dyt[k], v)]


def test_every_other_tensor_is_copied_bit_for_bit(converted):
    ln, dyt = converted["ln"], converted["dyt"]
    assert ln_to_dyt_differences(ln, dyt) == []
    extra = sorted(set(dyt) - set(ln))
    norms = dyts(converted["dyt_model"])
    assert len(extra) == 3 * len(norms) > 0
    assert sorted(converted["report"]["ln_to_dyt"]) == sorted(n.name for n in norms)
    tmodel = C.instantiate_torch_model(C.source_model_section([ORG / "config.yaml"]))
    assert len(norms) == sum(isinstance(m, torch.nn.LayerNorm) for m in tmodel.modules())
    init = {"_alpha/kernel": 0.5, "_affine/kernel": 1.0, "_affine/bias": 0.0}  # torch DyT init
    for key in extra:
        ends = [e for e in init if key.endswith(e)]
        assert len(ends) == 1, key
        assert np.all(dyt[key] == np.float32(init[ends[0]])), key
    # control: one ulp on one ported variable is found
    key = next(k for k in ln if k.endswith("q_proj/kernel"))
    bumped = dict(dyt)
    bumped[key] = np.nextafter(dyt[key], np.inf).astype(dyt[key].dtype)
    assert ln_to_dyt_differences(ln, bumped) == [key]


def test_ln_to_dyt_is_opt_in_and_non_affine_only(converted):
    with pytest.raises(C.ConversionError, match="native DyT needs a torch DyT"):
        C.convert([ORG / "config.yaml"], ORG / "model.ckpt", target_config=[converted["dyt_cfg"]])
    d = DyT(8, name="d")
    d.build((None, 4, 8))
    C._COVERAGE = C._Coverage()  # noqa: SLF001
    C._COVERAGE.ln_to_dyt = True  # noqa: SLF001
    try:
        with pytest.raises(C.ConversionError, match="non-affine LayerNorms only"):
            C.port_norm(torch.nn.LayerNorm(8), d)
        C.port_norm(torch.nn.LayerNorm(8, elementwise_affine=False), d)  # control: the non-affine one converts
    finally:
        C._COVERAGE = None  # noqa: SLF001


def test_torch_dyt_model_converts_and_matches(tmp_path):
    """Org fixture with ``norm: DyT`` in torch, DyT parameters moved off their init: the native float forward equals torch.

    Bound per output 64 eps max|x| (at least 2.2e-6). Measured worst: the regression output (|x| up to 4.7e5), 4.5 eps
    max|x| on macOS arm64 and 22.5 on xwing (Linux x86_64); 2.7 or less elsewhere. A DyT ported wrongly moves it by O(1).
    """
    run = tmp_path / "run"
    run.mkdir()
    cfg = with_dyt(ORG / "config.yaml", run / "config.yaml")
    torch.manual_seed(0)
    tmodel = C.instantiate_torch_model(C.source_model_section([cfg]))
    n_dyt = 0
    with torch.no_grad():
        for m in tmodel.modules():
            if type(m).__name__ == "DyT":
                m.alpha.fill_(0.3 + 0.01 * n_dyt)
                m.weight.normal_(1.0, 0.2)
                m.bias.normal_(0.0, 0.2)
                n_dyt += 1
    assert n_dyt > 0
    torch.save({"state_dict": {f"model.{k}": v for k, v in tmodel.state_dict().items()}, "epoch": 0, "global_step": 0}, run / "model.ckpt")
    rc, nmodel, report = C.convert([cfg], run / "model.ckpt")
    assert report["native_written"] == report["native_variables"] and report["ln_to_dyt"] == []
    inputs, _ = C.sample_batch(rc, 2)
    out = {k: np.asarray(v) for k, v in C._flatten(nmodel(inputs, training=False)).items()}  # noqa: SLF001
    eps = float(np.finfo(np.float32).eps)
    for key, diff in report["max_abs_diff"].items():
        value = out[key]
        if value.dtype == bool:
            assert diff == 0, key
            continue
        scale = float(np.abs(value[np.abs(value) < SENTINEL]).max())
        assert diff <= max(2.2e-6, 64 * eps * scale), (key, diff, scale)


def float_twin(path: Path, out: Path) -> Path:
    cfg = yaml.safe_load(path.read_text())
    cfg["model"]["model"]["init_args"].pop("quant")
    beta = ("BetaPID", "BetaScheduler", "PeriodicEBOPs")
    cfg["trainer"]["callbacks"] = [c for c in cfg["trainer"]["callbacks"] if not c.get("class_path", "").endswith(beta)]
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("twins")
    qcfg = with_dyt(HYB / "config.yaml", tmp / "qat.yaml")
    fcfg = float_twin(qcfg, tmp / "float.yaml")
    ln_cfg = float_twin(HYB / "config.yaml", tmp / "float_ln.yaml")
    rc_f, rc_q, rc_ln = load_config([fcfg]), load_config([qcfg]), load_config([ln_cfg])
    assert rc_q.quant.spec is not None and rc_f.quant.spec is None
    inputs, _ = C.sample_batch(rc_q, 2)
    models = {}
    for name, rc in (("float", rc_f), ("qat", rc_q), ("float_ln", rc_ln)):
        models[name] = rc.build_model()
        models[name](inputs, training=False)
    rng = np.random.default_rng(0)
    for v in models["float"].variables:  # every float value moved off its init
        v.assign((np.asarray(v.value) + rng.normal(0, 0.1, v.shape)).astype(v.dtype))
    return models


def state(model) -> tuple[list, list]:
    return [v.value for v in model.trainable_variables], [v.value for v in model.non_trainable_variables]


def dyts(model) -> list[DyT]:
    return [layer for layer in model._flatten_layers(include_self=False, recursive=True) if isinstance(layer, DyT)]  # noqa: SLF001


def test_float_state_into_qat_twin(twins):
    from hgq.layers import QEinsumDense, QUnaryFunctionLUT  # noqa: PLC0415

    fmodel, qmodel = twins["float"], twins["qat"]
    # every DyT of the QAT model (residual and q/k/v norms) has quantized leaves
    assert len(dyts(qmodel)) == len(dyts(fmodel)) > 0
    assert all(
        isinstance(d.alpha, QEinsumDense) and isinstance(d.tanh, QUnaryFunctionLUT) and isinstance(d.affine, QEinsumDense) for d in dyts(qmodel)
    )
    assert {d.name.rsplit("_", 1)[-1] for d in dyts(qmodel)} >= {"norm"} and any("q_norm" in d.name for d in dyts(qmodel))
    qinit = keyed(qmodel)
    report = C.port_native_state(fmodel, state(fmodel), qmodel)
    f, q = keyed(fmodel), keyed(qmodel)
    assert report["native_written"] == len(f) == report["source_variables"]
    assert all(np.array_equal(f[k], q[k]) and f[k].dtype == q[k].dtype for k in f)
    qstate = C.quantizer_state_ids(qmodel)
    keys = nckpt.variable_keys(qmodel)
    rest = [v for v in qmodel.variables if keys[id(v)] not in f]
    assert rest and all(id(v) in qstate for v in rest)
    assert all(np.array_equal(q[keys[id(v)]], qinit[keys[id(v)]]) for v in rest)  # quantizer state, beta, ebops: the init


def test_transfer_refuses_gaps(twins, monkeypatch):
    fmodel, qmodel = twins["float"], twins["qat"]
    # a QAT weight that no float variable fills: a LayerNorm float run into the DyT QAT model (the DyT variables)
    with pytest.raises(C.ConversionError, match="destination weights were not written"):
        C.port_native_state(twins["float_ln"], state(twins["float_ln"]), qmodel)
    # a float variable that the QAT model does not have
    keys = dict(nckpt.variable_keys(fmodel))
    first = fmodel.trainable_variables[0]
    keys[id(first)] += "_other"
    orig = nckpt.variable_keys
    monkeypatch.setattr(nckpt, "variable_keys", lambda m: keys if m is fmodel else orig(m))
    with pytest.raises(C.ConversionError, match="no variable of that key"):
        C.port_native_state(fmodel, state(fmodel), qmodel)
