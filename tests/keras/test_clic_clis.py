"""S7 item 3: the hybrid CLIs ``port_to_keras`` and ``evaluate_hgq`` on the org model (R4b / S56 open issues).

Inputs are the S7 fixtures (``tests/keras_native/s7_fixtures``: real Lightning runs of ``main.py`` on org
``base.yaml`` + ``linformer.yaml`` and of ``main_hgq.py`` on the generated Polaris config, at test sizes).

- ``port_to_keras.load_torch_model`` keeps the linformer attentions (control: ``change_attn_backends(model, "torch")``,
  what it called before, raises for them);
- ``port_to_keras.main`` writes a complete state: every keras weight (the E / F projection of every masked-Linformer
  core, every Dense kernel), which a fresh hybrid loads strictly and which reproduces the torch model from the
  checkpoint (``test_org_model_parity`` linformer output tolerance). Control: without ``materialize`` (what the CLI
  did before) the unbuilt linformer cores refuse the save;
- ``evaluate_hgq.main`` runs on the hybrid Lightning checkpoint with the converted org float state as the float
  baseline (built without the quant spec: ``build_keras_maskformer_from_config(cfg, quant=None)`` keeps the spec of a
  ``KerasMaskFormer`` config, so the old baseline was a second quantized model, refused by the strict load); the
  quantized model it evaluates reproduces the hybrid's stored outputs bit for bit. Control: loading without
  registering the keras weights (what it did before) is refused;
- ``eval_keras.main`` (the same defect as ``evaluate_hgq``: strict load into an unregistered model) runs on the hybrid
  Lightning checkpoint, and the model it evaluates reproduces the stored outputs bit for bit. Control: its old load
  (forward, then ``load_state_dict`` without ``register_keras_parameters``) is refused;
- ``s7_fixtures/hybrid/einsum_inputs.json`` (the hybrid's ``QEinsum`` input order, which the native converter test
  checks its quantizer pairing against) is what the hybrid model built from the fixture holds;
- ``port_to_keras.read_model_state`` (the loader of all three CLIs) refuses a checkpoint carrying pickled code.
  Control: an unrestricted ``torch.load`` of the same file runs that code.
"""

import json
from pathlib import Path
from pickle import UnpicklingError  # noqa: S403 (the exception type only)

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from test_maskformer_parity import flatten_outputs  # ty: ignore [unresolved-import]
from test_org_model_parity import TOLS, scaled_err  # ty: ignore [unresolved-import]

from hepattn.experiments.clic import eval_keras, evaluate_hgq, port_to_keras
from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.models.attention import Attention
from hepattn.models.encoder import change_attn_backends

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests/keras_native/s7_fixtures"
ORG, HYB = FIX / "org_float", FIX / "hybrid"
OUT_TOL = TOLS["linformer"][0]


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO)  # the fixture configs hold repo-relative scale-dict paths, as a run directory's config does


def model_cfg(run: Path) -> dict:
    return yaml.safe_load((run / "config.yaml").read_text())["model"]["model"]


def data_cfg(run: Path) -> dict:
    return yaml.safe_load((run / "config.yaml").read_text())["data"]


def test_load_torch_model_keeps_the_linformer_attention():
    model = port_to_keras.load_torch_model(model_cfg(ORG), ORG / "model.ckpt")
    kinds = [m.attn_type for m in model.modules() if isinstance(m, Attention)]
    assert kinds and set(kinds) == {"linformer"}, kinds
    control = port_to_keras.instantiate_model_section(model_cfg(ORG))
    with pytest.raises(ValueError, match="linformer owns trained projection parameters"):
        change_attn_backends(control, "torch")


@pytest.fixture(scope="module")
def ported(tmp_path_factory):
    out = tmp_path_factory.mktemp("port") / "keras_state.pt"
    port_to_keras.main(["--config", str(ORG / "config.yaml"), "--ckpt", str(ORG / "model.ckpt"), "--out", str(out)])
    return out


def kernel_keys(state: dict) -> set[str]:
    return {k.split("._torch_params.", 1)[1] for k in state if "._torch_params." in k and k.endswith("/kernel")}


def test_port_to_keras_state_is_complete_and_reproduces_torch(ported, monkeypatch):
    monkeypatch.chdir(REPO)
    state = torch.load(ported, weights_only=True)
    tmodel = port_to_keras.load_torch_model(model_cfg(ORG), ORG / "model.ckpt")
    n_linformer = sum(isinstance(m, Attention) and m.attn_type == "linformer" for m in tmodel.modules())
    kernels = kernel_keys(state)
    assert sum(k.endswith(("_E_proj/kernel", "_F_proj/kernel")) for k in kernels) == 2 * n_linformer, sorted(kernels)
    inputs = port_to_keras.dummy_inputs(data_cfg(ORG), num_events=2)
    kmodel = port_to_keras.load_keras_state(port_to_keras.build_keras_maskformer_from_config(model_cfg(ORG)).eval(), ported, inputs)
    with torch.no_grad():
        flat_t, flat_k = flatten_outputs(tmodel(inputs)), flatten_outputs(kmodel(inputs))
    assert set(flat_t) == set(flat_k)
    for key, ref in flat_t.items():
        if ref.dtype == torch.bool:
            assert torch.equal(ref, flat_k[key]), key
        else:
            assert scaled_err(ref, flat_k[key]) <= OUT_TOL, (key, scaled_err(ref, flat_k[key]))


def test_control_port_to_keras_without_materialize(tmp_path, monkeypatch):
    """What the CLI did before (``state_dict()`` right after the port): the unbuilt linformer cores refuse the save."""
    monkeypatch.setattr(KerasMaskFormer, "materialize", lambda self, _inputs: self)
    with pytest.raises(RuntimeError, match="linformer"):
        port_to_keras.main(["--config", str(ORG / "config.yaml"), "--ckpt", str(ORG / "model.ckpt"), "--out", str(tmp_path / "s.pt")])


def stored_hybrid() -> tuple[dict, dict]:
    stored = dict(np.load(HYB / "outputs.npz"))
    inputs = {k.removeprefix("in::"): torch.from_numpy(v) for k, v in stored.items() if k.startswith("in::")}
    outputs = {k.removeprefix("out::"): v for k, v in stored.items() if k.startswith("out::")}
    return inputs, outputs


def test_evaluate_hgq_loads_the_hybrid_checkpoint_bit_for_bit():
    cfg = model_cfg(HYB)
    inputs, ref = stored_hybrid()
    model = port_to_keras.build_keras_maskformer_from_config(cfg, quant=cfg["init_args"]["quant"]).eval()
    model = port_to_keras.load_keras_state(model, HYB / "model.ckpt", inputs)
    with torch.no_grad():
        got = flatten_outputs(model(inputs))
    assert set(got) == set(ref)
    for key, value in ref.items():
        assert np.array_equal(got[key].numpy(), value), key


def test_control_unregistered_load_is_refused():
    cfg = model_cfg(HYB)
    inputs, _ = stored_hybrid()
    model = port_to_keras.build_keras_maskformer_from_config(cfg, quant=cfg["init_args"]["quant"]).eval()
    model.materialize(inputs)  # what evaluate_hgq did before: no register_keras_parameters
    with pytest.raises(RuntimeError, match="Unexpected key"):
        evaluate_hgq.load_state(model, HYB / "model.ckpt")


def test_evaluate_hgq_main_with_a_float_baseline(ported, tmp_path):
    out = tmp_path / "report.md"
    evaluate_hgq.main([
        "--config",
        str(HYB / "config.yaml"),
        "--state",
        str(HYB / "model.ckpt"),
        "--float-state",
        str(ported),
        "--num-events",
        "4",
        "--out",
        str(out),
    ])
    text = out.read_text()
    assert "| Metric | Quantized | Float baseline | Δ |" in text
    assert "Total EBOPs:" in text and "nan" not in text.split("## Resource cost")[1].split("\n")[2].lower()


def loaded_hybrid():
    cfg = model_cfg(HYB)
    inputs, _ = stored_hybrid()
    model = port_to_keras.build_keras_maskformer_from_config(cfg, quant=cfg["init_args"]["quant"]).eval()
    return port_to_keras.load_keras_state(model, HYB / "model.ckpt", inputs), inputs


@pytest.mark.parametrize("which", ["innet0", "mask_object"])
def test_hybrid_export_assembles_swiglu_nets(which):
    """``hepattn.keras.export.build_functional_dense`` no longer refuses SwiGLU (it raised NotImplementedError)."""
    from hepattn.keras.export import build_functional_dense  # noqa: PLC0415

    model, _ = loaded_hybrid()
    net = model.input_nets[0].net if which == "innet0" else model.tasks[1].object_net
    assert net.gate
    shape = tuple(net.hidden[0]._build_shapes_dict["input_shape"][1:])  # noqa: SLF001
    functional = build_functional_dense(net, input_shape=shape, name=which)
    x = torch.randn(3, *shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = net(x)
        assert torch.equal(ref, functional(x, training=False))
        assert not torch.allclose(ref, net.final(net.hidden[0](x, training=False), training=False), atol=1e-3)  # control


def test_evaluate_hgq_hls_report_on_the_org_head(tmp_path):
    """``--hls`` converts the org classification head (SiLU) in its LUT export form (inline silu does not convert)."""
    out = tmp_path / "report.md"
    evaluate_hgq.main(["--config", str(HYB / "config.yaml"), "--state", str(HYB / "model.ckpt"), "--num-events", "4", "--hls", "--out", str(out)])
    text = out.read_text()
    assert "## hls4ml conversion" in text and '"n_layers"' in text and "warning" not in text
    assert (tmp_path / "hls_clf_head/firmware/myproject.cpp").is_file()


def test_eval_keras_main_on_the_hybrid_checkpoint(tmp_path, monkeypatch):
    """``eval_keras`` loads a ``MPflowHGQ`` Lightning checkpoint (strict) and evaluates the trained model."""
    loaded = []
    original = eval_keras.load_keras_state

    def capture(model, state_path, inputs):
        loaded.append(original(model, state_path, inputs))
        return loaded[-1]

    monkeypatch.setattr(eval_keras, "load_keras_state", capture)
    out = tmp_path / "metrics.json"
    eval_keras.main(["--config", str(HYB / "config.yaml"), "--state", str(HYB / "model.ckpt"), "--num-events", "4", "--out", str(out)])
    summary = json.loads(out.read_text())
    assert summary["num_events"] == 4 and summary["mean_losses"]
    assert all(np.isfinite(v) for v in summary["mean_losses"].values()), summary["mean_losses"]
    inputs, ref = stored_hybrid()
    with torch.no_grad():
        got = flatten_outputs(loaded[0](inputs))
    for key, value in ref.items():
        assert np.array_equal(got[key].numpy(), value), key


def test_control_eval_keras_old_load_is_refused(monkeypatch):
    """What ``eval_keras`` did before: a forward to materialize, then a strict load without registering."""
    from torch import nn  # noqa: PLC0415

    def old_load(model, state_path, inputs):
        with torch.no_grad():
            model(inputs)
        nn.Module.load_state_dict(model, port_to_keras.read_model_state(state_path))
        return model

    monkeypatch.setattr(eval_keras, "load_keras_state", old_load)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        eval_keras.main(["--config", str(HYB / "config.yaml"), "--state", str(HYB / "model.ckpt"), "--num-events", "4"])


def test_einsum_input_order_fixture_is_current():
    """The hybrid's QEinsum input order (read from the layer objects) is the committed ``einsum_inputs.json``."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("make_s7_fixtures", FIX / "make_s7_fixtures.py")
    make = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make)
    model, _ = loaded_hybrid()
    order = make.einsum_inputs(model)
    assert order == json.loads((HYB / "einsum_inputs.json").read_text())
    assert len(order) == 52 and all(len(paths) == 2 for paths in order.values())


TRIPWIRE: list[str] = []


def _tripwire(tag: str) -> str:
    TRIPWIRE.append(tag)
    return tag


class _PickledCall:
    """Unpickles by calling ``_tripwire``: the code a checkpoint file can carry."""

    def __reduce__(self):
        return (_tripwire, ("ran",))


def test_read_model_state_refuses_pickled_code(tmp_path):
    state = port_to_keras.read_model_state(HYB / "model.ckpt")
    path = tmp_path / "code.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}, "callbacks": _PickledCall()}, path)
    TRIPWIRE.clear()
    with pytest.raises(UnpicklingError, match="Weights only load failed"):
        port_to_keras.read_model_state(path)
    assert TRIPWIRE == []
    torch.load(path, map_location="cpu", weights_only=False)  # control: unrestricted, the file runs its code
    assert TRIPWIRE == ["ran"]
