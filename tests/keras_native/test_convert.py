"""S7 gates: Lightning checkpoints -> keras_native (``hepattn.experiments.clic.convert_to_native``).

Fixtures (``s7_fixtures/``, written by ``make_s7_fixtures.py`` in the hybrid env): two real Lightning runs at test
sizes (dim 16, 2 heads, 2 + 2 layers, masked Linformer k 8, 64 nodes, 16 queries), each one epoch of dummy events:
``org_float`` (``main.py``: org ``base.yaml`` + ``linformer.yaml``, the float torch MaskFormer) and ``hybrid``
(``main_hgq.py``: the generated Polaris config, ``KerasMaskFormer`` with HGQ2 and its BetaPID / EBOPs callbacks), plus
the hybrid's eval outputs from its checkpoint (``hybrid/outputs.npz``).

Gates (plan section 4, S7; tolerances are 2x the measured maxima, the R5 rule; measured 2026-10-20, macOS arm64 CPU):
- org float: the converted native forward equals the torch model loaded from the checkpoint, per output; the attention
  masks exactly, on a fixture whose valid mask logits are >= 10x the tolerance from the threshold (asserted). Controls:
  one mis-mapped tensor (an attention out_proj kernel not transposed, q / k of the packed in_proj swapped, the SwiGLU
  halves swapped) moves an output by > 10x its tolerance;
- coverage: every torch tensor read (task ``loss_weights`` compared with the native constants), every native variable
  written; controls: a porter that skips a tensor, or reads it without writing it, raises;
- hybrid: the converted native forward equals the hybrid's (measured exactly equal except float softmax rounding and
  the regression rows whose proxy depends on top-k tie-breaking, ``mf_golden_utils.regression_tie_rows``), every hybrid
  variable value is in the native model bit for bit (quantizer bitwidths included, which training moved away from
  their initial values: asserted), and survives the native checkpoint round trip. Control: a state without one
  quantizer variable raises; one bit less in one quantizer changes the outputs.
The ``main_jax fit --init-from`` gate is in ``test_convert_fit.py``.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hepattn.experiments.clic import convert_to_native as C
from hepattn.keras_native import checkpoint as nckpt

from .mf_golden_utils import regression_tie_rows

FIX = Path(__file__).parent / "s7_fixtures"
ORG = FIX / "org_float"
HYB = FIX / "hybrid"
SENTINEL = 1e30
LOGIT_THRESHOLD = math.log(0.1 / 0.9)  # pred_threshold 0.1 of the mask task

ORG_MEASURED = {  # max |native - torch| on the converter's 2 dummy events (valid slots)
    "final.classification.pflow_class_prob": 8.941e-08,
    "final.classification.pflow_logit": 4.769e-07,
    "final.incidence.pflow_incidence": 6.333e-08,
    "final.mask.pflow_node_logit": 2.742e-06,
    "final.regression.pflow_proxy_regr": 7.325e-04,  # |value| up to 3.6e3 (0.2 ppm)
    "final.regression.pflow_regr": 0.375,  # |value| up to 8.9e5: 6 float32 ulp
    "layer_0.classification.pflow_class_prob": 2.981e-08,
    "layer_0.classification.pflow_logit": 1.193e-07,
    "layer_0.mask.pflow_node_logit": 1.237e-06,
    "layer_1.classification.pflow_class_prob": 5.961e-08,
    "layer_1.classification.pflow_logit": 2.981e-07,
    "layer_1.mask.pflow_node_logit": 1.222e-06,
}
ORG_TOL = {key: 2 * value for key, value in ORG_MEASURED.items()}
HYB_MEASURED = {  # max |native - hybrid| on the 3 stored events; 0 = bitwise equal (kept at 0, R4b rule)
    "final.classification.pflow_class_prob": 4.471e-08,
    "final.classification.pflow_logit": 0.0,
    "final.incidence.pflow_incidence": 0.0,
    "final.mask.pflow_node_logit": 0.0,
    "final.regression.pflow_proxy_regr": 9.537e-07,  # tie rows excluded
    "final.regression.pflow_regr": 2.385e-07,
    "layer_0.classification.pflow_class_prob": 2.981e-08,
    "layer_0.classification.pflow_logit": 0.0,
    "layer_0.mask.pflow_node_logit": 0.0,
    "layer_1.classification.pflow_class_prob": 4.471e-08,
    "layer_1.classification.pflow_logit": 0.0,
    "layer_1.mask.pflow_node_logit": 0.0,
}
HYB_TOL = {key: 2 * value for key, value in HYB_MEASURED.items()}
MASKS = ["layer_0.attn_mask", "layer_1.attn_mask"]
BITWIDTH_LEAVES = ("/b", "/i", "/f")


def torch_inputs(inputs: dict) -> dict:
    return {k: torch.from_numpy(v.astype(np.int64) if v.dtype == np.int32 else v) for k, v in inputs.items()}


def max_errors(ref: dict, got: dict, keep_rows: dict | None = None) -> dict[str, float]:
    """Max |got - ref| per float output over the valid slots (the padding sentinel compared exactly)."""
    out = {}
    for key, value in ref.items():
        if value.dtype == bool:
            continue
        cand = np.asarray(got[key])
        assert cand.shape == value.shape, f"{key}: {cand.shape} vs {value.shape}"
        keep = np.abs(value) < SENTINEL
        assert np.array_equal(value[~keep], cand[~keep]), f"{key}: padding sentinel differs"
        if keep_rows is not None and key in keep_rows:
            keep &= keep_rows[key]
        out[key] = float(np.abs(value[keep].astype(np.float64) - cand[keep]).max()) if keep.any() else 0.0
    return out


# ------------------------------------------------------------------------------------------- org float model


@pytest.fixture(scope="module")
def org():
    rc, nmodel, report = C.convert([ORG / "config.yaml"], ORG / "model.ckpt", verify=False)
    tmodel = C.instantiate_torch_model(C.source_model_section([ORG / "config.yaml"]))
    state, _ = C.read_lightning_state(ORG / "model.ckpt")
    tmodel.load_state_dict(state, strict=True)
    tmodel.eval()
    inputs, _ = C.sample_batch(rc, 2)
    assert (~inputs["node_valid"]).any(-1).all(), "every event needs a padded node (batch of one all valid: org's unmasked path)"
    with torch.no_grad():
        ref = {k: v.numpy() for k, v in C._flatten(tmodel(torch_inputs(inputs))).items()}  # noqa: SLF001
    return {"rc": rc, "nmodel": nmodel, "report": report, "tmodel": tmodel, "inputs": inputs, "ref": ref}


def native_outputs(nmodel, inputs) -> dict:
    return {k: np.asarray(v) for k, v in C._flatten(nmodel(inputs, training=False)).items()}  # noqa: SLF001


def test_org_float_forward_equals_torch(org):
    ref, got = org["ref"], native_outputs(org["nmodel"], org["inputs"])
    assert set(ref) == set(got) == set(ORG_TOL) | set(MASKS)
    for key in MASKS:
        assert np.array_equal(ref[key], got[key]), f"{key}: {int((ref[key] != got[key]).sum())} flips"
        assert 0.05 < ref[key].mean() < 0.95, f"{key}: mask fraction {ref[key].mean():.3f}"
    errors = max_errors(ref, got)
    for key, tol in ORG_TOL.items():
        assert errors[key] <= tol, f"{key}: {errors[key]:.3e} > {tol:.3e}"


@pytest.mark.parametrize("layer", [0, 1])
def test_org_mask_logits_are_far_from_the_threshold(org, layer):
    """The exact mask check means something: every valid logit is >= 10x its tolerance from logit(0.1)."""
    ref = org["ref"][f"layer_{layer}.mask.pflow_node_logit"].astype(np.float64)
    margin = float(np.abs(ref[np.abs(ref) < SENTINEL] - LOGIT_THRESHOLD).min())
    assert margin >= 10 * ORG_TOL[f"layer_{layer}.mask.pflow_node_logit"], f"layer {layer}: margin {margin:.3e}"


def _swap_kernels(a, b) -> None:
    ka, kb = np.array(a.kernel.value), np.array(b.kernel.value)
    a.kernel.assign(kb)
    b.kernel.assign(ka)


CONTROLS = {
    # an attention out_proj kernel assigned as torch's weight, without the transpose (square, so it loads)
    "out_proj_not_transposed": lambda m: m.encoder.layers[1].attn.fn.out_proj.kernel.assign(
        np.array(m.encoder.layers[1].attn.fn.out_proj.kernel.value).T
    ),
    # q and k rows of the packed in_proj swapped in one decoder cross-attention
    "in_proj_q_k_swapped": lambda m: _swap_kernels(m.decoder.layers[0].q_ca.fn.q_proj, m.decoder.layers[0].q_ca.fn.k_proj),
    # the SwiGLU halves of one encoder FFN swapped (x1 = rows [:h], x2 = rows [h:])
    "swiglu_halves_swapped": lambda m: _swap_kernels(m.encoder.layers[0].dense.fn.hidden[0], m.encoder.layers[0].dense.fn.gates[0]),
}


@pytest.mark.parametrize("control", sorted(CONTROLS))
def test_org_float_mis_mapped_tensor_fails(org, control):
    rc = org["rc"]
    nmodel = rc.build_model()
    nmodel(org["inputs"], training=False)
    C.port_torch_model(org["tmodel"], nmodel)
    CONTROLS[control](nmodel)
    errors = max_errors(org["ref"], native_outputs(nmodel, org["inputs"]))
    ratio = max(errors[key] / ORG_TOL[key] for key in ORG_TOL)
    assert ratio > 10, f"{control}: the worst output is only {ratio:.1f}x its tolerance"


def test_org_float_coverage(org):
    report, rc = org["report"], org["rc"]
    assert report["source"] == "torch"
    assert report["native_written"] == report["native_variables"] == len(org["nmodel"].variables)
    assert report["native_kept_initial"] == 0
    assert report["torch_tensors_read"] + len(report["ignored"]) == report["torch_tensors"]
    assert sorted(report["ignored"]) == ["decoder.tasks.0.loss_weights", "tasks.0.loss_weights"]
    # only the decoder query self-attention drops rows: proj_k / proj_v (linformer_seq_len 64, k) -> the 16 query rows
    assert report["dropped_rows"] == {f"decoder_l{i}_q_sa_project_unmasked_{p}": 64 - 16 for i in range(2) for p in "EF"}
    assert rc.quant.spec is None


def test_coverage_check_catches_an_unread_tensor(org, monkeypatch):
    """Control: a porter that leaves the register tokens out is refused (the torch tensor is never read)."""
    original = C.port_encoder

    def skip_registers(tenc, nenc):
        tokens = tenc.register_tokens
        tenc.register_tokens = None
        nregs, nenc.num_register_tokens = nenc.num_register_tokens, None
        try:
            original(tenc, nenc)
        finally:
            tenc.register_tokens = tokens
            nenc.num_register_tokens = nregs

    monkeypatch.setattr(C, "port_encoder", skip_registers)
    nmodel = org["rc"].build_model()
    nmodel(org["inputs"], training=False)
    with pytest.raises(C.ConversionError, match=r"not ported: \['encoder.register_tokens'\]"):
        C.port_torch_model(org["tmodel"], nmodel)


def test_coverage_check_catches_an_unwritten_variable(org, monkeypatch):
    """Control: a porter that reads the Fourier ``B`` but never writes it is refused."""
    original = C._assign  # noqa: SLF001

    def drop_b(var, value):
        if var.path.endswith("/B"):
            return
        original(var, value)

    monkeypatch.setattr(C, "_assign", drop_b)
    nmodel = org["rc"].build_model()
    nmodel(org["inputs"], training=False)
    with pytest.raises(C.ConversionError, match=r"native weights were not written: \['maskformer/innet0/innet0_posenc/B'\]"):
        C.port_torch_model(org["tmodel"], nmodel)


# ------------------------------------------------------------------------------------------- hybrid HGQ2 model


@pytest.fixture(scope="module")
def hybrid():
    stored = dict(np.load(HYB / "outputs.npz"))
    inputs = {k.removeprefix("in::"): v for k, v in stored.items() if k.startswith("in::")}
    ref = {k.removeprefix("out::"): v for k, v in stored.items() if k.startswith("out::")}
    rc, nmodel, report = C.convert([HYB / "config.yaml"], HYB / "model.ckpt")
    state, _ = C.read_lightning_state(HYB / "model.ckpt")
    return {"rc": rc, "nmodel": nmodel, "report": report, "inputs": inputs, "ref": ref, "state": state}


def hybrid_errors(h, nmodel) -> tuple[dict, int]:
    ref = h["ref"]
    ties = regression_tie_rows(ref["final.incidence.pflow_incidence"], h["inputs"]["node_is_track"])
    keep = {key: ~ties[..., None] for key in ref if key.startswith("final.regression")}
    return max_errors(ref, native_outputs(nmodel, h["inputs"]), keep), int(ties.sum())


def test_hybrid_forward_equals_hybrid(hybrid):
    ref, got = hybrid["ref"], native_outputs(hybrid["nmodel"], hybrid["inputs"])
    assert set(ref) == set(got) == set(HYB_TOL) | set(MASKS)
    assert hybrid["rc"].quant.spec is not None, "the hybrid fixture must be the quantized model"
    for key in MASKS:
        assert np.array_equal(ref[key], got[key]), f"{key}: {int((ref[key] != got[key]).sum())} flips"
    errors, tie_rows = hybrid_errors(hybrid, hybrid["nmodel"])
    assert 0 < tie_rows < 10, tie_rows  # measured 5 of 48 particle rows; the rest of the regression is compared
    for key, tol in HYB_TOL.items():
        assert errors[key] <= tol, f"{key}: {errors[key]:.3e} > {tol:.3e}"


def test_hybrid_controls_change_the_outputs(hybrid):
    """Controls: a fresh native model, and one fractional bit less in one weight quantizer, are far from the hybrid."""
    rc = hybrid["rc"]
    fresh = rc.build_model()
    fresh(hybrid["inputs"], training=False)
    errors, _ = hybrid_errors(hybrid, fresh)
    assert errors["final.mask.pflow_node_logit"] > 1.0, errors

    nmodel = rc.build_model()
    nmodel(hybrid["inputs"], training=False)
    C.port_hybrid_state(hybrid["state"], nmodel)
    var = next(v for v in nmodel.variables if "encoder_l1_attn_out" in v.path and "_kq/" in v.path and v.path.endswith("/b"))
    var.assign(np.array(var.value) - 1)
    errors, _ = hybrid_errors(hybrid, nmodel)
    assert errors["final.mask.pflow_node_logit"] > 1e-3, errors


def test_hybrid_state_is_copied_bit_for_bit(hybrid):
    """Every distinct hybrid keras variable equals its native twin, bitwidths included; training moved the bitwidths."""
    keras_vars, _ = C.hybrid_keras_state(hybrid["state"])
    groups: dict = {}
    for path in keras_vars:
        groups.setdefault(C._canonical(C._segments(path)), []).append(C._segments(path))  # noqa: SLF001
    native = C._match_native(hybrid["nmodel"], groups)  # noqa: SLF001
    fresh = hybrid["rc"].build_model()
    fresh(hybrid["inputs"], training=False)
    initial = {id(v): np.array(f.value) for v, f in zip(hybrid["nmodel"].variables, fresh.variables, strict=True)}
    checked = moved = 0
    for canon, members in groups.items():
        for segs, var in zip(sorted(members, key=C._suffix_order), native[canon], strict=True):  # noqa: SLF001
            path = "/".join(segs)
            got = np.array(var.value)
            assert np.array_equal(got, keras_vars[path].numpy().astype(got.dtype)), path
            checked += 1
            moved += path.endswith(BITWIDTH_LEAVES) and not np.array_equal(got, initial[id(var)])
    assert checked == len(keras_vars) == hybrid["report"]["hybrid_keras_variables"]
    assert moved > 50, f"only {moved} bitwidth arrays differ from the initial state: the copy check would be vacuous"


def test_hybrid_state_round_trips_through_the_native_checkpoint(hybrid, tmp_path):
    rc, nmodel = hybrid["rc"], hybrid["nmodel"]
    path = C.write_native_checkpoint(tmp_path / "hyb.npz", rc, nmodel, hybrid["report"])
    fresh = rc.build_model()
    fresh(hybrid["inputs"], training=False)
    (tv, ntv, ov), meta = nckpt.load_checkpoint(path, fresh, None, expect_config_hash=rc.model_hash)
    assert ov == [] and meta["step"] == 0 and meta["converted_from"]["source"] == "hybrid"
    nckpt.assign_state(fresh, None, (tv, ntv, []))
    for a, b in zip(nmodel.variables, fresh.variables, strict=True):
        assert np.array_equal(np.array(a.value), np.array(b.value)), a.path
    errors, _ = hybrid_errors(hybrid, fresh)
    for key, tol in HYB_TOL.items():
        assert errors[key] <= tol, f"{key}: {errors[key]:.3e} > {tol:.3e}"


def test_a_missing_quantizer_variable_is_refused(hybrid):
    """Control: the state without one quantizer variable (under every module path that holds it) does not convert."""
    state = hybrid["state"]
    victim = next(k.split(C.KERAS_PARAM_MARK, 1)[1] for k in state if C.KERAS_PARAM_MARK in k and k.endswith("/f") and "decoder_l1" in k)
    pruned = {k: v for k, v in state.items() if not k.endswith(C.KERAS_PARAM_MARK + victim)}
    assert len(pruned) < len(state)
    nmodel = hybrid["rc"].build_model()
    nmodel(hybrid["inputs"], training=False)
    with pytest.raises(C.ConversionError, match=r"1 native variables have no hybrid source") as err:
        C.port_hybrid_state(pruned, nmodel)
    leaf = C._canonical(C._segments(victim))  # noqa: SLF001
    assert C._canonical(C._segments(str(err.value).split("['")[1].rstrip("']"))).endswith(leaf), (victim, str(err.value))  # noqa: SLF001


def test_hybrid_unknown_torch_tensor_is_refused(hybrid):
    state = copy.copy(hybrid["state"])
    state["decoder.some_new_buffer"] = torch.zeros(3)
    nmodel = hybrid["rc"].build_model()
    nmodel(hybrid["inputs"], training=False)
    with pytest.raises(C.ConversionError, match=r"decoder\.some_new_buffer"):
        C.port_hybrid_state(state, nmodel)


def test_cli_writes_a_checkpoint_main_jax_loads(tmp_path, capsys):
    out = tmp_path / "org.npz"
    assert (
        C.main(["--config", str(ORG / "config.yaml"), "--ckpt", str(ORG / "model.ckpt"), "--out", str(out), "--report", str(tmp_path / "r.json")])
        == 0
    )
    assert "RESULT" in capsys.readouterr().out
    meta = nckpt.read_meta(out)
    assert meta["converted_from"]["source"] == "torch" and meta["converted_from"]["checkpoint"]["global_step"] == 4
    assert max(meta["converted_from"]["max_abs_diff"][k] / ORG_TOL[k] for k in ORG_TOL) <= 1.0
    with pytest.raises(FileExistsError):
        C.main(["--config", str(ORG / "config.yaml"), "--ckpt", str(ORG / "model.ckpt"), "--out", str(out)])
