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
- the converter's own verify report (``compare_forward``, the ``--report`` ``max_abs_diff``) equals the forward
  difference measured here, and reports a mis-mapped tensor; ``renorm_lane`` reports the renorm-lane premise (control:
  a projection column with ``S_c`` = 64 drops one entry and warns); a checkpoint carrying pickled code is refused.
The ``main_jax fit --init-from`` gate is in ``test_convert_fit.py``.
"""

import copy
import json
import math
from pathlib import Path
from pickle import UnpicklingError  # noqa: S403 (the exception type only)

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


def test_converter_report_is_the_measured_forward_difference(org):
    """``compare_forward`` (``max_abs_diff`` of ``--report``, the one-command check of a production conversion) equals
    this test's own measure, per output, and is not zero (every ORG_MEASURED value is > 0).
    """
    report = C.compare_forward(org["tmodel"], org["nmodel"], org["inputs"])
    expected = max_errors(org["ref"], native_outputs(org["nmodel"], org["inputs"]))
    assert report == {**expected, **dict.fromkeys(MASKS, 0.0)}
    assert all(report[key] > 0 for key in ORG_TOL), report


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
    reported = C.compare_forward(org["tmodel"], nmodel, org["inputs"])
    ratio = max(reported[key] / ORG_TOL[key] for key in ORG_TOL)
    assert ratio > 10, f"{control}: the converter's report gives only {ratio:.1f}x the tolerance"


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


def test_cli_writes_a_checkpoint_main_jax_loads(org, tmp_path, capsys):
    out = tmp_path / "org.npz"
    assert (
        C.main(["--config", str(ORG / "config.yaml"), "--ckpt", str(ORG / "model.ckpt"), "--out", str(out), "--report", str(tmp_path / "r.json")])
        == 0
    )
    assert "RESULT" in capsys.readouterr().out
    meta = nckpt.read_meta(out)
    assert meta["converted_from"]["source"] == "torch" and meta["converted_from"]["checkpoint"]["global_step"] == 4
    assert max(meta["converted_from"]["max_abs_diff"][k] / ORG_TOL[k] for k in ORG_TOL) <= 1.0
    # the CLI verifies on the fixture's events: its report is the measured difference (test_converter_report_is_...)
    assert meta["converted_from"]["max_abs_diff"] == C.compare_forward(org["tmodel"], org["nmodel"], org["inputs"])
    assert json.loads((tmp_path / "r.json").read_text())["max_abs_diff"] == meta["converted_from"]["max_abs_diff"]
    assert "renorm_lane" not in meta["converted_from"]  # a float target has no renorm lane
    with pytest.raises(FileExistsError):
        C.main(["--config", str(ORG / "config.yaml"), "--ckpt", str(ORG / "model.ckpt"), "--out", str(out)])


# ------------------------------------------------------------------------------------------- pairing of same-named quantizers


def einsum_input_check(nmodel, keras_vars: dict) -> tuple[list[str], int]:
    """Native ``QEinsum`` input quantizer j vs the hybrid's input quantizer j (``hybrid/einsum_inputs.json``, read from the
    hybrid layer objects, not from the converter's suffix order). Returns (mismatches, pairs whose two inputs differ).
    """
    import json  # noqa: PLC0415

    from hgq.layers import QEinsum  # noqa: PLC0415

    order = json.loads((HYB / "einsum_inputs.json").read_text())
    native = {layer.name: layer for layer in nmodel._flatten_layers(include_self=False, recursive=True) if isinstance(layer, QEinsum)}  # noqa: SLF001
    assert set(native) == set(order), sorted(set(native) ^ set(order))
    mismatches, distinct = [], 0
    for name, positions in order.items():
        quantizers = native[name].iq.quantizers
        assert len(quantizers) == len(positions) == 2, name
        values = []
        for j, (quantizer, paths) in enumerate(zip(quantizers, positions, strict=True)):
            source = {path.rsplit("/", 1)[1]: path for path in paths}
            assert sorted(source) == sorted(v.path.rsplit("/", 1)[1] for v in quantizer.variables), name
            values.append({leaf: keras_vars[path].numpy() for leaf, path in source.items()})
            for var in quantizer.variables:
                got = np.asarray(var.value)
                if not np.array_equal(got, values[j][var.path.rsplit("/", 1)[1]].astype(got.dtype)):
                    mismatches.append(f"{name} input {j} {var.path}")
        distinct += any(not np.array_equal(values[0][leaf], values[1][leaf]) for leaf in values[0])
    return mismatches, distinct


def test_hybrid_einsum_inputs_pair_by_input_position(hybrid):
    """Same-named quantizers (a ``QEinsum``'s ``quantizer`` / ``quantizer_1``) land on the same input as in the hybrid.

    The forward gate cannot see a swap here (the GLU input quantizers' ``f`` does not bind on the fixture's events), so
    this compares against the hybrid's own input order. 10 of the 52 pairs hold different values (asserted).
    """
    keras_vars, _ = C.hybrid_keras_state(hybrid["state"])
    mismatches, distinct = einsum_input_check(hybrid["nmodel"], keras_vars)
    assert mismatches == []
    assert distinct >= 10, f"only {distinct} QEinsum leaves have two different input quantizers: the check would be vacuous"


def test_control_swapped_einsum_inputs_are_found(hybrid):
    """Control: the two input quantizers of one GLU product swapped after the conversion are reported."""
    keras_vars, _ = C.hybrid_keras_state(hybrid["state"])
    nmodel = hybrid["rc"].build_model()
    nmodel(hybrid["inputs"], training=False)
    C.port_hybrid_state(hybrid["state"], nmodel)
    glu = nmodel.input_nets[0].net.products[0]
    for a, b in zip(glu.iq.quantizers[0].variables, glu.iq.quantizers[1].variables, strict=True):
        va, vb = np.array(a.value), np.array(b.value)
        a.assign(vb)
        b.assign(va)
    mismatches, _ = einsum_input_check(nmodel, keras_vars)
    assert mismatches and all(glu.name in m for m in mismatches), mismatches


# ------------------------------------------------------------------------------------------- refusals


def test_hybrid_variable_without_a_native_twin_is_refused(hybrid):
    state = copy.copy(hybrid["state"])
    state["encoder._torch_params.extra_layer/kernel"] = torch.zeros(3)
    nmodel = hybrid["rc"].build_model()
    nmodel(hybrid["inputs"], training=False)
    with pytest.raises(C.ConversionError, match=r"1 hybrid variables without a native twin: \['extra_layer/kernel: 1 hybrid, 0 native'\]"):
        C.port_hybrid_state(state, nmodel)


def test_hybrid_variable_with_two_values_is_refused(hybrid):
    """One keras variable under two module paths with different values (the hybrid stores it under every enclosing layer)."""
    state = copy.copy(hybrid["state"])
    by_path: dict = {}
    for key in state:
        if C.KERAS_PARAM_MARK in key:
            by_path.setdefault(key.split(C.KERAS_PARAM_MARK, 1)[1], []).append(key)
    path, keys = next((p, k) for p, k in by_path.items() if len(k) > 1 and p.endswith("/kernel"))
    state[keys[1]] = state[keys[1]] + 1
    nmodel = hybrid["rc"].build_model()
    nmodel(hybrid["inputs"], training=False)
    with pytest.raises(C.ConversionError, match=f"{path}: different values under two module paths"):
        C.port_hybrid_state(state, nmodel)


def test_hybrid_dtype_mismatch_is_refused(hybrid):
    state = copy.copy(hybrid["state"])
    keys = [k for k in state if C.KERAS_PARAM_MARK in k and k.endswith("innet0_net_final/kernel")]
    assert keys
    for key in keys:
        state[key] = state[key].double()
    nmodel = hybrid["rc"].build_model()
    nmodel(hybrid["inputs"], training=False)
    with pytest.raises(C.ConversionError, match=r"source dtype float64, native float32"):
        C.port_hybrid_state(state, nmodel)


def _ebops_state(hybrid, value: int, rename: bool) -> tuple[dict, str, str]:
    """The hybrid state with every module-path copy of one ``ebops`` counter set to ``value`` (int64); with ``rename``,
    under a variable path that does not end in ``/ebops``. Returns (state, variable path, its canonical).
    """
    state = copy.copy(hybrid["state"])
    path = next(k.split(C.KERAS_PARAM_MARK, 1)[1] for k in state if C.KERAS_PARAM_MARK in k and k.endswith("/ebops"))
    new = path + "_count" if rename else path
    for key in [k for k in state if k.endswith(C.KERAS_PARAM_MARK + path)]:
        assert state[key].dtype == torch.int64
        state[key.removesuffix(path) + new] = torch.full_like(state.pop(key), value)
    return state, new, C._canonical(C._segments(path))  # noqa: SLF001


@pytest.mark.parametrize("rename", [False, True])
def test_only_ebops_saturate_in_an_integer_cast(hybrid, monkeypatch, rename):
    """A value past uint32 saturates for an ``ebops`` counter (as on JAX) and is refused for any other integer variable.

    The same value goes into the same native uint32 variable; only the hybrid variable's name differs (the renamed
    path is matched to the ebops variable by a patched ``_match_native``).
    """
    state, path, canon = _ebops_state(hybrid, 2**33, rename)
    if rename:
        new_canon, match = C._canonical(C._segments(path)), C._match_native  # noqa: SLF001

        def renamed(nmodel, groups):
            found = match(nmodel, {canon if c == new_canon else c: m for c, m in groups.items()})
            found[new_canon] = found.pop(canon)
            return found

        monkeypatch.setattr(C, "_match_native", renamed)
    nmodel = hybrid["rc"].build_model()
    nmodel(hybrid["inputs"], training=False)
    if rename:
        with pytest.raises(C.ConversionError, match=f"{path}: values outside uint32"):
            C.port_hybrid_state(state, nmodel)
        return
    report = C.port_hybrid_state(state, nmodel)
    assert report["saturated_ebops"] == [path]
    (var,) = C._match_native(nmodel, {canon: [C._segments(path)]})[canon]  # noqa: SLF001
    assert np.dtype(var.dtype) == np.uint32 and np.all(np.asarray(var.value) == 2**32 - 1)


TRIPWIRE: list[str] = []


def _tripwire(tag: str) -> str:
    TRIPWIRE.append(tag)
    return tag


class _PickledCall:
    """Unpickles by calling ``_tripwire``: the code a checkpoint file can carry."""

    def __reduce__(self):
        return (_tripwire, ("ran",))


def test_a_checkpoint_carrying_pickled_code_is_refused(tmp_path):
    state, _ = C.read_lightning_state(ORG / "model.ckpt")
    path = tmp_path / "code.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}, "callbacks": _PickledCall()}, path)
    TRIPWIRE.clear()
    with pytest.raises(UnpicklingError, match="Weights only load failed"):
        C.read_lightning_state(path)
    assert TRIPWIRE == []
    torch.load(path, map_location="cpu", weights_only=False)  # control: unrestricted, the file runs its code
    assert TRIPWIRE == ["ran"]


@pytest.mark.parametrize("source", ["hybrid", "torch"])
def test_other_class_weights_are_refused(org, hybrid, source):
    """``loss_weights`` buffers are not ported but compared: other class weights are another config."""
    if source == "hybrid":
        state = copy.copy(hybrid["state"])
        state["tasks.0.loss_weights"] = state["tasks.0.loss_weights"] * 2
        nmodel = hybrid["rc"].build_model()
        nmodel(hybrid["inputs"], training=False)
        convert = lambda: C.port_hybrid_state(state, nmodel)  # noqa: E731
    else:
        tmodel = copy.deepcopy(org["tmodel"])
        tmodel.tasks[0].loss_weights.mul_(2)
        nmodel = org["rc"].build_model()
        nmodel(org["inputs"], training=False)
        convert = lambda: C.port_torch_model(tmodel, nmodel)  # noqa: E731
    with pytest.raises(C.ConversionError, match=r"tasks\.0\.loss_weights: \[.*\] differs from the native task constant"):
        convert()


def test_source_kind_refuses_another_model_class():
    assert C.source_kind({"class_path": "hepattn.models.MaskFormer"}) == C.TORCH_SOURCE
    assert C.source_kind({"class_path": "hepattn.keras.maskformer.KerasMaskFormer"}) == C.HYBRID_SOURCE
    with pytest.raises(C.ConversionError, match="neither MaskFormer nor KerasMaskFormer"):
        C.source_kind({"class_path": "hepattn.models.Encoder"})


def test_port_norm_without_bias():
    """An affine LayerNorm without bias (``bias=False``) ports its weight into a native norm without ``beta``."""
    import keras  # noqa: PLC0415

    tnorm = torch.nn.LayerNorm(4, bias=False)
    with torch.no_grad():
        tnorm.weight.copy_(torch.arange(1.0, 5.0))
    nnorm = keras.layers.LayerNormalization(epsilon=tnorm.eps, center=False)
    nnorm.build((None, 4))
    C.port_norm(tnorm, nnorm)
    assert np.array_equal(np.asarray(nnorm.gamma.value), [1.0, 2.0, 3.0, 4.0])
    assert nnorm.beta is None
    with pytest.raises(C.ConversionError, match="affine flags differ"):  # control: a native norm with beta
        C.port_norm(tnorm, keras.layers.LayerNormalization(epsilon=tnorm.eps))


# ------------------------------------------------------------------------------------------- flash attention types


def quadratic_run(tmp_path: Path, state: dict | None, enc_type: str, dec_type: str) -> tuple[Path, dict]:
    """The org fixture config with quadratic attention of the given torch types, and a checkpoint of ``state`` (default:
    the torch model's own initial weights). Returns (config, state).
    """
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load((ORG / "config.yaml").read_text())
    init = cfg["model"]["model"]["init_args"]
    init["encoder"]["init_args"].update(attn_type=enc_type, attn_kwargs={"num_heads": 2})
    init["decoder"]["init_args"]["decoder_layer_config"]["attn_kwargs"] = {"attn_type": dec_type, "num_heads": 2}
    run = tmp_path / f"{enc_type}_{dec_type}"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    if state is None:
        torch.manual_seed(0)
        state = C.instantiate_torch_model(C.source_model_section([run / "config.yaml"])).state_dict()
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}, "epoch": 0, "global_step": 0}, run / "model.ckpt")
    return run, state


# measured max |native - torch| on these seeded weights: 4.88e-3 at |regr| up to 1.44e4 (5 float32 ulp there, 2.9 eps*|x|),
# <= 1.1e-6 elsewhere (mask logits up to 9 ulp, 7.0 eps*|x|); the bound per output is 16 eps * max|x| (16-32 ulp of
# max|x|, by where it sits in its binade), at least 2.2e-6
QUADRATIC_ULPS, QUADRATIC_ABS = 16, 2.2e-6


def test_flash_attention_checkpoints_convert(tmp_path):
    """Org ``base.yaml`` trains the encoder with ``flash-varlen``; the native model coerces flash types to ``torch``.

    The converter compares the coerced types and runs its verify forward on the CPU with the torch SDPA backend: a
    flash / flash-varlen run converts to the same native model as the same weights under ``torch`` (bitwise), with the
    same forward difference. Before the fix the conversion raised "attn_type mismatch: torch flash-varlen vs native torch".
    """
    from hepattn.keras_native.encoder import COERCE_TO_TORCH  # noqa: PLC0415

    assert C.FLASH_TYPES == COERCE_TO_TORCH
    ref_run, state = quadratic_run(tmp_path, None, "torch", "torch")
    _, ref_model, ref_report = C.convert([ref_run / "config.yaml"], ref_run / "model.ckpt")
    inputs, _ = C.sample_batch(_rc(ref_run), 2)
    ref_out = native_outputs(ref_model, inputs)
    for key, diff in ref_report["max_abs_diff"].items():
        value = ref_out[key]
        scale = float(np.abs(value[np.abs(value) < SENTINEL]).max()) if value.dtype != bool else 0.0
        assert diff <= max(QUADRATIC_ABS, QUADRATIC_ULPS * np.finfo(np.float32).eps * scale), (key, diff, scale)
    for enc_type, dec_type in (("flash-varlen", "torch"), ("flash", "flash-varlen")):
        run, _ = quadratic_run(tmp_path, state, enc_type, dec_type)
        with pytest.warns(UserWarning, match="coerces attn_type"):
            _, model, report = C.convert([run / "config.yaml"], run / "model.ckpt")
        assert report["torch_tensors_read"] + len(report["ignored"]) == report["torch_tensors"]
        assert report["native_written"] == report["native_variables"]
        assert report["max_abs_diff"] == ref_report["max_abs_diff"], (enc_type, dec_type)
        out = native_outputs(model, inputs)
        assert all(np.array_equal(out[key], ref_out[key]) for key in ref_out), (enc_type, dec_type)
        # the porter itself accepts the flash types (no backend switch: port_torch_model on the model as configured)
        tmodel = C.instantiate_torch_model(C.source_model_section([run / "config.yaml"]))
        assert tmodel.encoder.attn_type == enc_type
        nmodel = _rc(run).build_model()
        nmodel(inputs, training=False)
        assert C.port_torch_model(tmodel, nmodel)["native_written"] == report["native_written"]


def _rc(run: Path):
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    return load_config([run / "config.yaml"])


def test_quadratic_into_linformer_is_refused(org, tmp_path):
    """Control for the type comparison: a quadratic torch attention does not port into a native linformer one."""
    run, _ = quadratic_run(tmp_path, None, "torch", "torch")
    tmodel = C.instantiate_torch_model(C.source_model_section([run / "config.yaml"]))
    nmodel = org["rc"].build_model()
    nmodel(org["inputs"], training=False)
    with pytest.raises(C.ConversionError, match="attn_type mismatch: torch torch vs native linformer"):
        C.port_torch_model(tmodel, nmodel)


# ------------------------------------------------------------------------------------------- renorm lane premise


def test_renorm_lane_premise_is_reported(org, hybrid):
    """Both quantized conversions report the premise of FPGA_STATUS footnote 5; it holds on the fixtures."""
    lane = hybrid["report"]["renorm_lane"]
    assert lane["projections"] == 12 and lane["dropped_entries"] == 0, lane  # 3 cores x 2 layers x (E, F)
    assert 2.0 < lane["s_c_max"] < 5.0, lane  # measured 3.68: below 32 by a factor 8.7
    assert "renorm_lane" not in org["report"]  # float target


def test_renorm_lane_check_counts_a_dropped_entry(hybrid, monkeypatch):
    """Control: one E column holds 2**-6 and 2**-7 in its head rows and 8 in its tail (S_c = 64 + 2**-12 + 2**-14).
    W(2**-7) ~ 2**-20 is under half the 2**-18 lane step, so it rounds to 0 (dropped); W(2**-6) ~ 2**-18 is kept. The
    converter reports it and warns. The 8 sits in the tail, so S_c must count the tail rows.
    """
    from hepattn.keras_native.masked_linformer import AttendCore, SeqProjection  # noqa: PLC0415

    core = next(c for c in hybrid["nmodel"]._flatten_layers(include_self=False) if isinstance(c, AttendCore) and c.renorm and c.E.tail is not None)  # noqa: SLF001
    head, tail = (np.array(t) for t in core.E.rows(False))  # a decoder kv cross-attention: the tail enters S_c
    head[:, 0], tail[:, 0] = 0.0, 0.0
    head[:2, 0], tail[0, 0] = [2.0**-6, 2.0**-7], 8.0
    rows = SeqProjection.rows
    monkeypatch.setattr(SeqProjection, "rows", lambda self, training=None: (head, tail) if self.path == core.E.path else rows(self, training))
    with pytest.warns(UserWarning, match="renorm lane premise fails"):
        _, _, report = C.convert([HYB / "config.yaml"], HYB / "model.ckpt")
    lane = report["renorm_lane"]
    assert lane["dropped_entries"] == 1, lane
    assert lane["s_c_max"] == 64 + 2.0**-12 + 2.0**-14 and lane["s_c_max_at"] == core.E.path, lane


# ------------------------------------------------------------------------------------------- GPU matcher solver in the source


def test_a_source_trained_with_a_gpu_matcher_solver_converts(org, tmp_path):
    """A run saved with ``device_solver: jv`` (the small CLIC reference run) converts like the same run without it.

    The torch Matcher checks for the CUDA ``torch-linear-assignment`` build in ``__init__``; the converter drops the key
    (it never matches) and reports it. The target is the source config without the key (the native driver refuses it).
    """
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load((ORG / "config.yaml").read_text())
    cfg["model"]["model"]["init_args"]["matcher"]["init_args"]["device_solver"] = "jv"
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(cfg))
    # control: the source as saved does not instantiate here (jsonargparse wraps the Matcher's RuntimeError)
    with pytest.raises((RuntimeError, ValueError), match="torch-linear-assignment"):
        C.instantiate_torch_model(C.source_model_section([source]))
    _, nmodel, report = C.convert([source], ORG / "model.ckpt", target_config=[ORG / "config.yaml"], verify=False)
    assert report["source_device_solver_dropped"] == "jv"
    assert "source_device_solver_dropped" not in org["report"]
    for a, b in zip(nmodel.variables, org["nmodel"].variables, strict=True):
        assert a.path == b.path and np.array_equal(np.asarray(a.value), np.asarray(b.value)), a.path
