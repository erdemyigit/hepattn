"""S3 MaskFormer parity against the hybrid golden files, and the full-model paired delta (carried from S2).

Golden files: ``golden/make_golden_maskformer.py`` (hybrid ``KerasMaskFormer``, torch env). Modes: float, high, low
(the paired-delta configs), prod (``pflow_hgq.yaml`` quantizers) and prodsat (prod with SAT_SYM datalane overflow).
Tolerances: 2x the maxima measured by ``golden/measure_mf_golden.py`` (``golden/mf_measured.json``), per output, for
the free-running model (mask feedback included) and for the teacher-forced one (hybrid masks injected), plus the
attention-mask flip counts. Under prod's WRAP overflow, an ulp-level backend difference at a rounding or wrap edge
jumps by up to 2**(i+1) and then feeds back through the masks, so prod is gated only before the first feedback
(layer-0 mask logits and flips); see docs/hgq/port_log/S3.md.
"""

import json

import keras
import numpy as np
import pytest

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch, torch_model
from .mf_golden_utils import (
    GOLDEN_DIR,
    JIT_ROWS,
    LAYER0_LOGIT,
    MF_CASES,
    MF_MODES,
    build_mf,
    build_mf_from_golden,
    compare_mf,
    load_mf_golden,
    mf_init_state_mismatches,
    mf_inputs,
    mf_meta,
    qloss_terms,
)
from .parity_utils import record_measurement
from .porting import port_maskformer

MEASURED = json.loads((GOLDEN_DIR / "mf_measured.json").read_text())
CHAOTIC = {"prod"}
LAYER0_KEYS = {f"free.{LAYER0_LOGIT}", "free.flips.layer_0.attn_mask"}


def gated(mode: str, key: str) -> bool:
    if key in {"tie_rows", "qlosses.max_rel"}:
        return False
    return key in LAYER0_KEYS if mode in CHAOTIC else True


ROWS = [(case, mode, False) for case in MF_CASES for mode in MF_MODES] + [(case, mode, True) for case, mode in sorted(JIT_ROWS)]

# R0 xfail (docs/hgq/RETARGET_ORG_MAIN.md section 2, item 4); strict, so the marker fails once the named lane fixes the test.
R5_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="R5: the torch reference passes linformer_proj_dim, which org Attention rejects since PR #9; goldens regenerate in R5",
)
XROWS = [pytest.param(*row, marks=R5_XFAIL) if row[0] == "mf_lin" else row for row in ROWS]


@pytest.mark.parametrize(("case", "mode", "jit"), XROWS, ids=[f"{c}-{m}{'-jit' if j else ''}" for c, m, j in ROWS])
def test_mf_golden_parity(case, mode, jit):
    """Eager rows, and the jitted forward (as training runs it) for JIT_ROWS: XLA may fuse ops, so its rounding at
    quantizer boundaries can differ from eager (measured: high layer-0 mask logits up to 1.2e-3 apart, jit vs eager).
    """
    model, golden = build_mf_from_golden(case, mode)
    metrics = compare_mf(model, golden, mode, jit=jit)
    row = f"{case}/{mode}" + ("/jit" if jit else "")
    measured = {k: v for k, v in MEASURED[row].items() if k != "qlosses.max_rel"}
    assert set(metrics) == set(measured), f"metric keys changed: {set(metrics) ^ set(measured)}; re-run measure_mf_golden.py"
    assert metrics["tie_rows"] == measured["tie_rows"]
    if mode in {"float", "high"}:
        assert metrics["tie_rows"] == 0, "top-k ties in a float-like mode: the tie exclusion would hide real differences"
    for key, value in metrics.items():
        if key != "tie_rows":
            record_measurement({"tag": f"maskformer.golden.{row}", "config": key, "max_abs_err": value, "atol": 2 * measured[key], "rtol": 0.0})
    bad = {key: (value, 2 * measured[key]) for key, value in metrics.items() if gated(mode, key) and value > 2 * measured[key]}
    assert not bad, f"{row}: above 2x measured (value, tolerance): {bad}"
    if mode == "float":  # the e2e row: masks exactly equal, values far inside 5e-5 + 1e-4 |ref|
        assert all(v == 0 for k, v in metrics.items() if ".flips." in k)


@pytest.mark.parametrize("mode", [m for m in MF_MODES if m != "float"])
@pytest.mark.parametrize("case", [pytest.param(c, marks=R5_XFAIL) if c == "mf_lin" else c for c in MF_CASES])
def test_quant_losses_match_hybrid_per_layer(case, mode):
    """The training-mode HGQ2 loss terms of ``stateless_call(return_losses=True)`` are the hybrid's, each layer once.

    Same number of terms, and each term (sorted) within 2x the measured relative difference. The hybrid's own
    ``quant_losses()`` counts the task-head layers twice (``model.tasks`` and ``model.decoder.tasks`` are the same
    modules and ``keras_layers()`` does not deduplicate); the golden meta records that, and the last assert pins it,
    so S5 compares penalties against the per-layer sum, not against quant_losses().
    """
    model, golden = build_mf_from_golden(case, mode)
    native, hybrid = qloss_terms(model, golden, mode)
    assert native.size == hybrid.size > 0, f"{native.size} native vs {hybrid.size} hybrid loss terms"
    rel = float((np.abs(native - hybrid) / np.abs(hybrid)).max())
    tol = 2 * MEASURED[f"{case}/{mode}"]["qlosses.max_rel"]
    record_measurement({
        "tag": f"maskformer.qlosses.{case}.{mode}",
        "config": "sorted terms",
        "max_abs_err": float(np.abs(native - hybrid).max()),
        "max_rel_err": rel,
        "atol": 0.0,
        "rtol": tol,
    })
    assert rel <= tol, f"loss terms differ by rel {rel:.3e} > {tol:.3e}"
    q = mf_meta(golden)[mode]["quant_losses"]
    n_task_leaves = sum(1 for name in mf_meta(golden)[mode]["leaves"] if name.startswith("task"))
    assert q["n_layers_raw"] - q["n_layers_unique"] == n_task_leaves, q
    # counts, not sums: quant_losses() sums in float32, whose rounding at ~3e12 (high mode) hides the doubled terms
    assert q["n_terms_raw"] > hybrid.size, "hybrid quant_losses() no longer double counts: revisit S3.md"


@pytest.mark.parametrize("mode", [m for m in MF_MODES if m != "float"])
@pytest.mark.parametrize("case", list(MF_CASES))
def test_init_quantizer_state_matches_hybrid(case, mode):
    """A freshly built native model has the hybrid's initial quantizer state (all scopes applied), bit for bit."""
    golden = load_mf_golden(case)
    model = build_mf(MF_CASES[case], mf_meta(golden)[mode]["quant"], mf_inputs(golden))
    bad = mf_init_state_mismatches(model, golden, mode)
    assert not bad, f"{len(bad)} leaf variables differ from the hybrid's initial state, e.g. {bad[:5]}"


def test_chaotic_mode_is_really_wrap():
    """Prod and prodsat differ only in the datalane overflow mode (so the prod gating is about WRAP, nothing else)."""
    meta = json.loads(str(np.load(GOLDEN_DIR / "mf_lin.npz")["meta"]))
    prod, sat = meta["prod"]["quant"], meta["prodsat"]["quant"]
    assert {**prod["datalane"], "overflow_mode": "SAT_SYM"} == sat["datalane"]
    assert {k: v for k, v in prod.items() if k != "datalane"} == {k: v for k, v in sat.items() if k != "datalane"}


# The hybrid's own paired-delta measurement (tests/keras/test_quantized.py, layer-0 mask logits, seed 50, max finite
# |quantized - float|), read from tests/keras/.parity_ledger.jsonl of the hybrid run on this machine (PARITY.md rows
# quantized.delta.mask_l0: 3.01e-01 / 3.11e+01).
HYBRID_DELTA = {"high": 0.3014124035835266, "low": 31.091598510742188}
# native - hybrid, measured: high 2.81e-5 (0.3014404773712158), low 0.0 (bit-exact); tolerance 2x measured
DELTA_TOL = {"high": 5.62e-5, "low": 0.0}


def test_full_model_paired_delta_reproduces_hybrid():
    """Carried from S2: high ~0.30 vs low ~31 on the layer-0 mask logits, >= 10x apart, as the hybrid measured it.

    Same model, seed, inputs and quantizer configs as tests/keras/test_quantized.py. If the quantizers were inert the
    two deltas would not separate; if the scopes did not apply, high would not stay near float.
    """
    inputs, _ = clic_dummy_batch()

    def layer0(mode):
        tmodel = torch_model("torch", seed=50, perturb=False)
        model = build_mf("torch", QUANT_SPECS[mode], inputs)
        port_maskformer(tmodel, model)
        return keras.ops.convert_to_numpy(model(inputs, training=False)["layer_0"]["mask"]["pflow_node_logit"]).astype(np.float64)

    ref = layer0("float")
    deltas = {}
    for mode in ("high", "low"):
        out = layer0(mode)
        keep = np.isfinite(ref) & np.isfinite(out)
        deltas[mode] = float(np.abs(out[keep] - ref[keep]).max())
        record_measurement({
            "tag": "maskformer.paired_delta.mask_l0",
            "config": mode,
            "max_abs_err": deltas[mode],
            "atol": DELTA_TOL[mode],
            "rtol": 0.0,
        })
    for mode, value in deltas.items():
        assert abs(value - HYBRID_DELTA[mode]) <= DELTA_TOL[mode], f"{mode}: native delta {value!r} vs hybrid {HYBRID_DELTA[mode]!r}"
    assert deltas["low"] > max(10 * deltas["high"], 1.0), deltas
