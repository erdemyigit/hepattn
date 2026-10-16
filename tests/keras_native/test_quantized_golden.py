"""Layer golden parity: keras_native (JAX) vs golden files from the hybrid keras-on-torch layers.

The golden files (``golden/<case>.npz``, written by ``golden/make_golden.py`` in the hybrid torch env) hold the
hybrid's weights and quantizer state, inputs and outputs for float / high / low bitwidth (the HIGH_QUANT /
LOW_QUANT specs of ``tests/keras/test_quantized.py``). The native twin gets exactly those weights.

Cases: the S2 layers (dense, dense attention, input net, encoder with dense attention), the quantized SwiGLU denses,
and the org/main masked-Linformer cases of R5 (``golden/cases.py``): the three Linformer call paths and the PR #8
encoder. For the masked-Linformer core this is backend consistency (torch vs JAX backend of the same keras_native
core, docs/hgq/RETARGET_ORG_MAIN.md section 4); everything around the core is independent hybrid code.

Tolerances: measured first (S2 rows 2026-10-14, R5 rows 2026-10-16; macOS arm64 CPU, jax 0.11.2 vs torch 2.10.0,
keras 3.15.0, hgq2 0.1.9), then fixed at 2x the measured max abs error. A measured 0 means bit-exact, so the
tolerance is 0 (carried gate: re-measure on the first Linux/GPU run).
"""

import keras
import numpy as np
import pytest

from .golden.cases import CASES, modes
from .golden_utils import build_from_golden, build_native, init_state_mismatches, inputs_of, load_golden, run
from .parity_utils import record_measurement

# measured max |native - hybrid| per (case, mode); tolerance = 2x
MEASURED = {
    ("dense_ffn", "float"): 1.192e-07, ("dense_ffn", "high"): 3.599e-06, ("dense_ffn", "low"): 0.0,
    ("dense_innet", "float"): 1.192e-07, ("dense_innet", "high"): 4.053e-06, ("dense_innet", "low"): 0.0,
    ("attn_self", "float"): 1.490e-07, ("attn_self", "high"): 2.548e-05, ("attn_self", "low"): 0.0,
    ("attn_cross", "float"): 2.384e-07, ("attn_cross", "high"): 5.341e-05, ("attn_cross", "low"): 0.0,
    ("dense_ffn_swiglu", "float"): 5.961e-08, ("dense_ffn_swiglu", "high"): 3.785e-06, ("dense_ffn_swiglu", "low"): 0.0,
    ("dense_innet_swiglu", "float"): 5.961e-08, ("dense_innet_swiglu", "high"): 4.147e-06, ("dense_innet_swiglu", "low"): 0.0,
    ("lin_enc", "float"): 2.087e-07, ("lin_enc", "high"): 3.411e-05, ("lin_enc", "low"): 0.0,
    ("lin_q_ca", "float"): 2.981e-07, ("lin_q_ca", "high"): 7.478e-05, ("lin_q_ca", "low"): 0.0,
    ("lin_kv_ca", "float"): 1.789e-07, ("lin_kv_ca", "high"): 6.695e-05, ("lin_kv_ca", "low"): 0.0,
    ("enc_lin", "float"): 7.153e-07, ("enc_lin", "high"): 8.828e-05, ("enc_lin", "low"): 4.769e-07,
    ("enc_torch", "float"): 7.153e-07, ("enc_torch", "high"): 6.116e-05, ("enc_torch", "low"): 7.153e-07,
    ("innet", "float"): 2.384e-07, ("innet", "high"): 4.053e-06, ("innet", "low"): 1.192e-07,
}  # fmt: skip
TOLERANCE = {key: 2.0 * value for key, value in MEASURED.items()}
CASE_MODES = [(case, mode) for case in CASES for mode in modes(case)]


@pytest.fixture(scope="module")
def native_outputs():
    """Native outputs for every case and mode (built once per module)."""
    cache = {}

    def get(case: str):
        if case not in cache:
            golden = load_golden(case)
            outs = {}
            for mode in modes(case):
                module, _, _ = build_from_golden(case, mode, golden)
                outs[mode] = keras.ops.convert_to_numpy(run(case, module, inputs_of(golden))).astype(np.float64)
            cache[case] = (golden, outs)
        return cache[case]

    return get


def test_tolerance_table_covers_every_golden_case():
    assert set(MEASURED) == set(CASE_MODES)


@pytest.mark.parametrize(("case", "mode"), [(case, mode) for case, mode in CASE_MODES if mode != "float"])
def test_init_quantizer_state_matches_hybrid(case, mode):
    """Built in the same HGQ2 scopes, the native leaves start from the hybrid's quantizer state, bit for bit.

    The parity test below overwrites all leaf variables, so on its own it cannot see a native layer that
    ignores the config scopes; this test can (every non-kernel/bias variable is compared before loading).
    """
    golden = load_golden(case)
    module, _ = build_native(case, mode, golden)
    mismatches = init_state_mismatches(module, golden, mode)
    assert not mismatches, f"{case}/{mode}: initial quantizer state differs from the hybrid at {mismatches[:8]}"


@pytest.mark.parametrize(("case", "mode"), CASE_MODES)
def test_golden_parity(native_outputs, case, mode):
    golden, outs = native_outputs(case)
    ref = golden[f"{mode}::out"].astype(np.float64)
    diff = np.abs(outs[mode] - ref)
    assert np.isfinite(outs[mode]).all()
    record_measurement({"tag": f"golden.{case}", "config": mode, "max_abs_err": float(diff.max()), "atol": TOLERANCE[case, mode], "rtol": 0.0})
    tol = TOLERANCE[case, mode]
    assert diff.max() <= tol, f"{case}/{mode}: max |native - hybrid| = {diff.max():.3e} > {tol:.3e} ({int((diff > tol).sum())} elements)"


@pytest.mark.parametrize("case", CASES)
def test_paired_delta(native_outputs, case):
    """High bitwidth stays near float, low bitwidth does not: separation >= 10x (inert quantizers give 0 / 0).

    The native deltas must also equal the hybrid's (both use the same weights), to within the parity tolerances.
    """
    golden, outs = native_outputs(case)
    deltas = {mode: float(np.abs(outs[mode] - outs["float"]).max()) for mode in ("high", "low")}
    hybrid = {mode: float(np.abs(golden[f"{mode}::out"].astype(np.float64) - golden["float::out"]).max()) for mode in ("high", "low")}
    for mode in ("high", "low"):
        record_measurement({"tag": f"quantized.delta.{case}", "config": mode, "max_abs_err": deltas[mode], "hybrid": hybrid[mode]})
        assert abs(deltas[mode] - hybrid[mode]) <= TOLERANCE[case, mode] + TOLERANCE[case, "float"], (case, mode, deltas, hybrid)
    assert deltas["high"] > 0, f"{case}: high-bitwidth output equals float exactly; quantizers look inert"
    ratio = deltas["low"] / deltas["high"]
    assert ratio >= 10, f"{case}: low/high delta ratio {ratio:.2f} < 10 (high {deltas['high']:.3e}, low {deltas['low']:.3e})"


def _drop(key: str):
    def edit(module, inputs: dict) -> dict:  # noqa: ARG001  (same signature as _zero_tail)
        return {**inputs, key: None}

    return edit


def _zero_tail(module, inputs: dict) -> dict:
    for proj in (module.core.E, module.core.F):
        proj.tail.assign(np.zeros(proj.tail.shape, dtype="float32"))
    return inputs


# One org feature removed from the native float twin per masked-Linformer case; the output must move by >= 10x the
# golden tolerance, so the golden rows see that feature (measured in docs/hgq/port_log/R5.md).
LIN_CONTROLS = {
    "lin_enc": _drop("kv_mask"),  # no key mask: org project without renorm, padding attended
    "lin_q_ca": _drop("kv_sort_idx"),  # keys enter E/F in input order
    "lin_kv_ca": _zero_tail,  # E/F rows 150-159 enter only the renorm numerator
    "enc_lin": _drop("x_sort_value"),  # no phi sorting
}


@pytest.mark.parametrize("case", sorted(LIN_CONTROLS))
def test_linformer_golden_rows_see_the_org_features(case):
    module, _, golden = build_from_golden(case, "float")
    inputs = LIN_CONTROLS[case](module, inputs_of(golden))
    out = keras.ops.convert_to_numpy(run(case, module, inputs)).astype(np.float64)
    diff = float(np.abs(out - golden["float::out"]).max())
    record_measurement({"tag": f"golden.control.{case}", "config": "float", "max_abs_err": diff, "atol": TOLERANCE[case, "float"]})
    assert diff > 10 * TOLERANCE[case, "float"], f"{case}: control moves the output by only {diff:.3e}"
