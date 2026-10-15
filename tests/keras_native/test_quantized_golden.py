"""S2 quantized parity: keras_native (JAX) vs golden files from the hybrid keras-on-torch layers.

The golden files (``golden/<case>.npz``, written by ``golden/make_golden.py`` in the hybrid torch env) hold the
hybrid's weights and quantizer state, inputs and outputs for float / high / low bitwidth (the HIGH_QUANT /
LOW_QUANT specs of ``tests/keras/test_quantized.py``). The native twin gets exactly those weights.

Tolerances: measured first (2026-10-14, macOS arm64 CPU, jax 0.11.2 vs torch 2.10.0, keras 3.15.0, hgq2 0.1.9),
then fixed at 2x the measured max abs error. A measured 0 means bit-exact, so the tolerance is 0.
"""

import keras
import numpy as np
import pytest

from .golden.cases import CASES, EXTRA_MODES, modes
from .golden_utils import build_from_golden, build_native, init_state_mismatches, inputs_of, load_golden, run
from .parity_utils import record_measurement

# measured max |native - hybrid| per (case, mode); tolerance = 2x
MEASURED = {
    ("dense_ffn", "float"): 1.192e-07, ("dense_ffn", "high"): 3.599e-06, ("dense_ffn", "low"): 0.0,
    ("dense_innet", "float"): 1.192e-07, ("dense_innet", "high"): 4.053e-06, ("dense_innet", "low"): 0.0,
    ("attn_self", "float"): 1.490e-07, ("attn_self", "high"): 2.548e-05, ("attn_self", "low"): 0.0,
    ("attn_cross", "float"): 2.384e-07, ("attn_cross", "high"): 5.341e-05, ("attn_cross", "low"): 0.0,
    ("lin_self", "float"): 2.980e-08, ("lin_self", "high"): 0.0, ("lin_self", "low"): 0.0, ("lin_self", "mid"): 0.0,
    ("lin_cross", "float"): 2.980e-08, ("lin_cross", "high"): 0.0, ("lin_cross", "low"): 0.0, ("lin_cross", "mid"): 0.0,
    ("enc_lin", "float"): 7.153e-07, ("enc_lin", "high"): 4.548e-05, ("enc_lin", "low"): 7.153e-07,
    ("enc_torch", "float"): 7.153e-07, ("enc_torch", "high"): 6.116e-05, ("enc_torch", "low"): 7.153e-07,
    ("innet", "float"): 2.384e-07, ("innet", "high"): 4.053e-06, ("innet", "low"): 1.192e-07,
}  # fmt: skip
TOLERANCE = {key: 2.0 * value for key, value in MEASURED.items()}
CASE_MODES = [(case, mode) for case in CASES for mode in modes(case)]

# R0 xfail (docs/hgq/RETARGET_ORG_MAIN.md section 2, item 4); strict, so the marker fails once the named lane fixes the test.
R5_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="R5: the torch reference passes linformer_proj_dim, which org Attention rejects since PR #9; goldens regenerate in R5",
)
XCASE_MODES = [pytest.param(*row, marks=R5_XFAIL) if row[0] == "enc_lin" else row for row in CASE_MODES]

# The hybrid oracle itself separates a single Linformer layer by only 5.3x (self) / 8.5x (cross): at 4-bit the
# attention output underflows and the layer returns its quantized bias, so the low delta is capped by the
# output scale (~0.3). There the native ratio must equal the hybrid's instead (see docs/hgq/port_log/S2.md).
RATIO_LIKE_HYBRID = {"lin_self", "lin_cross"}


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


@pytest.mark.parametrize(("case", "mode"), XCASE_MODES)
def test_golden_parity(native_outputs, case, mode):
    golden, outs = native_outputs(case)
    ref = golden[f"{mode}::out"].astype(np.float64)
    diff = np.abs(outs[mode] - ref)
    assert np.isfinite(outs[mode]).all()
    record_measurement({"tag": f"golden.{case}", "config": mode, "max_abs_err": float(diff.max()), "atol": TOLERANCE[case, mode], "rtol": 0.0})
    tol = TOLERANCE[case, mode]
    assert diff.max() <= tol, f"{case}/{mode}: max |native - hybrid| = {diff.max():.3e} > {tol:.3e} ({int((diff > tol).sum())} elements)"


@pytest.mark.parametrize("case", [pytest.param(c, marks=R5_XFAIL) if c == "enc_lin" else c for c in CASES])
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
    ratio, hybrid_ratio = deltas["low"] / deltas["high"], hybrid["low"] / hybrid["high"]
    if case in RATIO_LIKE_HYBRID:
        assert ratio == pytest.approx(hybrid_ratio, rel=1e-2), (case, ratio, hybrid_ratio)
    else:
        assert ratio >= 10, f"{case}: low/high delta ratio {ratio:.2f} < 10 (high {deltas['high']:.3e}, low {deltas['low']:.3e})"


@pytest.mark.parametrize("case", sorted(EXTRA_MODES))
def test_mid_rows_reach_the_quantized_linformer_graph(native_outputs, case):
    """The "mid" rows exist because "low" collapses a single Linformer layer to its quantized bias.

    If "low" ever becomes input-dependent, or "mid" stops being so, the extra mode needs revisiting.
    """
    golden, outs = native_outputs(case)

    def unique_rows(out) -> int:
        return len(np.unique(out.reshape(-1, out.shape[-1]), axis=0))

    assert unique_rows(golden["low::out"]) == 1, f"{case}: the hybrid low output depends on the input now"
    assert unique_rows(outs["low"]) == 1
    rows = unique_rows(golden["float::out"])
    assert unique_rows(golden["mid::out"]) == rows, f"{case}: the hybrid mid output is not input-dependent"
    assert unique_rows(outs["mid"]) == rows
