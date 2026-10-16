"""Measure native vs hybrid S2 golden differences per (case, mode); also paired deltas. JAX env, run from the worktree root."""
import sys
import numpy as np
import keras
sys.path.insert(0, ".")
from tests.keras_native.golden.cases import CASES, modes
from tests.keras_native.golden_utils import build_from_golden, inputs_of, load_golden, run, build_native, init_state_mismatches

cases = sys.argv[1:] or list(CASES)
for case in cases:
    golden = load_golden(case)
    outs = {}
    for mode in modes(case):
        m, _, _ = build_from_golden(case, mode, golden)
        out = keras.ops.convert_to_numpy(run(case, m, inputs_of(golden))).astype(np.float64)
        ref = golden[f"{mode}::out"].astype(np.float64)
        outs[mode] = out
        d = np.abs(out - ref)
        mis = []
        if mode != "float":
            mod, _ = build_native(case, mode, golden)
            mis = init_state_mismatches(mod, golden, mode)
        print(f"{case} {mode} max|n-h|={d.max():.4e} n_diff={(d>0).sum()} absmax={np.abs(ref).max():.3e} init_mismatch={len(mis)} {mis[:3]}", flush=True)
    hd = {mo: float(np.abs(golden[f'{mo}::out'].astype(np.float64) - golden['float::out']).max()) for mo in ('high', 'low')}
    nd = {mo: float(np.abs(outs[mo] - outs['float']).max()) for mo in ('high', 'low')}
    print(f"{case} DELTA native {nd} hybrid {hd} ratio {nd['low']/max(nd['high'],1e-30):.2f}", flush=True)
