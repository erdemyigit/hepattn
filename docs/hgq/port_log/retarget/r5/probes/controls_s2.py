import sys
import numpy as np, keras
sys.path.insert(0, ".")
from tests.keras_native.test_quantized_golden import LIN_CONTROLS, TOLERANCE
from tests.keras_native.golden_utils import build_from_golden, inputs_of, run
for case, edit in sorted(LIN_CONTROLS.items()):
    m, _, g = build_from_golden(case, "float")
    out = keras.ops.convert_to_numpy(run(case, m, edit(m, inputs_of(g)))).astype(np.float64)
    d = float(np.abs(out - g["float::out"]).max())
    print(f"{case}: control {d:.4e} tol {TOLERANCE[case, 'float']:.3e} ratio {d / TOLERANCE[case, 'float']:.3g}")
