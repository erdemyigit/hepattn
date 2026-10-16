import sys
import numpy as np, keras
sys.path.insert(0, ".")
from tests.keras_native.mf_golden_utils import build_mf_from_golden, mf_inputs, mf_outputs, LAYER0_LOGIT, flatten
nat, hyb = {}, {}
for mode in ("float", "high", "low"):
    m, g = build_mf_from_golden("mf_org", mode)
    nat[mode] = flatten(m(mf_inputs(g), training=False))[LAYER0_LOGIT].astype(np.float64)
    hyb[mode] = mf_outputs(g, mode)[LAYER0_LOGIT].astype(np.float64)
keep = np.isfinite(nat["float"]) & (np.abs(nat["float"]) < 1e30)
for mode in ("high", "low"):
    dn = np.abs(nat[mode][keep] - nat["float"][keep]).max(); dh = np.abs(hyb[mode][keep] - hyb["float"][keep]).max()
    print(f"{mode} native {dn!r} hybrid {dh!r} diff {abs(dn-dh):.3e}")
