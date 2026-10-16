"""Controls for the 16-head rows: one datalane fractional bit less (high), and the prod spec with 8 -> 7."""
import sys, copy, numpy as np
sys.path.insert(0, sys.argv[1] + "/tests/keras")
import linformer_backend_cases as C
g = np.load(sys.argv[1] + "/tests/keras/golden/linformer_backend.npz")
for spec_name in C.SPECS:
    spec = copy.deepcopy(C.SPECS[spec_name]); spec["datalane"]["f0"] -= 1
    for site in C.PROD_HEAD_SITES:
        for mode, training in C.MODES.items():
            d = np.abs(C.run_case(site, spec, training, C.PROD_HEADS) - g[C.key(spec_name, site, mode, C.PROD_HEADS)])
            print(f"f0-1 {spec_name}.{site}.{mode}: max {d.max():.4g} n_diff {(d>0).sum()}")
