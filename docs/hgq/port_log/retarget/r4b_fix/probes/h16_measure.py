"""Torch-backend 16-head attend rows vs the JAX golden: max diff, count of differing elements, LSB of the datalane."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1] + "/tests/keras")
import linformer_backend_cases as C
g = np.load(sys.argv[1] + "/tests/keras/golden/linformer_backend.npz")
assert str(g["meta.inputs_sha256"]) == C.inputs_digest()
for spec_name, spec in C.SPECS.items():
    for site in C.PROD_HEAD_SITES:
        for mode, training in C.MODES.items():
            k = C.key(spec_name, site, mode, C.PROD_HEADS)
            got = C.run_case(site, spec, training, C.PROD_HEADS)
            d = np.abs(got - g[k])
            rows = np.argwhere(d > 0)[:, :3]
            print(f"{k:24s} max {d.max():.4g} n_diff {(d>0).sum()} rows {np.unique(rows, axis=0).tolist()[:4]} lsb {2.0**-spec['datalane']['f0']:.3g} size {g[k].size}")
