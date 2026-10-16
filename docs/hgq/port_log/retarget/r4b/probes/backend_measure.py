import sys, copy, pathlib
W = pathlib.Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
sys.path.insert(0, str(W / "tests/keras"))
import conftest
import numpy as np
import linformer_backend_cases as C
g = np.load(W / "tests/keras/golden/linformer_backend.npz")
assert str(g["meta.inputs_sha256"]) == C.inputs_digest()
def onebit(spec):
    s = copy.deepcopy(spec); s["datalane"]["f0"] -= 1; return s
for spec_name, spec in C.SPECS.items():
    for site in C.SITES:
        fl = C.run_case(site, None, False)
        for mode, training in C.MODES.items():
            ref = g[C.key(spec_name, site, mode)]
            got = C.run_case(site, spec, training)
            d = np.abs(got - ref)
            ob = np.abs(C.run_case(site, onebit(spec), training) - ref).max()
            print(f"{spec_name}.{site}.{mode}: max {d.max():.3e} n {int((d>0).sum())}/{d.size} | float {np.abs(fl-ref).max():.3e} | datalane f0-1 {ob:.3e} | finite {np.isfinite(got).all()}", flush=True)
