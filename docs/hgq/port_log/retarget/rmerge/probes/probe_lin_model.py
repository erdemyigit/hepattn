"""Org Linformer MaskFormer (base+linformer, dim 32) vs ported native: per-key max |diff| on valid slots; controls."""
import copy, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from tests.keras_native import test_pr8_layers as T
from tests.keras_native.porting import perturb_affine_params, port_maskformer
from tests.keras_native.test_maskformer import run_pair

batch, _ = T.clic_dummy_batch()
print("padded node in every event:", (~batch["node_valid"]).any(-1).all(), {k: v.shape for k, v in batch.items()})

def pair(seed=30, sort=True):
    section = T.org_model_section(dim=32)
    tmodel = T.torch_from_section(section, seed)
    perturb_affine_params(tmodel, seed)
    ns = copy.deepcopy(section)
    if not sort:
        ns["model"]["init_args"]["input_sort_field"] = None
    nmodel = T.native_from_section(ns)
    nmodel(batch, training=False)
    port_maskformer(tmodel, nmodel)
    return tmodel, nmodel

out_t, out_n = run_pair(*pair(), batch)
worst = {}
for k in sorted(out_t):
    a, b = out_t[k], np.asarray(out_n[k])
    if a.dtype == bool:
        print(f"{k:50s} bool  mismatches {int((a != b).sum())} / {a.size}")
        continue
    keep = np.isfinite(a) & (a > -1e30)
    d = float(np.abs(a[keep].astype(np.float64) - b[keep]).max()) if keep.any() else 0.0
    worst[k] = d
    print(f"{k:50s} {d:.3e}  (|ref| max {np.abs(a[keep]).max() if keep.any() else 0:.3e})")
_, out_c = run_pair(*pair(sort=False), batch)
k = "final.mask.pflow_node_logit"
keep = out_t[k] > -1e30
print("control no-sort final mask logit diff", float(np.abs(out_t[k][keep].astype(np.float64) - np.asarray(out_c[k])[keep]).max()))
k0 = "layer_0.mask.pflow_node_logit"
keep = out_t[k0] > -1e30
print("control no-sort layer_0 mask logit diff", float(np.abs(out_t[k0][keep].astype(np.float64) - np.asarray(out_c[k0])[keep]).max()))
