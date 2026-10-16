"""Measured values of test_org_linformer_model_equals_torch: max parity error, margins, both controls."""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from tests.keras_native import test_pr8_layers as T
from tests.keras_native.test_maskformer import flatten_torch, run_pair

batch, _ = T.clic_dummy_batch()
tm, nm = T.org_linformer_pair(batch)
out_t, out_n = run_pair(tm, nm, batch)
errs = {k: float(np.abs(out_t[k][out_t[k] > -1e30].astype(np.float64) - np.asarray(out_n[k])[out_t[k] > -1e30]).max()) for k in out_t if out_t[k].dtype != bool}
print("max err", max(errs.values()), max(errs, key=errs.get))
for i in range(4):
    lk = f"layer_{i}.mask.pflow_node_logit"; t = out_t[lk].astype(np.float64); keep = t > -1e30
    m = np.abs(t[keep] - T.LOGIT_ATTN_THRESHOLD).min(); e = np.abs(t[keep] - np.asarray(out_n[lk])[keep]).max()
    print(lk, "margin", m, "err", e, "ratio", m / e)
key = "final.mask.pflow_node_logit"; keep = out_t[key] > -1e30
tol = T.E2E["atol"] + T.E2E["rtol"] * np.abs(out_t[key][keep]).max()
print("e2e tol on", key, tol)
call = T.NDecoderLayer.call
T.NDecoderLayer.call = lambda self, *a, kv_sort_idx=None, **k: call(self, *a, kv_sort_idx=None, **k)
got = np.asarray(flatten_torch(nm(batch, training=False))[key])
d = np.abs(out_t[key][keep].astype(np.float64) - got[keep]).max()
print("control kv_sort_idx withheld from decoder:", d, "x tol", d / tol)
