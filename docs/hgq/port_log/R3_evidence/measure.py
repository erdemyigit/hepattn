"""R3 measurements: parity maxima and control magnitudes for test_pr8_layers.py tolerances."""

import sys

sys.path.insert(0, ".")
import numpy as np
import tests.keras_native.test_pr8_layers as T
import torch
from tests.keras_native.parity_utils import max_errors


def show(tag, v):
    print(f"{tag:60s} {v:.3e}")


# FastLayerNorm
rng = np.random.default_rng(0)
x = (3.0 * rng.standard_normal((2, 50, 64)) + 1.0).astype(np.float32)
show("fast_layer_norm parity", max_errors(T.FastLayerNorm(64)(torch.from_numpy(x)).detach(), T.make_norm("FastLayerNorm")(x))[0])

# SwiGLU float + swap control (abs, and |ref| max)
for name in T.SWIGLU_CONFIGS:
    t, n = T.swiglu_pair(name)
    xx = T.swiglu_input(name)
    ref = t(torch.from_numpy(xx)).detach().numpy()
    show(f"swiglu float {name} (|ref|max {np.abs(ref).max():.2f})", max_errors(ref, n(xx))[0])
    _, sw = T.swiglu_pair(name, swap=True)
    show(f"  control swap {name}", max_errors(ref, sw(xx))[0])
for name in ("ffn_64", "innet_27_to_64"):
    show(f"swiglu high {name}", T.quantized_error(name, "high"))
    show(f"swiglu low {name}", T.quantized_error(name, "low"))

# encoder legacy
for nl in (1, 2):
    torch.manual_seed(1)
    tenc = T.Encoder(**T.encoder_cfg("legacy", nl)).eval()
    T.perturb_affine_params(tenc, 1)
    nenc = T.ported_encoder(tenc, "legacy", nl)
    x, m = T.padded(2)
    ref = tenc(torch.from_numpy(x), kv_mask=torch.from_numpy(m)).detach()
    show(f"encoder legacy depth{nl - 1}", max_errors(ref, nenc(x, kv_mask=m), m)[0])
    c = T.ported_encoder(tenc, "hybridnorm", nl, check_placement=False)
    show(f"  control hybridnorm depth{nl - 1}", max_errors(ref, c(x, kv_mask=m), m)[0])

# decoder
for d in (0, 1):
    tl, (q, kv, am, km), (qt, kvt) = T.run_decoder_pair(d)
    nl = T.native_decoder_layer(tl, d)
    qn, kvn = nl(q, kv, attn_mask=am, kv_mask=km)
    show(f"decoder q depth{d}", max_errors(qt, qn)[0])
    show(f"decoder kv depth{d}", max_errors(kvt, kvn, km)[0])
    o = T.native_decoder_layer(tl, d, query_update_order="ca_dense_sa")
    show(f"  control order depth{d}", max_errors(qt, o(q, kv, attn_mask=am, kv_mask=km)[0])[0])
    o = T.native_decoder_layer(tl, d, self_attn_norm_kv=False)
    show(f"  control norm_kv=False depth{d}", max_errors(qt, o(q, kv, attn_mask=am, kv_mask=km)[0])[0])

# B=1 torch
for nn_ in (100, 160):
    s = T.one_event(nn_)
    dbl = {k: np.concatenate([v, v]) for k, v in s.items()}
    o1, o2 = T.torch_linformer_outputs(s), T.torch_linformer_outputs(dbl)
    k = "final.mask.pflow_node_logit"
    show(f"B=1 vs B=2 torch {nn_} nodes", float(np.abs(o1[k][0].astype(np.float64) - o2[k][0]).max()))

# model dense
batch, _ = T.clic_dummy_batch()
ot, on = T.run_pair(*T.org_dense_pair(batch), batch)
worst = {}
for k in ot:
    if ot[k].dtype == bool:
        continue
    t, n = ot[k].astype(np.float64), np.asarray(on[k]).astype(np.float64)
    f = ~np.isnan(t)
    worst[k] = (
        np.abs(t[f] - n[f]).max(),
        (np.abs(t[f] - n[f]) / (np.abs(t[f]) + 1e-12)).max(),
        (np.abs(t[f] - n[f]) / (5e-5 + 1e-4 * np.abs(t[f]))).max(),
    )
for k, v in sorted(worst.items(), key=lambda kv: -kv[1][2])[:6]:
    print(f"model {k:50s} abs {v[0]:.2e} rel {v[1]:.2e} frac-of-tol {v[2]:.3f}")
_, oc = T.run_pair(*T.org_dense_pair(batch, query_update_order="ca_dense_sa"), batch)
k = "final.mask.pflow_node_logit"
keep = ot[k] > -1e30
show("  control model ca_dense_sa final mask logit", np.abs(ot[k][keep].astype(np.float64) - np.asarray(oc[k])[keep]).max())
print("  e2e tol at |ref| max", 5e-5 + 1e-4 * np.abs(ot[k][keep]).max())
