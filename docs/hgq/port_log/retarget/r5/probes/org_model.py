"""5.8 measurement: native org model (dim 32, from_config) vs org torch, outputs, masks, margins, matched loss, controls."""
import sys, math
import numpy as np, torch, keras, jax
sys.path.insert(0, ".")
from tests.keras_native.golden.mf_cases import clic_dummy_batch, torch_model
from tests.keras_native.mf_golden_utils import build_mf, flatten
from tests.keras_native.porting import port_maskformer
from tests.keras_native.objective_utils import flatten as lflat
from hepattn.keras_native import objective as O
from hepattn.keras_native import losses as L
from hepattn.models.matcher import Matcher
from hepattn.models.wrapper import ModelWrapper
from types import SimpleNamespace

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 0
THR = math.log(0.1 / 0.9)
inputs, targets = clic_dummy_batch(2, OFF)
assert (~inputs["node_valid"]).any(-1).all()

def tt(d): return {k: torch.from_numpy(v.astype(np.int64) if v.dtype == np.int32 else v) for k, v in d.items()}

def pair(**kw):
    t = torch_model("org", seed=SEED)
    n = build_mf("org", None, inputs, **kw)
    port_maskformer(t, n)
    return t, n

t, n = pair()
with torch.no_grad():
    out_t_raw = t(tt(inputs))
out_t = {k: v.numpy() for k, v in lflat(out_t_raw).items()}
out_n = flatten(n(inputs, training=False))
errs = {}
for k, ref in sorted(out_t.items()):
    got = np.asarray(out_n[k])
    if ref.dtype == bool:
        print("BOOL", k, int((ref != got).sum()), "frac", ref.mean()); continue
    keep = np.abs(ref) < 1e30
    assert np.array_equal(ref[~keep], got[~keep])
    errs[k] = float(np.abs(ref[keep].astype(np.float64) - got[keep]).max()) if keep.any() else 0.0
for k, v in sorted(errs.items()): print(f"ERR {k} {v:.3e}")
for layer in ("layer_0", "layer_1", "layer_2", "layer_3"):
    k = f"{layer}.mask.pflow_node_logit"; ref = out_t[k].astype(np.float64); keep = ref > -1e30
    print(f"MARGIN {layer} {np.abs(ref[keep]-THR).min():.3e} err {errs[k]:.3e}")

# matched loss
t.matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)
outputs = {k: v for k, v in {kk: vv for kk, vv in flatten(n(inputs, training=False)).items()}.items()}
nat_out = n(inputs, training=False)
with torch.no_grad():
    out2 = t(tt(inputs))
    _, _, losses_t = t.loss(out2, tt(targets))
specs = O.clic_specs()
total, losses, idx = O.matched_loss(nat_out, targets, specs)
ft = {k: float(v) for k, v in lflat(losses_t).items()}
fn = {k: float(np.asarray(v)) for k, v in lflat(losses).items()}
print("LOSSKEYS", len(ft), set(ft) == set(fn))
for k in sorted(ft): print(f"LOSS {k} t={ft[k]:.6f} abs={abs(ft[k]-fn[k]):.3e} rel={abs(ft[k]-fn[k])/abs(ft[k]):.3e}")
total_t = float(ModelWrapper.aggregate_losses(SimpleNamespace(model=t, log=lambda *a, **k: None), losses_t, stage="train"))
print(f"TOTAL t={total_t:.6f} n={float(total):.6f} abs={abs(total_t-float(total)):.3e}")

key = "final.mask.pflow_node_logit"; ref = out_t[key].astype(np.float64); keep = ref > -1e30
for name, kw in (("nosort", {"sort": False}), ("ca_dense_sa", {"decoder_layer_overrides": {"query_update_order": "ca_dense_sa"}})):
    c = build_mf("org", None, inputs, **kw); port_maskformer(t, c)
    oc = flatten(c(inputs, training=False))
    print(f"CONTROL {name} {np.abs(ref[keep] - np.asarray(oc[key])[keep]).max():.3e}")
