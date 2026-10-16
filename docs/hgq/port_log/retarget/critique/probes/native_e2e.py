"""Critic item 1: native from_config(org base+linformer via jsonargparse) vs org torch, production dims, end to end."""
import sys
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import numpy as np, torch
import orgref
from hepattn.keras_native.maskformer import from_config
from hepattn.keras_native import objective as O
from hepattn.models.matcher import Matcher
from tests.keras_native.porting import port_maskformer

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
inputs, targets = orgref.batch()
tmodel, cfg = orgref.torch_model(seed)
assert "hepattn.keras" not in str(type(tmodel)), type(tmodel)
print("torch encoder attn", type(tmodel.encoder.layers[0].attn.fn).__name__, tmodel.encoder.layers[0].attn.fn.attn_type,
      "proj_k", tuple(tmodel.encoder.layers[0].attn.fn.proj_k.shape) if hasattr(tmodel.encoder.layers[0].attn.fn, "proj_k") else None)
import yaml, copy
def dm(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = dm(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out
print("normalized activation in jsonargparse cfg:", cfg["init_args"]["input_nets"]["init_args"]["modules"][0]["init_args"]["net"]["init_args"]["activation"])
raw = dm(*[yaml.safe_load((orgref.CLIC / "configs" / n).read_text()) for n in ("base.yaml", "linformer.yaml")])["model"]["model"]
init = dict(raw["init_args"])
for t in init["tasks"]["init_args"]["modules"]:
    if "scale_dict_path" in t["init_args"]:
        t["init_args"]["scale_dict_path"] = str(orgref.CLIC / t["init_args"]["scale_dict_path"])
specs_entries = init["tasks"]["init_args"]["modules"]
init["matcher"] = None if mode != "keep_matcher" else init["matcher"]
if mode == "nosort":
    init["input_sort_field"] = None
try:
    nmodel = from_config({"class_path": cfg["class_path"], "init_args": init})
except Exception as e:  # report and retry without unported keys
    print("FROM_CONFIG_RAISED", type(e).__name__, e)
    raise
nmodel(inputs, training=False)
port_maskformer(tmodel, nmodel)
with torch.no_grad():
    out_t = {k: v.numpy() for k, v in orgref.flat(tmodel(orgref.tt(inputs))).items()}
out_n = {k: np.asarray(v) for k, v in orgref.flat(nmodel(inputs, training=False)).items()}
worst = orgref.compare(out_t, out_n, f"native[{mode}]")
# losses: torch with scipy (lap1015 not built), native specs from the same config
tmodel.matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)
with torch.no_grad():
    _, _, lt = tmodel.loss(tmodel(orgref.tt(inputs)), orgref.tt(targets))
specs = O.specs_from_config(specs_entries)
total_n, ln, _ = O.matched_loss(nmodel(inputs, training=False), targets, specs)
ft = {k: float(v) for k, v in orgref.flat(lt).items()}
fn = {k: float(np.asarray(v)) for k, v in orgref.flat(ln).items()}
assert set(ft) == set(fn), set(ft) ^ set(fn)
lw = max(abs(ft[k] - fn[k]) for k in ft)
print("LOSSES", len(ft), "max abs", lw, "sum torch", sum(ft.values()), "native total", float(total_n))
