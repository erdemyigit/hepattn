"""Critic item 2: hybrid KerasMaskFormer (float) on the org base+linformer config vs org torch, production dims."""
import sys, copy
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import numpy as np, torch, yaml
import orgref
from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.keras.porting import port_maskformer
from hepattn.models.matcher import Matcher

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
inputs, targets = orgref.batch()
tmodel, _ = orgref.torch_model(seed)
shell, _ = orgref.torch_model(seed + 100)  # fresh input nets and tasks for the hybrid (kerasified in place)

def dm(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = dm(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out
raw = dm(*[yaml.safe_load((orgref.CLIC / "configs" / n).read_text()) for n in ("base.yaml", "linformer.yaml")])["model"]["model"]["init_args"]
enc = dict(raw["encoder"]["init_args"]); assert enc.pop("dim") == raw["dim"]
kmodel = KerasMaskFormer(input_nets=shell.input_nets, encoder=enc, decoder=raw["decoder"], tasks=shell.tasks, dim=raw["dim"],
                         input_sort_field=None if mode == "nosort" else raw["input_sort_field"], quant=None).eval()
port_maskformer(tmodel, kmodel)
with torch.no_grad():
    out_t = {k: v.numpy() for k, v in orgref.flat(tmodel(orgref.tt(inputs))).items()}
    out_k = {k: v.detach().cpu().numpy() for k, v in orgref.flat(kmodel(orgref.tt(inputs))).items()}
orgref.compare(out_t, out_k, f"hybrid[{mode}]")
for m in (tmodel, kmodel):
    m.matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)
with torch.no_grad():
    _, _, lt = tmodel.loss(tmodel(orgref.tt(inputs)), orgref.tt(targets))
    _, _, lk = kmodel.loss(kmodel(orgref.tt(inputs)), orgref.tt(targets))
ft = {k: float(v) for k, v in orgref.flat(lt).items()}; fk = {k: float(v) for k, v in orgref.flat(lk).items()}
assert set(ft) == set(fk)
print("LOSSES", len(ft), "max abs", max(abs(ft[k] - fk[k]) for k in ft), "sum", sum(ft.values()), sum(fk.values()))
print("core kinds", sorted({type(m).__name__ for m in kmodel.modules() if "Core" in type(m).__name__ or "Linformer" in type(m).__name__}))
