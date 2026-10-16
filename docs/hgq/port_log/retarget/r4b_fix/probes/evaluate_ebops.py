"""evaluate.* vs callbacks.* after the fix: generated linformer/dense model and the dim-32 test model."""
import pathlib, sys, tempfile, torch
import hepattn.keras; hepattn.keras.set_keras_default_device("cpu")
sys.path.insert(0, sys.argv[1] + "/tests/keras")
import org_model_utils as U
from test_maskformer_parity import clic_dummy_batch, make_keras_model
import test_effective_ebops as TE
from hepattn.keras import evaluate as EV, callbacks as CB

def tasks_ebops(m):
    ids = {id(x) for x in m.quantized_leaves()}
    return sum(CB.leaf_ebops(x) for x in {id(y): y for y in m.tasks.modules() if id(y) in ids}.values())

def report(name, m, inputs):
    ev = EV.total_ebops(m, inputs); cb = CB.total_ebops(m); reg = EV.ebops_by_region(m, inputs)
    eff, floor = EV.effective_ebops(m, inputs); t = tasks_ebops(m)
    print(f"{name}: evaluate.total {ev:.6g} callbacks.total {cb:.6g} regions {sum(reg.values()):.6g} {reg} eff {eff:.6g} "
          f"tasks {t:.4g} eff/(total+tasks)-1 {eff/(cb+t)-1:.3g} eff/total-1 {eff/cb-1:.3g} floor {floor:.4g}")

m = make_keras_model(seed=0, quant=TE.QUANT); inputs, _ = clic_dummy_batch(2)
with torch.no_grad(): m.eval()(inputs)
m.train(); report("dim32", m, inputs)
base = U.generate_polaris_config(pathlib.Path(tempfile.mkdtemp()) / "p.yaml")
for name, cfg in (("linformer", base), ("dense", U.to_dense_attention(base))):
    torch.manual_seed(0); m = U.instantiate(cfg, quantized=True); inputs, _ = U.clic_dummy_batch()
    with torch.no_grad(): m(inputs)
    m.train(); report(name, m, inputs)
