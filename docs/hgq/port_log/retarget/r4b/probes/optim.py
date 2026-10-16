import sys, pathlib, collections
W = pathlib.Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
sys.path.insert(0, str(W / "tests/keras"))
import conftest, torch
torch.compiler.set_stance("force_eager")
import org_model_utils as U
from hepattn.keras_native.masked_linformer import AttendCore, ProjectCore
gen = U.generate_polaris_config(pathlib.Path(sys.argv[1]))
torch.manual_seed(0)
m = U.instantiate(gen, quantized=True)
inputs, targets = U.clic_dummy_batch()
with torch.no_grad(): m(inputs)
m.register_keras_parameters()
cores = [mod for mod in m.modules() if isinstance(mod, (AttendCore, ProjectCore))]
print("cores", len(cores))
params = {id(p) for p in m.parameters()}
cvars = {id(v): v for c in cores for v in c.variables}
print("core vars", len(cvars), "registered", sum(id(v.value) in params for v in cvars.values()))
decay, quant = m.trainable_parameter_groups()
grp = {id(p) for p in decay + quant}
train = [v for v in cvars.values() if v.trainable]
print("trainable", len(train), "in groups", sum(id(v.value) in grp for v in train), "non-trainable", collections.Counter(v.path.rsplit('/',1)[-1] for v in cvars.values() if not v.trainable))
for layer in m.keras_layers():
    if getattr(layer, "_beta", None) is not None: layer._beta.assign(1e-6)
m.train()
before = {id(v): v.value.detach().clone() for v in train}
opt = torch.optim.SGD([{"params": decay}, {"params": quant}], lr=1e-2)
opt.zero_grad()
out = m(inputs)
_, _, ld = m.loss(out, dict(targets))
total = sum(v for l in ld.values() for t in l.values() for v in t.values() if torch.isfinite(v)) + m.quant_losses()
total.backward(); opt.step()
nograd = [v.path for v in train if v.value.grad is None or not v.value.grad.abs().sum() > 0]
moved = [v for v in train if not torch.equal(before[id(v)], v.value.detach())]
print("loss", float(total), "zero/none grad", len(nograd), "moved", len(moved), "of", len(train))
print(collections.Counter(p.rsplit('/',1)[-1] + ":" + p.split('/')[1].split('_')[-1] for p in nograd))
print(nograd[:10])
still = [v for v in train if torch.equal(before[id(v)], v.value.detach())]
for v in still[:12]:
    g = v.value.grad
    print(v.path.split("/", 1)[1][-70:], tuple(v.shape), f"grad max {float(g.abs().max()):.2e}", f"val max {float(v.value.abs().max()):.2e}", f"lr*g/val {float((g.abs()*1e-2).max()):.2e}")
print(collections.Counter(v.path.rsplit('/',1)[-1] for v in still))
