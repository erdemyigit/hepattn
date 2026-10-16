import sys, pathlib
W = pathlib.Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
sys.path.insert(0, str(W / "tests/keras"))
import conftest, torch
torch.compiler.set_stance("force_eager")
import org_model_utils as U
from hepattn.keras.callbacks import BetaScheduler
gen = U.generate_polaris_config(pathlib.Path(sys.argv[1]))
for attn, cfg in (("linformer", gen), ("dense", U.to_dense_attention(gen))):
    torch.manual_seed(0)
    m = U.instantiate(cfg, quantized=True)
    inputs, _ = U.clic_dummy_batch()
    with torch.no_grad(): m(inputs)
    top = {id(l) for l in m.keras_layers() if getattr(l, "_beta", None) is not None}
    allb = [mod for mod in m.modules() if getattr(mod, "_beta", None) is not None]
    allb_u = {id(x): x for x in allb}
    # emulate BetaScheduler at beta 1e-6
    class T: global_step = 10**6
    class P:
        model = m
        def log(self, *a, **k): pass
    BetaScheduler(0.0, 1e-6, 1).on_train_batch_start(T(), P(), None, 0)
    missed = [x for x in allb_u.values() if abs(float(x._beta.value) - 1e-6) > 1e-12]
    m.train()
    with torch.no_grad(): m(inputs)
    own = {id(x): float(x._ebops.value) for x in allb_u.values() if getattr(x, "_ebops", None) is not None}
    tot = sum(own.values()); miss_e = sum(own.get(id(x), 0.0) for x in missed)
    toplevel_read = sum(float(torch.as_tensor(l.ebops)) for l in m.keras_layers() if getattr(l, "ebops", None) is not None)
    print(f"{attn}: beta layers top {len(top)} all {len(allb_u)} missed-by-scheduler {len(missed)} ({sorted({type(x).__name__ for x in missed})}); "
          f"EBOPs total(all leaves own) {tot:.4e}, of which missed {miss_e:.4e} ({miss_e/tot:.4%}); top-level read {toplevel_read:.4e}", flush=True)
