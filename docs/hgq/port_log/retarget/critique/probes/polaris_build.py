"""Critic item 6: polaris/03_make_config.py output -> hybrid (jsonargparse, as LightningCLI) and native; float = org torch."""
import sys, os, copy, tempfile
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from pathlib import Path
import numpy as np, torch, yaml, keras
import orgref
which, cfg_path = sys.argv[1], sys.argv[2]
gen = yaml.safe_load(open(cfg_path))["model"]["model"]
inputs, targets = orgref.batch()
tmodel, _ = orgref.torch_model(7)
if which == "hybrid":
    from jsonargparse import ArgumentParser, ActionConfigFile
    from hepattn.keras.porting import port_maskformer
    def build(quant):
        spec = copy.deepcopy(gen); spec["init_args"]["quant"] = quant
        f = Path(tempfile.mkdtemp()) / "m.yaml"; f.write_text(yaml.safe_dump({"model": spec}))
        p = ArgumentParser(); p.add_argument("--config", action=ActionConfigFile); p.add_subclass_arguments(torch.nn.Module, "model")
        torch.manual_seed(3)
        return p.instantiate_classes(p.parse_args(["--config", str(f)])).model
    kf = build(None).eval()
    print("hybrid class", type(kf).__name__, "matcher", vars(kf.matcher).get("solver", "?"))
    port_maskformer(tmodel, kf)
    with torch.no_grad():
        out_t = {k: v.numpy() for k, v in orgref.flat(tmodel(orgref.tt(inputs))).items()}
        out_k = {k: v.detach().cpu().numpy() for k, v in orgref.flat(kf(orgref.tt(inputs))).items()}
    orgref.compare(out_t, out_k, "gen-hybrid-float")
    kq = build(gen["init_args"]["quant"]).train()
    out = kq(orgref.tt(inputs))
    total, _, losses = None, None, None
    _, _, losses = kq.loss(out, orgref.tt(targets))
    flat = {k: float(v) for k, v in orgref.flat(losses).items()}
    print("gen-hybrid-quant losses", len(flat), "finite", all(np.isfinite(list(flat.values()))), "sum", sum(flat.values()))
    print("cores", sorted({type(m).__name__ for m in kq.modules() if "Core" in type(m).__name__}))
else:
    from hepattn.keras_native.maskformer import from_config
    from hepattn.keras_native import objective as O
    from tests.keras_native.porting import port_maskformer
    spec = copy.deepcopy(gen); spec["init_args"]["quant"] = None
    nf = from_config(spec); nf(inputs, training=False)
    port_maskformer(tmodel, nf)
    with torch.no_grad():
        out_t = {k: v.numpy() for k, v in orgref.flat(tmodel(orgref.tt(inputs))).items()}
    orgref.compare(out_t, {k: np.asarray(v) for k, v in orgref.flat(nf(inputs, training=False)).items()}, "gen-native-float")
    nq = from_config(copy.deepcopy(gen)); out = nq(inputs, training=True)
    specs = O.specs_from_config(gen["init_args"]["tasks"]["init_args"]["modules"])
    total, losses, _ = O.matched_loss(out, targets, specs)
    print("gen-native-quant total", float(total), "finite", bool(np.isfinite(float(total))), "qlosses", float(sum(keras.ops.convert_to_numpy(l) for l in nq.losses)))
