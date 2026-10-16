"""Critic: org/main torch reference built through jsonargparse (LightningCLI's merge) from base.yaml + linformer.yaml."""
import os, sys, tempfile
from pathlib import Path
import numpy as np, torch, yaml
from jsonargparse import ArgumentParser, ActionConfigFile

KJ = Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
CLIC = KJ / "src/hepattn/experiments/clic"
sys.path.insert(0, str(KJ))  # tests package


def parse(overlay=True, extra=None):
    """(parser, cfg) of model.model from the YAML files, merged as LightningCLI merges --config files."""
    from hepattn.models import MaskFormer
    files = []
    tmp = Path(tempfile.mkdtemp())
    names = ["base.yaml"] + (["linformer.yaml"] if overlay else [])
    for name in names:
        doc = yaml.safe_load((CLIC / "configs" / name).read_text())
        out = tmp / name
        out.write_text(yaml.safe_dump({"model": doc["model"]["model"]}))
        files += ["--config", str(out)]
    if extra:
        out = tmp / "extra.yaml"
        out.write_text(yaml.safe_dump({"model": extra}))
        files += ["--config", str(out)]
    p = ArgumentParser()
    p.add_argument("--config", action=ActionConfigFile)
    p.add_subclass_arguments(MaskFormer, "model")
    cwd = os.getcwd(); os.chdir(CLIC)
    try:
        cfg = p.parse_args(files)
    finally:
        os.chdir(cwd)
    return p, cfg


def torch_model(seed, overlay=True, extra=None):
    p, cfg = parse(overlay, extra)
    torch.manual_seed(seed)
    cwd = os.getcwd(); os.chdir(CLIC)
    try:
        model = p.instantiate_classes(cfg).model
    finally:
        os.chdir(cwd)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if name.endswith("bias") or ".norm" in name and name.endswith("weight"):
                prm.add_(0.05 * torch.randn(prm.shape, generator=g))
    return model.eval(), cfg.model.as_dict()


def batch(n=3, offset=5):
    from tests.keras_native.golden.mf_cases import clic_dummy_batch
    inputs, targets = clic_dummy_batch(num_events=n, offset=offset)
    nv = inputs["node_valid"].sum(-1)
    print("valid nodes per event", nv.tolist(), "max", inputs["node_valid"].shape[-1])
    assert (~inputs["node_valid"]).any(-1).all()
    return inputs, targets


def flat(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def tt(arrays):
    return {k: torch.from_numpy(v.astype(np.int64) if v.dtype == np.int32 else v) for k, v in arrays.items()}


def compare(ref, got, label):
    worst = 0.0
    rows = []
    assert set(ref) == set(got), set(ref) ^ set(got)
    for k in sorted(ref):
        r = np.asarray(ref[k]); g = np.asarray(got[k])
        if r.dtype == bool:
            flips = int((r != g).sum()); rows.append((k, "flips", flips)); worst = max(worst, flips); continue
        r = r.astype(np.float64); g = g.astype(np.float64)
        keep = np.isfinite(r) & (np.abs(r) < 1e30)
        same_pad = bool(np.array_equal(r[~keep], g[~keep]))
        err = float(np.abs(r[keep] - g[keep]).max()) if keep.any() else 0.0
        scale = float(np.abs(r[keep]).max()) if keep.any() else 0.0
        rows.append((k, err, scale, same_pad)); worst = max(worst, err)
    for row in rows:
        print(label, *row)
    print(label, "WORST", worst)
    return worst
