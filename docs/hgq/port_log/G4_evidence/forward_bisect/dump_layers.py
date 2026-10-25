# ruff: noqa  (one-off diagnostic, kept as run for docs/hgq/port_log/G4.md "Forward check")
"""Record every keras layer call (path -> outputs) of the hybrid (torch env) or native (jax env) model on the init batch.

dump_layers.py hybrid|native <config.yaml> <ckpt> <hybrid_forward.npz> <out.npz>
"""

import sys

import keras
import numpy as np
import yaml
from keras.src.layers.layer import Layer

side, config, ckpt, fwd, out = sys.argv[1:6]
calls = []
orig = Layer.__call__


def to_np(x):
    try:
        return np.asarray(keras.ops.convert_to_numpy(x))
    except Exception:
        return None


def rec(self, *args, **kwargs):
    res = orig(self, *args, **kwargs)
    flat = [to_np(t) for t in keras.tree.flatten(res)]
    ins = [to_np(t) for t in keras.tree.flatten(args) if hasattr(t, "shape")]
    calls.append((self.path, type(self).__name__, flat, ins))
    return res


with np.load(fwd) as d:
    inputs = {k[4:]: d[k] for k in d.files if k.startswith("in::")}

if side == "hybrid":
    import torch

    from hepattn.keras import set_keras_default_device

    set_keras_default_device("cpu")
    torch.set_float32_matmul_precision("highest")
    from hepattn.experiments.clic.port_to_keras import build_keras_maskformer_from_config, load_keras_state

    cfg = yaml.safe_load(open(config))["model"]["model"]
    model = build_keras_maskformer_from_config(cfg, quant=cfg["init_args"].get("quant")).eval()
    tin = {k: torch.from_numpy(v) for k, v in inputs.items()}
    load_keras_state(model, ckpt, tin)
    Layer.__call__ = rec
    with torch.no_grad():
        outputs = model(tin)
else:
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    from hepattn.keras_native import checkpoint as nckpt
    from hepattn.keras_native.config import load_config

    rc = load_config([config])
    model = rc.build_model()
    model(inputs, training=False)
    (tv, ntv, _), _ = nckpt.load_checkpoint(ckpt, model, None, expect_config_hash=rc.model_hash)
    nckpt.assign_state(model, None, (tv, ntv, []))
    Layer.__call__ = rec
    outputs = model(inputs, training=False)
Layer.__call__ = orig
arrays = {}
for i, (path, cls, flat, ins) in enumerate(calls):
    for j, a in enumerate(flat):
        if a is not None:
            arrays[f"{i:05d}|{path}|{cls}|out{j}"] = a
    for j, a in enumerate(ins):
        if a is not None:
            arrays[f"{i:05d}|{path}|{cls}|in{j}"] = a
np.savez(out, **arrays)
print(side, "calls", len(calls))
