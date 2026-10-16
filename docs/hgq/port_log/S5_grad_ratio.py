"""Penalty-gradient / task-gradient ratio of a generated Polaris HGQ2 model, per unit beta (hybrid, torch backend).

Usage (hybrid env, KERAS_BACKEND=torch):
    python grad_ratio.py --src <tree>/src --config <polaris_hgq.yaml> --batches batches.npz --out out.json

For each batch: build-once model (jsonargparse, MPflowHGQ, as main_hgq does), materialized and registered. Then
  1. one training-mode forward without grad (WRAP datalanes fit their integer bits, as on any training step);
  2. beta written THE WAY THAT TREE'S CALLBACKS WRITE IT (BetaScheduler.on_train_batch_start): old tree -> top-level
     keras layers only; new tree -> every HGQ2 leaf;
  3. g_task  = grad of the matched task loss (model.loss, sum of all leaves);
     g_q(b)  = grad of model.quant_losses() at beta b = 1 and b = 0; g_pen = g_q(1) - g_q(0) (EBOPs part only, per unit beta);
  4. norms over the quantizer variables (paths with 'quantizer', the bitwidths) and over all trainable variables.
The beta that preserves the old ratio is beta_new = beta_old * s_old / s_new with s = |g_pen| / |g_task|.
"""
# ruff: noqa: PTH123, SIM115, E741, ARG005, SLF001
# Evidence probe for docs/hgq/port_log/S5.md (scratch style: plain open(), short names, Keras/HGQ2 internals).

import argparse
import copy
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("KERAS_BACKEND", "torch")
p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--config", required=True)
p.add_argument("--batches", required=True)
p.add_argument("--out", required=True)
p.add_argument("--scale", required=True)
a = p.parse_args()
sys.path.insert(0, a.src)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from jsonargparse import ArgumentParser  # noqa: E402

import hepattn  # noqa: E402

assert hepattn.__file__.startswith(a.src), hepattn.__file__
from hepattn.experiments.clic.lightning_module_hgq import MPflowHGQ  # noqa: E402
from hepattn.keras import callbacks as C  # noqa: E402
from hepattn.keras import set_keras_default_device  # noqa: E402

set_keras_default_device("cpu")

torch.manual_seed(0)
cfg = yaml.safe_load(open(a.config))
net = cfg["model"]["model"]["init_args"]
for task in net["tasks"]["init_args"]["modules"]:
    if "scale_dict_path" in task.get("init_args", {}):
        task["init_args"]["scale_dict_path"] = a.scale
parser = ArgumentParser(exit_on_error=False)
parser.add_class_arguments(MPflowHGQ, "model", fail_untyped=False)
init = parser.instantiate_classes(parser.parse_object({"model": {**copy.deepcopy(cfg["model"]), "name": cfg["name"]}}))
model = init.model.model

data = np.load(a.batches)
n_batches = int(data["n"])


def batch(i):
    inputs = {k.split("::", 2)[2]: torch.as_tensor(data[k]) for k in data.files if k.startswith(f"{i}::in::")}
    targets = {k.split("::", 2)[2]: torch.as_tensor(data[k]) for k in data.files if k.startswith(f"{i}::tg::")}
    return inputs, targets


with torch.no_grad():
    model.eval()(batch(0)[0])
model.train()
model.register_keras_parameters()

params, names = [], []
seen = set()
for layer in model.keras_layers():
    for var in layer.weights:
        if var.trainable and id(var.value) not in seen and not var.path.endswith("/beta"):
            seen.add(id(var.value))
            params.append(var.value)
            names.append(var.path)
is_q = np.array(["quantizer" in n for n in names])


def write(beta):
    C.BetaScheduler(beta_start=beta, beta_end=beta, warmup_steps=1).on_train_batch_start(
        SimpleNamespace(global_step=0), SimpleNamespace(model=model, log=lambda *x, **k: None), None, 0
    )


def grads(loss):
    g = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if x is None else x for x, p in zip(g, params, strict=True)]


def norm(gs, mask):
    return float(torch.sqrt(sum((g.double() ** 2).sum() for g, m in zip(gs, mask, strict=True) if m)))


def total_ebops_exact():
    walk = getattr(model, "quantized_leaves", None)
    leaves = list(walk()) if walk else [l for l in model.modules() if getattr(l, "_beta", None) is not None]
    return float(sum(int(torch.as_tensor(l._ebops.value)) for l in leaves if getattr(l, "_ebops", None) is not None))


rows = []
for i in range(n_batches):
    inputs, targets = batch(i)
    with torch.no_grad():
        write(0.0)
        model(inputs)
    write(0.0)
    out = model(inputs)
    _, _, ld = model.loss(out, dict(targets))
    task = sum(v for layer in ld.values() for t in layer.values() for v in t.values())
    g_task = grads(task)
    q0 = model.quant_losses()
    g_q0 = grads(q0) if q0.requires_grad else [torch.zeros_like(p) for p in params]
    write(1.0)
    out = model(inputs)
    q1 = model.quant_losses()
    g_q1 = grads(q1)
    g_pen = [x - y for x, y in zip(g_q1, g_q0, strict=True)]
    row = {
        "task_loss": float(task),
        "e_eff_beta_weighted": float(q1) - float(q0),  # EBOPs that beta multiplies, as that tree's quant_losses sums them
        "ebops_total_leaves": total_ebops_exact(),
        "reg": float(q0),
        "task_norm_q": norm(g_task, is_q),
        "pen_norm_q": norm(g_pen, is_q),
        "task_norm_all": norm(g_task, np.ones_like(is_q)),
        "pen_norm_all": norm(g_pen, np.ones_like(is_q)),
        "n_params": len(params),
        "n_quant_params": int(is_q.sum()),
    }
    row["s_q"] = row["pen_norm_q"] / row["task_norm_q"]
    row["s_all"] = row["pen_norm_all"] / row["task_norm_all"]
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump({"src": a.src, "config": a.config, "rows": rows}, open(a.out, "w"), indent=1)
