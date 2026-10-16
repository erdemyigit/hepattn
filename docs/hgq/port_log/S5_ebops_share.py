"""Native JAX: EBOPs at init (read-out) and the EBOPs share of the COMPILED train step, on a generated Polaris config.

Usage (jax env, PYTHONPATH=<worktree>/src):
    python ebops_share.py --config polaris_hgq.yaml --batches batches.npz --out out.json [--dim D --batch-reps R]

The model is from_config(cfg["model"]) (production dims unless --dim overrides the width, as 15_g2_forward does not);
optimizer from the config (AdamW, OneCycle, clip, accum); matched loss from the config (n_jobs from the matcher block).
EBOPs on/off executables are timed interleaved (median of --iters each, after --warmup).
"""
# ruff: noqa: PTH123, SIM115, PLC2701
# Evidence probe for docs/hgq/port_log/S5.md (scratch style: plain open(), short names, Keras/HGQ2 internals).

import argparse
import json
import statistics
import time

import jax
import numpy as np
import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--batches", required=True)
p.add_argument("--out", required=True)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--iters", type=int, default=8)
p.add_argument("--beta", type=float, default=2.0**-40)
p.add_argument("--precision", default="highest")
p.add_argument("--scale", required=True)
a = p.parse_args()

from hepattn.keras_native.maskformer import _modules, from_config  # noqa: E402
from hepattn.keras_native.optim import optimizer_from_config  # noqa: E402
from hepattn.keras_native.train_step import TrainStep, maskformer_loss_from_config  # noqa: E402

cfg = yaml.safe_load(open(a.config))
init = cfg["model"]["model"]["init_args"]
for task in _modules(init["tasks"]):
    if "scale_dict_path" in task.get("init_args", {}):
        task["init_args"]["scale_dict_path"] = a.scale
data = np.load(a.batches)


def batch(i):
    def get(tag):
        out = {k.split("::", 2)[2]: data[k] for k in data.files if k.startswith(f"{i}::{tag}::")}
        return {k: v.astype(np.int32) if v.dtype == np.int64 else v for k, v in out.items()}

    return get("in"), get("tg")


batches = [batch(i) for i in range(int(data["n"]))]
model = from_config(cfg["model"])
model(batches[0][0], training=False)
opt = optimizer_from_config(cfg["model"], cfg["trainer"], model, total_steps=10_000)
steps = TrainStep(model, opt, maskformer_loss_from_config(init), matmul_precision=a.precision)
state = steps.set_beta(steps.init_state(), a.beta)
info = {
    "config": a.config,
    "beta": a.beta,
    "device": str(jax.devices()[0]),
    "n_leaves": len(steps.ebops_leaves),
    "n_trainable": len(model.trainable_variables),
    "batch": int(batches[0][0]["node_valid"].shape[0]),
}

compile_s = {}
for on in (True, False):
    t0 = time.perf_counter()
    state, out = steps.train_step(state, batches[0], ebops_on=on)
    jax.block_until_ready(out.loss)
    compile_s[on] = time.perf_counter() - t0
info["first_step_s"] = {"on": compile_s[True], "off": compile_s[False]}

# EBOPs at init: a fresh state, one active step's forward per batch
ebops = []
for b in batches:
    fresh = steps.set_beta(steps.init_state(), a.beta)
    _, out = steps.train_step(fresh, b, ebops_on=True)
    ebops.append(steps.ebops(out, a.beta))
info["ebops_init_per_batch"] = ebops

times = {True: [], False: []}
for i in range(a.warmup + a.iters):
    for on in (True, False):
        b = batches[i % len(batches)]
        t0 = time.perf_counter()
        state, out = steps.train_step(state, b, ebops_on=on)
        float(out.loss)
        if i >= a.warmup:
            times[on].append((time.perf_counter() - t0) * 1e3)
info["median_ms"] = {"on": statistics.median(times[True]), "off": statistics.median(times[False])}
info["times_ms"] = {"on": times[True], "off": times[False]}
info["ebops_share"] = 1 - info["median_ms"]["off"] / info["median_ms"]["on"]
info["compile_counts"] = steps.compile_counts()
print(json.dumps({k: v for k, v in info.items() if k != "times_ms"}), flush=True)
json.dump(info, open(a.out, "w"), indent=1)
