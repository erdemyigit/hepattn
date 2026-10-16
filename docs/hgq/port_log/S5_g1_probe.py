"""G1 gap probe: Keras compiled trainer vs our stateless jax.jit step, with and without buffer donation.

Usage: KERAS_BACKEND=jax python S5_g1_probe.py <path to polaris/14_backend_spike.py> --variant V --batch B [spike args]
  keras      model.compile(jit_compile=True) + train_on_batch (the spike's jax-jit mode)
  stateless  the spike's jax-stateless step (no donation)
  donate     the same step with donate_argnums=(0,) (what Keras' trainer does)
  trainstep  hepattn.keras_native.train_step.TrainStep (donating) with an MSE loss_fn
  passthrough  no timing: count the state outputs of the stateless step that are its inputs unchanged (jaxpr)
Builds the spike graph of polaris/14_backend_spike.py (imported by path), times median ms/step,
and dumps compiled-HLO statistics (copy ops, input/output aliases).
"""
# ruff: noqa: PTH123, SIM115, SLF001
# Evidence probe for docs/hgq/port_log/S5.md (scratch style: plain open(), short names, Keras/HGQ2 internals).

import importlib.util
import json
import re
import statistics
import sys
import time

SPIKE = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("spike", SPIKE)
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)

import argparse  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True)
known, rest = p.parse_known_args()
args = spike.parse_args(rest)

import jax  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import ops  # noqa: E402

keras.utils.set_random_seed(args.seed)
x_np, m_np, y_np = spike.make_data(args, np)
model = spike.build_model(args, keras, ops)
x, m, y = ops.convert_to_tensor(x_np), ops.convert_to_tensor(m_np), ops.convert_to_tensor(y_np)
model((x, m), training=False)
mse = keras.losses.MeanSquaredError()
model.compile(optimizer=keras.optimizers.Adam(args.lr), loss=mse, jit_compile=True)


def hlo_stats(compiled):
    text = compiled.as_text()
    return {
        "copy_ops": len(re.findall(r"= \S+ copy\(", text)),
        "copy_start": len(re.findall(r"copy-start", text)),
        "aliases": text.count("may-alias") + text.count("must-alias"),
        "hlo_chars": len(text),
    }


info = {"variant": known.variant, "batch": args.batch, "device": str(jax.devices()[0])}
if known.variant == "trainstep":
    from hepattn.keras_native.train_step import TrainStep

    model.optimizer.build(model.trainable_variables)

    def mse_loss(outputs, targets):
        loss = mse(targets, outputs)
        return loss, {"mse": loss}

    steps = TrainStep(model, model.optimizer, mse_loss, matmul_precision=None)
    holder = {"st": steps.init_state()}
    info["hlo"] = hlo_stats(steps._train.lower(holder["st"], ((x, m), y), ebops_on=True).compile())

    def step():
        holder["st"], out = steps.train_step(holder["st"], ((x, m), y))
        return float(out.loss)

elif known.variant == "keras":

    def step():
        return float(model.train_on_batch((x, m), y))

    step()
    model.make_train_function()
else:
    opt = model.optimizer
    opt.build(model.trainable_variables)
    state = [[jax.numpy.array(v.value, copy=True) for v in vs] for vs in (model.trainable_variables, model.non_trainable_variables, opt.variables)]

    def loss_fn(tv, ntv, x, m, y):
        out, ntv, losses = model.stateless_call(tv, ntv, (x, m), training=True, return_losses=True)
        return mse(y, out) + sum(losses), ntv

    def train(st, x, m, y):
        tv, ntv, ov = st
        (loss, ntv), grads = jax.value_and_grad(loss_fn, has_aux=True)(tv, ntv, x, m, y)
        tv, ov = opt.stateless_apply(ov, grads, tv)
        return loss, (tv, ntv, ov)

    if known.variant == "passthrough":
        # an output that is an input var unchanged: without donation XLA must copy it into a fresh output buffer
        closed = jax.make_jaxpr(train)(state, x, m, y)
        n_tv, n_ntv = len(state[0]), len(state[1])
        invars = closed.jaxpr.invars[: n_tv + n_ntv + len(state[2])]
        outvars = closed.jaxpr.outvars[1:]  # (loss, (tv, ntv, ov))
        same = [o is i for o, i in zip(outvars, invars, strict=True)]
        info["passthrough"] = {"tv": sum(same[:n_tv]), "ntv": sum(same[n_tv : n_tv + n_ntv]), "ov": sum(same[n_tv + n_ntv :])}
        info["passthrough"]["total"] = sum(same)
        print("RESULT " + json.dumps(info), flush=True)
        sys.exit(0)
    jitted = jax.jit(train, donate_argnums=(0,) if known.variant == "donate" else ())
    compiled = jitted.lower(state, x, m, y).compile()
    info["hlo"] = hlo_stats(compiled)
    import os

    if os.environ.get("HLO_DUMP"):
        open(os.environ["HLO_DUMP"], "w").write(compiled.as_text())
    holder = {"st": state}

    def step():
        loss, holder["st"] = jitted(holder["st"], x, m, y)
        return float(loss)


info["n_state_arrays"] = len(model.trainable_variables) + len(model.non_trainable_variables) + len(model.optimizer.variables)
times = []
for i in range(args.warmup + args.iters):
    t0 = time.perf_counter()
    step()
    if i >= args.warmup:
        times.append((time.perf_counter() - t0) * 1e3)
info["median_ms"] = statistics.median(times)
info["p10_ms"], info["p90_ms"] = np.percentile(times, [10, 90]).tolist()
print("RESULT " + json.dumps(info), flush=True)
