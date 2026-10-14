"""Backend spike: does the Keras backend (torch eager vs JAX jit) change the HGQ2 step time?

Pure keras.ops + HGQ2 (QDense/QEinsum/QSoftmax), no hepattn/torch model code, so the
same graph runs on every Keras backend. Shapes mirror 07_profile.py: 160 nodes + 8
register tokens = 168 encoder tokens, 150 queries cross-attending to the 160 nodes,
the CLIC pflow_hgq.yaml quantizer config, EBOPs penalty in the loss.

Modes (each (mode, batch) runs in its own process so peak RSS is per-config):
  torch-loop      KERAS_BACKEND=torch, eager, hand-written torch loop + torch.optim.Adam
                  (what Lightning drives in production, minus Lightning)
  torch-eager     KERAS_BACKEND=torch, model.compile(jit_compile=False) + train_on_batch
  jax-jit         KERAS_BACKEND=jax,   model.compile(jit_compile=True)  + train_on_batch
  jax-eager       KERAS_BACKEND=jax,   model.compile(jit_compile=False) + train_on_batch
  jax-stateless   KERAS_BACKEND=jax,   one jax.jit'ed pure step (stateless_call +
                  optimizer.stateless_apply), state kept on device between steps

Single config:  KERAS_BACKEND=jax python 14_backend_spike.py --mode jax-jit --batch 32
Sweep:          python 14_backend_spike.py --sweep --torch-python T --jax-python J --out res.json
"""
# ruff: noqa: PLC0415, SLF001
# PLC0415: keras/torch/jax import only after KERAS_BACKEND is checked, per mode.
# SLF001: reads HGQ2/keras internals (_ebops, _flatten_layers) on purpose, as checks.

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODES = {
    "torch-loop": "torch",
    "torch-eager": "torch",
    "jax-jit": "jax",
    "jax-eager": "jax",
    "jax-stateless": "jax",
}

QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default=None)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--attn", choices=["sdpa", "linformer"], default="sdpa")
    p.add_argument("--linformer-k", type=int, default=256)
    p.add_argument("--nodes", type=int, default=160)
    p.add_argument("--registers", type=int, default=8)
    p.add_argument("--queries", type=int, default=150)
    p.add_argument("--in-dim", type=int, default=8)
    p.add_argument("--out-dim", type=int, default=8)
    p.add_argument("--beta0", type=float, default=1e-12)  # 07_profile.py value; penalty ~ task loss
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    # sweep driver
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--batches", default="4,32")
    p.add_argument("--torch-python", default=sys.executable)
    p.add_argument("--jax-python", default=sys.executable)
    p.add_argument("--out", default=None)
    p.add_argument("--timeout", type=int, default=3600)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- sweep driver


def run_sweep(args) -> None:
    """Run every (mode, batch) in a fresh interpreter and tabulate the JSON each prints."""
    passthrough = []
    for k in (
        "dim",
        "heads",
        "enc_layers",
        "dec_layers",
        "attn",
        "linformer_k",
        "nodes",
        "registers",
        "queries",
        "in_dim",
        "out_dim",
        "beta0",
        "lr",
        "warmup",
        "iters",
        "seed",
    ):
        passthrough += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
    results = []
    for mode in args.modes.split(","):
        backend = MODES[mode]
        py = args.torch_python if backend == "torch" else args.jax_python
        for batch in (int(b) for b in args.batches.split(",")):
            env = {**os.environ, "KERAS_BACKEND": backend}
            cmd = [py, str(Path(__file__).resolve()), "--mode", mode, "--batch", str(batch), *passthrough]
            print(f"### {mode} batch={batch}: {' '.join(cmd)}", flush=True)
            try:
                cp = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired:
                results.append({"mode": mode, "batch": batch, "error": f"timeout {args.timeout}s"})
                continue
            line = next((ln for ln in reversed(cp.stdout.splitlines()) if ln.startswith("RESULT ")), None)
            if cp.returncode != 0 or line is None:
                err = (cp.stderr or cp.stdout).strip().splitlines()[-15:]
                print("\n".join(err), flush=True)
                results.append({"mode": mode, "batch": batch, "error": "\n".join(err)})
                continue
            res = json.loads(line[len("RESULT ") :])
            print(json.dumps(res, indent=1), flush=True)
            results.append(res)
    if args.out:
        with Path(args.out).open("w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
    print_table(results)


def dev_mem(r) -> str:
    return "" if r.get("peak_dev_gb") is None else f" (dev {r['peak_dev_gb']:.2f})"


def print_table(results) -> None:
    ok = [r for r in results if "error" not in r]
    cols = [
        "mode",
        "batch",
        "median ms/step",
        "p10-p90 ms",
        "ms/event",
        "peak RSS GB",
        "first-step s",
        "loss0 -> lossN",
        "task0 -> taskN",
        "bits changed",
        "EBOPs",
        "penalty check",
    ]
    print("\n| " + " | ".join(cols) + " |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        print(
            f"| {r['mode']} | {r['batch']} | {r['median_ms']:.1f} | {r['p10_ms']:.1f}-{r['p90_ms']:.1f} | "
            f"{r['median_ms'] / r['batch']:.2f} | {r['peak_rss_gb']:.2f}{dev_mem(r)} | {r['first_step_s']:.1f} | "
            f"{r['loss_first']:.4g} -> {r['loss_last']:.4g} | {r['task_before']:.4g} -> {r['task_after']:.4g} | "
            f"{r['bits']['n_changed']}/{r['bits']['n_total']} (mean {r['bits']['mean_before']:.4f} -> {r['bits']['mean_after']:.4f}) | "
            f"{r['ebops_total']:.4g} | {r['penalty_check']['ok']} |"
        )
    for r in results:
        if "error" in r:
            print(f"| {r['mode']} | {r['batch']} | ERROR | {r['error'].splitlines()[-1][:120]} |")
    # fixed cost / per-event slope from the two-point (or least-squares) fit per mode
    print("\n| mode | intercept ms (fixed) | slope ms/event | fixed share at largest batch |")
    print("|---|---|---|---|")
    for mode in dict.fromkeys(r["mode"] for r in ok):
        pts = sorted((r["batch"], r["median_ms"]) for r in ok if r["mode"] == mode)
        if len(pts) < 2:
            continue
        n = len(pts)
        mx = sum(b for b, _ in pts) / n
        my = sum(t for _, t in pts) / n
        slope = sum((b - mx) * (t - my) for b, t in pts) / sum((b - mx) ** 2 for b, _ in pts)
        icpt = my - slope * mx
        print(f"| {mode} | {icpt:.1f} | {slope:.2f} | {icpt / pts[-1][1]:.0%} |")


# --------------------------------------------------------------------------- single config


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20  # bytes on macOS, KiB on Linux


def peak_device_gb(backend: str) -> float | None:
    """Peak accelerator memory (None on CPU, where peak RSS is the number that matters)."""
    if backend == "torch":
        import torch

        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    import jax

    dev = jax.devices()[0]
    stats = dev.memory_stats() if dev.platform == "gpu" else None
    return stats["peak_bytes_in_use"] / 2**30 if stats else None


def build_model(args, keras, ops):
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.layers import QDense, QEinsum, QSoftmax

    def qdense(units, act=None, bias=True, name=None):
        return QDense(units, activation=act, use_bias=bias, name=name)

    class Attn(keras.layers.Layer):
        """Mirror of hepattn.keras.attention.KerasAttention / linformer.KerasLinformerAttention."""

        def __init__(self, dim, heads, kind, kv_len, lin_k, name):
            super().__init__(name=name)
            self.h, self.dh, self.kind, self.kv_len, self.lin_k = heads, dim // heads, kind, kv_len, lin_k
            bias = kind == "sdpa"  # linformer q/k/v are bias-free, as in KerasLinformerAttention
            self.q = qdense(dim, bias=bias, name=f"{name}_q")
            self.k = qdense(dim, bias=bias, name=f"{name}_k")
            self.v = qdense(dim, bias=bias, name=f"{name}_v")
            self.o = qdense(dim, name=f"{name}_o")
            if kind == "linformer":
                init = keras.initializers.RandomUniform(-(lin_k**-0.5), lin_k**-0.5)
                # float boundary, exactly as the nn.Parameter proj_k/proj_v in linformer.py
                self.proj_k = self.add_weight(shape=(kv_len, lin_k), initializer=init, name="proj_k")
                self.proj_v = self.add_weight(shape=(kv_len, lin_k), initializer=init, name="proj_v")
                self.seq_k = QEinsum("bnd,nk->bkd", name=f"{name}_seqk")
                self.seq_v = QEinsum("bnd,nk->bkd", name=f"{name}_seqv")
                self.scores = QEinsum("bhnd,bhkd->bhnk", name=f"{name}_scores")
                self.values = QEinsum("bhnk,bhkd->bhnd", name=f"{name}_values")
            else:
                self.scores = QEinsum("bhqd,bhkd->bhqk", name=f"{name}_scores")
                self.values = QEinsum("bhqk,bhkd->bhqd", name=f"{name}_values")
            self.softmax = QSoftmax(axis=-1, name=f"{name}_softmax")

        def heads(self, x):
            b, s = ops.shape(x)[0], ops.shape(x)[1]
            return ops.transpose(ops.reshape(x, (b, s, self.h, self.dh)), (0, 2, 1, 3))

        def call(self, x, kv, kv_mask, training=None):
            q = self.q(x, training=training)
            k = self.k(kv, training=training)
            v = self.v(kv, training=training)
            keep = ops.expand_dims(ops.expand_dims(kv_mask, 1), 1)  # (B,1,1,K): hepattn True=attend
            if self.kind == "linformer":
                k = self.seq_k([k, self.proj_k], training=training)
                v = self.seq_v([v, self.proj_v], training=training)
                pad = ops.zeros((ops.shape(keep)[0], 1, 1, self.lin_k - self.kv_len), dtype="bool")
                keep = ops.concatenate([keep, pad], axis=-1)  # projected-space mask, k >= kv_len
            q, k, v = self.heads(q), self.heads(k), self.heads(v)
            q = q * (self.dh**-0.5)
            s = self.scores([q, k], training=training)
            a = self.softmax(s, training=training, mask=keep)
            out = self.values([a, v], training=training)
            b, s_len = ops.shape(out)[0], ops.shape(out)[2]
            out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, s_len, self.h * self.dh))
            return self.o(out, training=training)

    class MLP(keras.layers.Layer):
        def __init__(self, dim, name):
            super().__init__(name=name)
            self.up = qdense(2 * dim, act="silu", name=f"{name}_up")
            self.down = qdense(dim, name=f"{name}_down")

        def call(self, x, training=None):
            return self.down(self.up(x, training=training), training=training)

    class Spike(keras.Model):
        def __init__(self):
            super().__init__(name="spike")
            d, n_tok = args.dim, args.nodes + args.registers
            self.embed = qdense(d, name="embed")
            self.registers = self.add_weight(shape=(args.registers, d), initializer="random_normal", name="registers")
            self.query_init = self.add_weight(shape=(args.queries, d), initializer="random_normal", name="query_init")
            ln = keras.layers.LayerNormalization  # float norms: same conversion boundary as production
            self.enc = [
                (ln(name=f"e{i}_n1"), Attn(d, args.heads, args.attn, n_tok, args.linformer_k, f"e{i}_sa"), ln(name=f"e{i}_n2"), MLP(d, f"e{i}_mlp"))
                for i in range(args.enc_layers)
            ]
            self.dec = [
                (
                    ln(name=f"d{i}_n1"),
                    Attn(d, args.heads, args.attn, args.nodes, args.linformer_k, f"d{i}_ca"),
                    ln(name=f"d{i}_n2"),
                    Attn(d, args.heads, args.attn, args.queries, args.linformer_k, f"d{i}_sa"),
                    ln(name=f"d{i}_n3"),
                    MLP(d, f"d{i}_mlp"),
                )
                for i in range(args.dec_layers)
            ]
            self.head = qdense(args.out_dim, name="head")

        def call(self, inputs, training=None):
            x, node_mask = inputs
            b = ops.shape(x)[0]
            x = self.embed(x, training=training)
            reg = ops.broadcast_to(ops.expand_dims(self.registers, 0), (b, args.registers, args.dim))
            x = ops.concatenate([x, ops.cast(reg, x.dtype)], axis=1)
            tok_mask = ops.concatenate([node_mask, ops.ones((b, args.registers), dtype="bool")], axis=1)
            for n1, sa, n2, mlp in self.enc:
                h = n1(x)
                x = x + sa(h, h, tok_mask, training=training)
                x = x + mlp(n2(x), training=training)
            mem = x[:, : args.nodes]  # registers are stripped before the decoder, as in hepattn
            q = ops.broadcast_to(ops.expand_dims(self.query_init, 0), (b, args.queries, args.dim))
            q = ops.cast(q, x.dtype)
            q_all = ops.ones((b, args.queries), dtype="bool")
            for n1, ca, n2, sa, n3, mlp in self.dec:
                q = q + ca(n1(q), mem, node_mask, training=training)
                h = n2(q)
                q = q + sa(h, h, q_all, training=training)
                q = q + mlp(n3(q), training=training)
            return self.head(q, training=training)

    # quantized layers read their config from the scopes active at CONSTRUCTION
    # (hepattn/keras/factory.py LayerFactory.scopes); they build lazily on the first call
    with (
        QuantizerConfigScope(place="weight", **QUANT["weight"]),
        QuantizerConfigScope(place="datalane", **QUANT["datalane"]),
        QuantizerConfigScope(place="table", **QUANT["table"]),
        LayerConfigScope(enable_ebops=True, beta0=args.beta0),
    ):
        return Spike()


def make_data(args, np):
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.batch, args.nodes, args.in_dim)).astype("float32")
    n_valid = rng.integers(args.nodes // 2, args.nodes + 1, size=args.batch)
    mask = np.arange(args.nodes)[None, :] < n_valid[:, None]
    x = np.where(mask[..., None], x, 0.0).astype("float32")
    y = rng.normal(size=(args.batch, args.queries, args.out_dim)).astype("float32")
    return x, mask, y


def qlayers(model):
    from hgq.layers.core.base import QLayerBase

    return [layer for layer in model._flatten_layers(include_self=False) if isinstance(layer, QLayerBase)]


def bit_vars(model):
    """Trainable quantizer bitwidth variables: kbi -> b/i, kif -> i/f (named by HGQ2)."""
    return [v for v in model.trainable_variables if v.path.split("/")[-1] in {"b", "i", "f"}]


def penalty_terms(model, x, m, y, mse, np, keras):
    """Eager training-mode forward; returns its task loss and add_loss terms (beta * EBOPs).

    Read from model.losses, not the layers' _ebops variables: HGQ2 stores _ebops as
    uint32 (hgq/layers/core/base.py build()); on the JAX backend that saturates at
    2**32-1 for the softmax exp tables (~3.7e13 here) while the float value still
    enters the loss. The torch backend stored the exact value in the same test.
    """
    task = float(keras.ops.convert_to_numpy(mse(y, model((x, m), training=True))))
    terms = np.array([float(keras.ops.convert_to_numpy(t)) for t in model.losses])
    sat = sum(
        int(keras.ops.convert_to_numpy(layer._ebops)) == 2**32 - 1 for layer in qlayers(model) if layer.enable_ebops and layer._ebops is not None
    )
    return task, terms, sat


def run_one(args) -> None:
    backend = MODES[args.mode]
    if os.environ.get("KERAS_BACKEND") != backend:
        raise SystemExit(f"--mode {args.mode} needs KERAS_BACKEND={backend}, got {os.environ.get('KERAS_BACKEND')}")
    import keras
    import numpy as np
    from keras import ops

    keras.utils.set_random_seed(args.seed)
    info = {"mode": args.mode, "batch": args.batch, "backend": keras.backend.backend(), "keras": keras.__version__}
    if backend == "torch":
        import torch

        info.update(torch=torch.__version__, threads=torch.get_num_threads(), device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        import jax

        info.update(jax=jax.__version__, device=str(jax.devices()[0]))

    x_np, m_np, y_np = make_data(args, np)
    model = build_model(args, keras, ops)
    x, m, y = ops.convert_to_tensor(x_np), ops.convert_to_tensor(m_np), ops.convert_to_tensor(y_np)
    model((x, m), training=False)  # lazy HGQ2 build from the true static shapes
    # with the initial weights: the task loss and penalty step 0's forward must reproduce
    mse = keras.losses.MeanSquaredError()
    task_train0, terms, n_sat = penalty_terms(model, x, m, y, mse, np, keras)
    pen0 = float(terms.sum())
    info["ebops_total"] = pen0 / args.beta0  # every beta starts at beta0
    info["n_penalty_terms"] = len(terms)
    info["n_ebops_var_saturated_uint32"] = n_sat
    info["n_qlayers"] = len(qlayers(model))
    info["n_params"] = int(sum(np.prod(v.shape) for v in model.trainable_variables))

    bvars = bit_vars(model)
    bits_before = [np.array(ops.convert_to_numpy(v)) for v in bvars]

    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss=mse, jit_compile=args.mode == "jax-jit")
    task_before = float(model.test_on_batch((x, m), y))  # training=False: no EBOPs term

    losses, times = [], []
    step = make_step(args, model, ops, x, m, y, mse)
    n = args.warmup + args.iters
    t_first = None
    for i in range(n):
        t0 = time.perf_counter()
        loss = step()  # returns a host float: every mode syncs the device once per step
        dt = time.perf_counter() - t0
        if i == 0:
            t_first = dt
            # EBOPs/penalty from step 0's forward (pre-update weights) vs loss - task loss:
            # if the penalty did not enter the loss this difference would be 0
            # step 0 loss minus the eager training-mode task loss must equal the eager penalty;
            # it would be 0 if EBOPs never reached the loss. (training=False changes the
            # quantized forward, so test_on_batch's task loss is not the reference here.)
            diff = loss - task_train0
            info["penalty_check"] = {
                "loss_step0_minus_task_train": diff,
                "sum_beta_ebops": pen0,
                "rel_err": abs(diff - pen0) / pen0,
                "ok": bool(pen0 > 0 and abs(diff - pen0) <= 1e-3 * abs(loss)),
            }
        losses.append(loss)
        if i >= args.warmup:
            times.append(dt * 1e3)
    if args.mode == "jax-stateless":
        step.sync_back()
    task_after = float(model.test_on_batch((x, m), y))

    bits_after = [np.array(ops.convert_to_numpy(v)) for v in bvars]
    changed = [not np.array_equal(a, b) for a, b in zip(bits_before, bits_after, strict=True)]
    flat_b = np.concatenate([a.ravel() for a in bits_before])
    flat_a = np.concatenate([a.ravel() for a in bits_after])
    info["bits"] = {
        "n_total": len(bvars),
        "n_changed": int(sum(changed)),
        "mean_before": float(flat_b.mean()),
        "mean_after": float(flat_a.mean()),
        "max_abs_delta": float(np.abs(flat_a - flat_b).max()),
        "example": f"{bvars[0].path}: {flat_b[:3].tolist()} -> {np.ravel(bits_after[0])[:3].tolist()}",
    }
    q = statistics.quantiles(times, n=10)
    info.update(
        median_ms=statistics.median(times),
        p10_ms=q[0],
        p90_ms=q[-1],
        first_step_s=t_first,
        loss_first=losses[0],
        loss_last=losses[-1],
        task_before=task_before,
        task_after=task_after,
        peak_rss_gb=peak_rss_gb(),
        peak_dev_gb=peak_device_gb(backend),
        warmup=args.warmup,
        iters=args.iters,
    )
    print("RESULT " + json.dumps(info), flush=True)


def make_step(args, model, ops, x, m, y, mse):
    if args.mode in {"torch-eager", "jax-jit", "jax-eager"}:

        def step():
            return float(model.train_on_batch((x, m), y))

        return step

    if args.mode == "torch-loop":
        import torch

        # keras-torch variables are torch Parameters (KerasVariable.value), as Lightning sees them
        params = [v.value for v in model.trainable_variables]
        opt = torch.optim.Adam(params, lr=args.lr)

        def step():
            out = model((x, m), training=True)
            loss = mse(y, out) + ops.sum(ops.stack(model.losses))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return float(loss.detach())

        return step

    # jax-stateless: the whole step is one XLA program; variables stay as jax arrays
    import jax

    opt = model.optimizer
    opt.build(model.trainable_variables)
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    ov = [v.value for v in opt.variables]

    def loss_fn(tv, ntv, x, m, y):
        out, ntv, losses = model.stateless_call(tv, ntv, (x, m), training=True, return_losses=True)
        return mse(y, out) + sum(losses), ntv

    @jax.jit
    def train(tv, ntv, ov, x, m, y):
        (loss, ntv), grads = jax.value_and_grad(loss_fn, has_aux=True)(tv, ntv, x, m, y)
        tv, ov = opt.stateless_apply(ov, grads, tv)
        return loss, tv, ntv, ov

    state = {"tv": tv, "ntv": ntv, "ov": ov}

    def step():
        loss, state["tv"], state["ntv"], state["ov"] = train(state["tv"], state["ntv"], state["ov"], x, m, y)
        return float(loss)

    def sync_back():
        for v, a in zip(model.trainable_variables, state["tv"], strict=True):
            v.assign(a)
        for v, a in zip(model.non_trainable_variables, state["ntv"], strict=True):
            v.assign(a)

    step.sync_back = sync_back
    return step


if __name__ == "__main__":
    a = parse_args()
    if a.sweep:
        run_sweep(a)
    elif a.mode:
        run_one(a)
    else:
        raise SystemExit("pass --mode or --sweep")
