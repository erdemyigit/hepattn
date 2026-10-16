"""Gate G2 (KERAS_JAX_PLAN.md section 4, S3): jitted native forward + backward vs the hybrid's bare-loop step.

The model is the org/main model that Polaris trains: the config polaris/03_make_config.py generates (base.yaml +
linformer.yaml: masked Linformer k = 64 in every attention, phi sorting, PR #8 layers, SwiGLU), read once by the sweep
driver and fed to both frameworks (``model_config``). Its HGQ2 quantizers are the generated ``quant`` block with
``ebops.beta0`` = 1e-12 (the generated config starts at 0 and lets the beta callbacks raise it; a bare step has no
callbacks, so the EBOPs penalty would be 0). fp32 in every arm, EBOPs on in every step, one GPU. Each (mode, batch)
runs in its own process; the sweep driver runs the JAX batches first and saves the JSON after every run.

Modes:
  jax               KERAS_BACKEND=jax. ``hepattn.keras_native.maskformer.from_config``; one
                    jax.jit(jax.value_and_grad(...)) of stateless_call(training=True, return_losses=True) with a
                    surrogate loss (sum of output terms + the HGQ2 losses). The returned non-trainable state (EBOPs, WRAP
                    ranges) is carried to the next step. No optimizer, no matching, no task losses.
  hybrid-07         KERAS_BACKEND=torch. polaris/07_profile.py's step without its bf16 autocast: KerasMaskFormer
                    forward, model.loss (host Hungarian matching, scipy + task losses), backward. EBOPs is computed in
                    the forward but, as in 07, is not part of the backward.
  hybrid-surrogate  KERAS_BACKEND=torch. The same forward + backward as the jax mode: surrogate + the HGQ2 losses
                    of each keras layer ONCE. The hybrid's quant_losses() walks model.tasks and model.decoder.tasks
                    (the same modules) and so counts the task heads twice (docs/hgq/port_log/S3.md deviation 9); this
                    arm sums over the unique layers instead, which equals the per-layer terms of the jax arm.
                    hybrid-07 keeps 07's quant_losses() call, so its recorded penalty is the double-counted sum.
G2 compares jax with hybrid-07 at the same batch (plan: <= 0.5x); hybrid-surrogate is the like-for-like reference.
Torch fp32 OOMed at batch 32 on the pre-retarget model (job 7746577), so G2 is decided at batch 16, fp32: median jax /
median hybrid-07 <= 0.5. Batch 8 is the second point of the same comparison, and JAX batch 32 is reported for scaling.

``--dim``, ``--heads``, ``--enc-layers``, ``--dec-layers`` and ``--linformer-k`` shrink the generated model (for a
local check at tiny dims); left unset they keep the generated values (dim 256, 16 heads, 6 + 4 layers, k = 64). The
node count is fixed at 160 (+ 8 registers), the ``linformer_seq_len`` of the config.

Single run:  KERAS_BACKEND=jax python 15_g2_forward.py --mode jax --batch 16
Sweep:       python 15_g2_forward.py --sweep --torch-python T --jax-python J --out logs/g2_<jobid>.json
"""
# ruff: noqa: PLC0415, SLF001
# PLC0415: keras / torch / jax import only after the per-mode KERAS_BACKEND check.
# SLF001: reads jax's _cache_size() as a recompile check.

import argparse
import copy
import importlib
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCALE_DICT = str(REPO / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml")
MODES = {"jax": "jax", "hybrid-07": "torch", "hybrid-surrogate": "torch"}
MAKE_CONFIG = REPO / "polaris/03_make_config.py"
EBOPS_BETA0 = 1.0e-12  # 07_profile.py's beta0: a nonzero EBOPs penalty in every step
FIELDS = ["e", "pt", "eta", "sinphi", "cosphi"]
NODES, QUERIES = 160, 150  # the generated config: data.max_nodes / linformer_seq_len, decoder num_queries
SENTINEL = 1e30  # |x| above this is the finfo.min mask-logit padding


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--config", default="", help="generated Polaris config (YAML); empty = run 03_make_config.py")
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--enc-layers", type=int, default=None)
    p.add_argument("--dec-layers", type=int, default=None)
    p.add_argument("--linformer-k", type=int, default=None)
    p.add_argument("--data-file", default="", help="CLIC ROOT file; empty = CLICDataset dummy data")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    # sweep driver
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--modes", default="jax,hybrid-07,hybrid-surrogate")
    p.add_argument("--jax-batches", default="8,16,32")
    p.add_argument("--hybrid-batches", default="8,16")  # torch fp32 OOMed at batch 32 on a 40 GB A100 (job 7746577)
    p.add_argument("--torch-python", default=sys.executable)
    p.add_argument("--jax-python", default=sys.executable)
    p.add_argument("--out", default=None)
    p.add_argument("--timeout", type=int, default=3000)
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------------- config
def generate_config(out: Path) -> Path:
    """Run polaris/03_make_config.py (defaults: base.yaml + linformer.yaml) into ``out``."""
    env = {**os.environ, "HGQ_CONFIG_OUT": str(out), "HGQ_ATTN": "linformer"}
    subprocess.run([sys.executable, str(MAKE_CONFIG)], env=env, check=True, capture_output=True, text=True)
    return out


def model_config(args) -> dict:
    """``model.model.init_args`` of the generated config (KerasMaskFormer), with the size overrides of ``args``.

    The quant block gets ``ebops.beta0`` = EBOPS_BETA0. Every width that equals the model dim follows ``--dim``
    (input net output, posenc, encoder, decoder layers, task inputs, mask dim, regression input 2 dim + 6).
    """
    import yaml

    init = copy.deepcopy(yaml.safe_load(Path(args.config).read_text())["model"]["model"]["init_args"])
    init["quant"] = {**init["quant"], "ebops": {"beta0": EBOPS_BETA0}}
    enc, dec_layer = init["encoder"], init["decoder"]["decoder_layer_config"]
    if args.enc_layers is not None:
        enc["num_layers"] = args.enc_layers
    if args.dec_layers is not None:
        init["decoder"]["num_decoder_layers"] = args.dec_layers
    for attn in (enc["attn_kwargs"], dec_layer["attn_kwargs"]):
        if args.heads is not None:
            attn["num_heads"] = args.heads
        if args.linformer_k is not None:
            attn["linformer_k"] = args.linformer_k
    if args.dim is not None and args.dim != init["dim"]:
        old, dim = init["dim"], args.dim
        init["dim"] = dec_layer["dim"] = dim
        for net in init["input_nets"]["init_args"]["modules"]:
            net["init_args"]["net"]["init_args"]["output_size"] = dim
            net["init_args"]["posenc"]["init_args"]["dim"] = dim
        for task in init["tasks"]["init_args"]["modules"]:
            targs = task["init_args"]
            if targs.get("dim") == old:
                targs["dim"] = dim
            for key in ("net", "object_net", "constituent_net", "node_net"):
                if key in targs:
                    net = targs[key]["init_args"]
                    net["input_size"] = 2 * dim + len(FIELDS) + 1 if net["input_size"] == 2 * old + len(FIELDS) + 1 else dim
                    if net.get("hidden_layers"):
                        net["hidden_layers"] = [dim if h == old else h for h in net["hidden_layers"]]
    return init


def instantiate(spec):
    """jsonargparse-style instantiation of a YAML spec (``class_path`` / ``init_args``; ``torch.nn.*`` strings)."""
    if isinstance(spec, dict) and "class_path" in spec:
        module, name = spec["class_path"].rsplit(".", 1)
        args = {key: instantiate(value) for key, value in (spec.get("init_args") or {}).items()}
        return getattr(importlib.import_module(module), name)(**args)
    if isinstance(spec, list):
        return [instantiate(value) for value in spec]
    if isinstance(spec, str) and spec.startswith("torch.nn."):
        from torch import nn

        return getattr(nn, spec.removeprefix("torch.nn."))()
    return spec


def model_summary(init: dict) -> dict:
    enc, dec_attn = init["encoder"], init["decoder"]["decoder_layer_config"]["attn_kwargs"]
    return {
        "dim": init["dim"],
        "heads": enc["attn_kwargs"]["num_heads"],
        "enc_layers": enc["num_layers"],
        "dec_layers": init["decoder"]["num_decoder_layers"],
        "attn_type": [enc.get("attn_type", "torch"), dec_attn.get("attn_type", "torch")],
        "linformer_k": [enc["attn_kwargs"].get("linformer_k"), dec_attn.get("linformer_k")],
        "linformer_seq_len": [enc["attn_kwargs"].get("linformer_seq_len"), dec_attn.get("linformer_seq_len")],
        "input_sort_field": init.get("input_sort_field"),
        "quant": init["quant"],
    }


def load_batch(args):
    """``args.batch`` CLIC events (ROOT file or dummy data) as numpy dicts (inputs, targets).

    int64 -> int32 for the jax mode only (jax runs without x64); the torch arms get int64, as 07_profile.py does.
    """
    import numpy as np
    import torch

    from hepattn.experiments.clic.pflow_data import CLICDataset

    ds = CLICDataset(
        filepath=args.data_file or "dummy",
        inputs={"node": ["features"]},
        targets={"particle": FIELDS},
        scale_dict_path=SCALE_DICT,
        num_events=args.batch,
        num_objects=QUERIES,
        max_nodes=NODES,
        dummy_data=not args.data_file,
    )
    events = [ds[i] for i in range(args.batch)]

    def stack(i):
        out = {k: torch.stack([e[i][k] for e in events]).numpy() for k in events[0][i]}
        if MODES[args.mode] != "jax":
            return out
        return {k: v.astype(np.int32) if v.dtype == np.int64 else v for k, v in out.items()}

    return stack(0), stack(1)


# --------------------------------------------------------------------------------------------- sweep
def run_sweep(args) -> None:
    if not args.config:
        out = Path(args.out).with_suffix(".config.yaml") if args.out else Path(tempfile.mkdtemp()) / "polaris_hgq.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        args.config = str(generate_config(out))
        print(f"### generated {args.config}", flush=True)
    keys = ("config", "dim", "heads", "enc_layers", "dec_layers", "linformer_k", "data_file", "warmup", "iters", "seed")
    passthrough = []
    for k in keys:
        if getattr(args, k) is not None:
            passthrough += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
    results = []

    def save() -> None:
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"args": vars(args), "results": results}, indent=1))

    plan = []
    modes = args.modes.split(",")
    for mode in sorted(modes, key=lambda m: MODES[m] != "jax"):  # JAX first
        batches = args.jax_batches if MODES[mode] == "jax" else args.hybrid_batches
        plan += [(mode, int(b)) for b in batches.split(",") if b]
    for mode, batch in plan:
        backend = MODES[mode]
        py = args.jax_python if backend == "jax" else args.torch_python
        env = {**os.environ, "KERAS_BACKEND": backend, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        cmd = [py, str(Path(__file__).resolve()), "--mode", mode, "--batch", str(batch), *passthrough]
        print(f"### {mode} batch={batch}: {' '.join(cmd)}", flush=True)
        try:
            cp = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            results.append({"mode": mode, "batch": batch, "error": f"timeout {args.timeout}s"})
            save()
            continue
        line = next((ln for ln in reversed(cp.stdout.splitlines()) if ln.startswith("RESULT ")), None)
        if cp.returncode != 0 or line is None:
            err = (cp.stderr or cp.stdout).strip().splitlines()[-15:]
            print("\n".join(err), flush=True)
            results.append({"mode": mode, "batch": batch, "error": "\n".join(err)})
            save()
            continue
        res = json.loads(line[len("RESULT ") :])
        print(json.dumps(res, indent=1), flush=True)
        results.append(res)
        save()
    print_table(results)


def print_table(results) -> None:
    ok = [r for r in results if "error" not in r]
    print("\n| mode | batch | median ms | p10-p90 ms | first step s | peak dev GB | peak RSS GB | grads nonzero | penalty |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        dev = "-" if r.get("peak_dev_gb") is None else f"{r['peak_dev_gb']:.2f}"
        print(
            f"| {r['mode']} | {r['batch']} | {r['median_ms']:.1f} | {r['p10_ms']:.1f}-{r['p90_ms']:.1f} | {r['first_step_s']:.1f} | "
            f"{dev} | {r['peak_rss_gb']:.2f} | {r['grads_nonzero']}/{r['n_grads']} | {r['penalty']:.4g} |"
        )
    for r in results:
        if "error" in r:
            print(f"| {r['mode']} | {r['batch']} | ERROR | {r['error'].splitlines()[-1][:120]} |")
    hyb = {r["batch"]: r["median_ms"] for r in ok if r["mode"] == "hybrid-07"}
    sur = {r["batch"]: r["median_ms"] for r in ok if r["mode"] == "hybrid-surrogate"}
    print("\n| batch | jax ms | hybrid-07 ms | jax / hybrid-07 (G2: <= 0.5) | hybrid-surrogate ms | jax / hybrid-surrogate |")
    print("|---|---|---|---|---|---|")
    for r in (r for r in ok if r["mode"] == "jax"):
        b = r["batch"]
        h, s = hyb.get(b), sur.get(b)
        ratio_h = f"{r['median_ms'] / h:.3f}" if h else "-"
        ratio_s = f"{r['median_ms'] / s:.3f}" if s else "-"
        print(f"| {b} | {r['median_ms']:.1f} | {f'{h:.1f}' if h else '-'} | {ratio_h} | {f'{s:.1f}' if s else '-'} | {ratio_s} |")


# --------------------------------------------------------------------------------------------- one run
def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20  # bytes on macOS, KiB on Linux


def run_one(args) -> None:
    backend = MODES[args.mode]
    if os.environ.get("KERAS_BACKEND") != backend:
        raise SystemExit(f"--mode {args.mode} needs KERAS_BACKEND={backend}, got {os.environ.get('KERAS_BACKEND')}")
    import keras
    import numpy as np

    import hepattn

    keras.utils.set_random_seed(args.seed)
    inputs, targets = load_batch(args)
    info = {"mode": args.mode, "batch": args.batch, "precision": "fp32", "ebops_every_step": True, "keras": keras.__version__}
    info["hepattn_file"] = hepattn.__file__  # which tree's code ran (PYTHONPATH vs an installed hepattn)
    if not args.config:
        args.config = str(generate_config(Path(tempfile.mkdtemp()) / "polaris_hgq.yaml"))
    init = model_config(args)
    info.update(model_summary(init), config=args.config, nodes=NODES, queries=QUERIES)
    info["data"] = args.data_file or "dummy"
    step, extra = make_jax_step(init, inputs) if backend == "jax" else make_torch_step(args, init, inputs, targets)
    info.update(extra)

    times = []
    t_first = None
    for i in range(args.warmup + args.iters):
        t0 = time.perf_counter()
        out = step()  # every step syncs the device before returning
        dt = time.perf_counter() - t0
        if i == 0:
            t_first = dt
            info.update(out)  # loss, penalty and gradient checks of step 0
        elif i >= args.warmup:
            times.append(dt * 1e3)
    q = statistics.quantiles(times, n=10)
    info.update(
        median_ms=statistics.median(times),
        p10_ms=q[0],
        p90_ms=q[-1],
        first_step_s=t_first,
        warmup=args.warmup,
        iters=args.iters,
        peak_rss_gb=peak_rss_gb(),
        peak_dev_gb=extra_peak_dev(backend),
    )
    if backend == "jax":
        info["cache_size"] = step.cache_size()
        assert info["cache_size"] == 1, f"{info['cache_size']} compilations: the step retraced"
    assert np.isfinite(info["loss"]), info
    print("RESULT " + json.dumps(info), flush=True)


def extra_peak_dev(backend: str):
    if backend == "torch":
        import torch

        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    import jax

    dev = jax.devices()[0]
    stats = dev.memory_stats() if dev.platform == "gpu" else None
    return stats["peak_bytes_in_use"] / 2**30 if stats else None


def linformer_ranks(layers) -> dict:
    """{attention name: (core kind, k)} of every built masked-Linformer attention among ``layers``."""
    return {layer.name: (layer.core_kind, int(layer.core.E.kernel.shape[-1])) for layer in layers if getattr(layer, "core", None) is not None}


def check_linformer(ranks: dict, init: dict) -> None:
    """Refuse an arm that did not build the configured masked Linformer (every attention, rank linformer_k).

    Raises:
        SystemExit: an attention without a masked-Linformer core, or with another rank.
    """
    enc, dec = init["encoder"], init["decoder"]
    expected = enc["num_layers"] + 3 * dec["num_decoder_layers"]
    ks = {k for _, k in ranks.values()}
    want = {enc["attn_kwargs"].get("linformer_k"), dec["decoder_layer_config"]["attn_kwargs"].get("linformer_k")}
    kinds = sorted({kind for kind, _ in ranks.values()})
    if len(ranks) != expected or ks != want or kinds != ["attend", "project", "project_unmasked"]:
        raise SystemExit(f"{len(ranks)} masked-Linformer attentions (want {expected}), k {sorted(ks)} (want {sorted(want)}), cores {kinds}")


def make_jax_step(init, inputs):
    import jax
    import keras
    import numpy as np

    from hepattn.keras_native.maskformer import from_config

    model = from_config({"class_path": "hepattn.keras.maskformer.KerasMaskFormer", "init_args": copy.deepcopy(init)})
    model(inputs, training=False)  # eager build from the static shapes, before any tracing
    ranks = linformer_ranks(model._flatten_layers())
    check_linformer(ranks, init)
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    x = jax.device_put(inputs)

    def surrogate(outputs):
        total = 0.0
        for value in jax.tree_util.tree_leaves(outputs):
            if keras.backend.is_float_dtype(value.dtype):
                keep = keras.ops.logical_and(keras.ops.isfinite(value), keras.ops.abs(value) < SENTINEL)
                total = total + keras.ops.mean(keras.ops.square(keras.ops.where(keep, value, 0.0)))
        return total

    def loss_fn(tv, ntv, x):
        out, ntv, losses = model.stateless_call(tv, ntv, x, training=True, return_losses=True)
        penalty = sum(losses)
        return surrogate(out) + penalty, (ntv, penalty)

    train = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    state = {"ntv": ntv, "first": True}

    def step():
        (loss, (state["ntv"], penalty)), grads = train(tv, state["ntv"], x)
        jax.block_until_ready((loss, grads, state["ntv"]))
        if not state["first"]:
            return {}
        state["first"] = False
        g = [np.asarray(gi) for gi in grads]
        return {
            "loss": float(loss),
            "penalty": float(penalty),
            "n_grads": len(g),
            "grads_nonzero": int(sum(bool(np.any(gi != 0)) for gi in g)),
            "grads_nonfinite": int(sum(not np.isfinite(gi).all() for gi in g)),
        }

    step.cache_size = train._cache_size
    extra = {
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "n_params": int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        "linformer_cores": sorted({f"{kind}:{k}" for kind, k in ranks.values()}),
        "sort_field": model.input_sort_field,
    }
    return step, extra


def hybrid_model(init):
    """KerasMaskFormer from the generated config (as the LightningCLI builds it), solver scipy, one thread."""
    from hepattn.keras.maskformer import KerasMaskFormer
    from hepattn.models.matcher import Matcher

    init = copy.deepcopy(init)
    init.pop("matcher")
    return KerasMaskFormer(
        input_nets=instantiate(init.pop("input_nets")),
        tasks=instantiate(init.pop("tasks")),
        matcher=Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False),
        **init,
    )


def make_torch_step(args, init, inputs, targets):
    import numpy as np
    import torch

    import hepattn.keras  # noqa: F401  pins the keras backend
    from hepattn.keras import set_keras_default_device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_keras_default_device(device)
    torch.manual_seed(args.seed)
    model = hybrid_model(init).to(device)
    inp = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
    tgt = {k: torch.from_numpy(v).to(device) for k, v in targets.items()}
    model.materialize(inp)  # lazy HGQ2 leaves and linformer cores built from the true static shapes (as 07)
    model.train()
    ranks = linformer_ranks(model.modules())
    check_linformer(ranks, init)
    decay, quant = model.trainable_parameter_groups()
    params = decay + quant

    def surrogate(out) -> torch.Tensor:
        total = torch.zeros((), device=device)
        stack = [out]
        while stack:
            node = stack.pop()
            for value in node.values():
                if isinstance(value, dict):
                    stack.append(value)
                elif torch.is_tensor(value) and value.dtype.is_floating_point:
                    keep = torch.isfinite(value) & (value.abs() < SENTINEL)
                    total = total + torch.where(keep, value, torch.zeros_like(value)).square().mean()
        return total

    def total_loss(losses) -> torch.Tensor:  # polaris/07_profile.py total_loss
        terms = []
        stack = [losses]
        while stack:
            node = stack.pop()
            for value in node.values():
                if isinstance(value, dict):
                    stack.append(value)
                elif torch.is_tensor(value) and value.dtype.is_floating_point:
                    m = value.mean()
                    if torch.isfinite(m):  # 07 skips non-finite terms (mask_bce was inf on dummy data before bug 1)
                        terms.append(m)
        if not terms:
            raise RuntimeError("no finite loss terms")
        return torch.stack(terms).sum()

    unique_layers = list({id(layer): layer for layer in model.keras_layers()}.values())

    def unique_quant_losses() -> torch.Tensor:  # quant_losses() without the task-head double count
        return torch.stack([torch.as_tensor(t) for layer in unique_layers for t in layer.losses]).sum()

    first = {"done": False}

    def step():
        model.zero_grad(set_to_none=True)  # as 07; the keras variables are not registered on the module tree, so also:
        for p in params:
            p.grad = None
        out = model(inp)
        if args.mode == "hybrid-07":
            out, _, losses = model.loss(out, dict(tgt))
            loss = total_loss(losses)
            penalty = model.quant_losses().detach()  # computed in the forward, not in the backward (as 07)
        else:
            penalty = unique_quant_losses()
            loss = surrogate(out) + penalty
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        if first["done"]:
            return {}
        first["done"] = True
        grads = [p.grad for p in params]
        return {
            "loss": float(loss.detach()),
            "penalty": float(penalty),
            "n_grads": len(grads),
            "grads_nonzero": int(sum(g is not None and bool(torch.any(g != 0)) for g in grads)),
            "grads_nonfinite": int(sum(g is not None and not bool(torch.isfinite(g).all()) for g in grads)),
        }

    extra = {
        "keras_layers_walked": len(list(model.keras_layers())),
        "keras_layers_unique": len(unique_layers),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "n_params": int(sum(np.prod(p.shape) for p in params)),
        "linformer_cores": sorted({f"{kind}:{k}" for kind, k in ranks.values()}),
        "sort_field": model.input_sort_field,
    }
    return step, extra


if __name__ == "__main__":
    a = parse_args()
    if a.sweep:
        run_sweep(a)
    elif a.mode:
        run_one(a)
    else:
        raise SystemExit("pass --mode or --sweep")
