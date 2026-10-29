#!/usr/bin/env python
r"""CPU diagnostics of the QAT warm start (ws2): bitwidth gradients, and the WRAP calibration of the validation.

    KERAS_BACKEND=jax JAX_PLATFORMS=cpu PYTHONPATH=src python polaris/23_ws_diag.py grads --config INIT/config.yaml \
        --init INIT/init.npz --batches 16 --batch 128 --beta 8.9e-10 --out grads.npz
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu PYTHONPATH=src python polaris/23_ws_diag.py honest --config INIT/config.yaml \
        --init INIT/init.npz --batch 128 --val-batches 8 --cal-batches 1 4 16 64 --out honest.json

``grads``   At the converted weights (state fixed), for each of ``--batches`` train micro-batches: the gradient of every
            trainable bitwidth (HGQ2 ``f`` of kif, ``b`` of kbi quantizers) with the EBOPs term off (task + weight
            regularizers) and on at ``--beta`` (the logged, active-step beta), and the global gradient norms (the
            optimizer clips at ``trainer.gradient_clip_val``). Penalty gradient = on - off (same forward). Check: a +1
            change of one rounded bit changes the penalty by beta x its STE gradient (EBOPs is bilinear in the bits).
            Then a Keras-AdamW replay of the measured gradients (``--steps``, OneCycle of the config, ``--lr-scales``,
            penalty on every ``--every``-th step, random batch order) predicts the bitwidth drift of a run.
``honest``  The validation loss of the converted model on ``--val-batches`` val batches with the WRAP integer bits
            (a) as training leaves them for ``val/loss``: HGQ2's kif datalanes have ``i_decay_speed`` 0.01, so each
            training-mode call sets i = max(i - 0.01, the batch's minimal i), a running maximum; here from a reset over
            ``--running`` train batches (the spec's i0 of the init decays away within ~1000 steps in the pilot),
            (b) traced over the first N train batches (``Trainer.calibrated_ntv``, ``val_cal/loss``), for each N of
            ``--cal-batches``, (c) none: the training-mode forward on the val events (a WRAP lane only rounds in
            training, no overflow), which is what ``train/loss`` measures. EBOPs at each set of integer bits.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path


def setup(args, *, num_train: int, num_val: int = 0):
    from hepattn.experiments.clic.main_jax import ClicSource  # noqa: PLC0415
    from hepattn.keras_native import loop  # noqa: PLC0415
    from hepattn.keras_native.cli import apply_overrides  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    over = argparse.Namespace(batch_size=args.batch, num_workers=args.workers, num_train=num_train, num_val=num_val or None)
    rc = apply_overrides(load_config([args.config]), over)
    train = ClicSource(rc, "train", seed=42)
    val = ClicSource(rc, "val", seed=42) if num_val else None
    settings = loop.LoopSettings.from_run_config(rc, args.run_dir, max_epochs=10_000, matmul_precision="highest")  # no fit: a long OneCycle
    t = loop.Trainer(rc, settings, sample=next(iter(train.epoch(0))), train_batches=train.num_batches)
    t.warm_start(args.init)
    return rc, t, train, val


def bit_variables(model):
    """(index in trainable_variables, path) of every trainable bitwidth (the optimizer's quantizer group)."""
    return [(n, v.path) for n, v in enumerate(model.trainable_variables) if "quantizer" in v.path]


def cmd_grads(args) -> int:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    rc, t, train, _ = setup(args, num_train=int(args.batches * args.batch * 1.05) + 2 * args.batch)
    model, steps = t.steps.model, t.steps
    clip = rc.optimizer.clip_norm
    t._set_beta(args.beta)  # noqa: SLF001
    bits = bit_variables(model)
    idx = [n for n, _ in bits]
    grads_fn = jax.jit(lambda state, batch, on: steps.grads(state, batch, ebops_on=on), static_argnums=2)

    def norm(gs):
        return float(jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in gs)))

    task, pen, rows, var_norms = [], [], [], []
    for n, batch in enumerate(train.epoch(0)):
        if n >= args.batches:
            break
        t0 = time.time()
        g_on, o_on = grads_fn(t.state, (batch.inputs, batch.targets), True)
        g_off, o_off = grads_fn(t.state, (batch.inputs, batch.targets), False)
        b_on = [np.asarray(g_on[i], np.float64).ravel() for i in idx]
        b_off = [np.asarray(g_off[i], np.float64).ravel() for i in idx]
        task.append(np.concatenate(b_off))
        var_norms.append([float(jnp.linalg.norm(g)) for g in g_off])
        pen.append(np.concatenate(b_on) - task[-1])
        rows.append({
            "batch": n,
            "task_loss": float(o_off.task_loss),
            "task_loss_on": float(o_on.task_loss),
            "penalty": float(o_on.penalty),
            "ebops": float(o_on.penalty) / args.beta,
            "norm_on": norm(g_on),
            "norm_off": norm(g_off),
            "norm_bits_off": float(np.linalg.norm(task[-1])),
            "s": time.time() - t0,
        })
        print(json.dumps(rows[-1]), flush=True)
    task, pen = np.stack(task), np.stack(pen)  # (batches, elements)
    sizes = [int(np.prod(model.trainable_variables[i].shape)) for i in idx]
    owner = np.repeat(np.arange(len(idx)), sizes)
    paths = [p for _, p in bits]
    var_norms = np.asarray(var_norms)  # (batches, trainable variables): which tensors set the global norm
    share = (var_norms**2).mean(0) / (var_norms**2).sum(1).mean()
    norm_top = [
        {"path": model.trainable_variables[k].path, "share_of_norm2": float(share[k]), "norm_median": float(np.median(var_norms[:, k]))}
        for k in np.argsort(-share)[:15]
    ]
    norms_on = np.asarray([r["norm_on"] for r in rows])
    norms_off = np.asarray([r["norm_off"] for r in rows])
    np.savez_compressed(
        args.out,
        task=task.astype(np.float32),
        pen=pen.astype(np.float32),
        owner=owner,
        paths=np.asarray(paths),
        norm_on=norms_on,
        norm_off=norms_off,
        beta=args.beta,
        clip=np.nan if clip is None else clip,
    )

    check = ste_check(t, next(iter(train.epoch(0))), args.beta, paths, idx, sizes, pen)

    # per element statistics (raw gradients), then per tensor and per group
    mu_t, sd_t, mu_p = task.mean(0), task.std(0, ddof=1), pen.mean(0)
    live = np.any(task != 0, axis=0)  # elements with a task gradient (a bit with i + f <= 0 has none)
    sign_task = (np.sign(task) == np.sign(mu_t)).mean(0)  # agreement of each batch with the mean sign
    every = args.every
    total_signs = (task + mu_p / every > 0).mean(0)  # the time-averaged pull added to each batch: > 0 lowers the bit

    # per tensor summary for the top tensors by penalty pull and the groups
    def stats(sel):
        dead = int((~live[sel]).sum())
        sel = sel[live[sel]]
        if sel.size == 0:
            return {"elements": 0, "no_task_gradient": dead}
        clip_off = np.minimum(1.0, (clip or np.inf) / norms_off)
        sd_c = (task[:, sel] * clip_off[:, None]).std(0, ddof=1)
        return {
            "elements": int(sel.size),
            "no_task_gradient": dead,
            "task_mean_abs_median": float(np.median(np.abs(mu_t[sel]))),
            "task_std_median": float(np.median(sd_t[sel])),
            "task_std_clipped_median": float(np.median(sd_c)),
            "task_snr_median": float(np.median(np.abs(mu_t[sel]) / np.maximum(sd_t[sel], 1e-30))),
            "task_sign_agreement_mean": float(sign_task[sel].mean()),
            "pen_active_median": float(np.median(mu_p[sel])),
            "pen_active_min": float(mu_p[sel].min()),
            "pen_time_avg_over_task_std_median": float(np.median(mu_p[sel] / every / np.maximum(sd_t[sel], 1e-30))),
            "frac_batches_total_grad_positive": float(total_signs[sel].mean()),
            "beta_for_pull_eq_noise_median": float(np.median(args.beta * every * sd_t[sel] / np.maximum(mu_p[sel], 1e-30))),
        }

    tensors = []
    for k, p in enumerate(paths):
        sel = np.nonzero(owner == k)[0]
        tensors.append({"path": p, **stats(sel), "pen_sum": float(mu_p[sel].sum()), "size": int(sel.size)})
    tensors.sort(key=lambda d: -d["pen_sum"])

    def group(pred):
        return np.nonzero(np.isin(owner, [k for k, p in enumerate(paths) if pred(p)]))[0]

    groups = {
        "all": np.arange(task.shape[1]),
        "f datalane (kif WRAP)": group(lambda p: p.endswith("/f") and "_kq/" not in p),
        "f weight (kif kq)": group(lambda p: p.endswith("/f") and "_kq/" in p),
        "i weight (kif kq, SAT_SYM)": group(lambda p: p.endswith("/i")),
        "b weight (kbi)": group(lambda p: p.endswith("/b")),
    }
    report = {
        "config": args.config,
        "init": args.init,
        "beta_active": args.beta,
        "every": every,
        "clip": clip,
        "epsilon": rc.optimizer.epsilon,
        "batch": args.batch,
        "batches": rows,
        "ste_check": check,
        "norm2_top_tensors": norm_top,
        "groups": {name: stats(sel) for name, sel in groups.items() if sel.size},
        "top_tensors": tensors[: args.top],
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"ste_check": check, "groups": report["groups"]}, indent=1), flush=True)
    return 0


def ste_check(t, batch, beta: float, paths, idx, sizes, pen, events: int = 8) -> list[dict]:
    """No scaling, normalisation or stop_gradient between beta x EBOPs and the bitwidth gradient.

    In a training forward the WRAP integer bits follow the data, so a bit change moves EBOPs also through every
    downstream batch maximum, a path without gradient. Here every WRAP integer bit is frozen at 20 (tracing mode keeps
    max(stored, batch), 20 > any batch), so EBOPs depends on the bitwidths alone and is bilinear for an einsum leaf: a +1
    and a -1 step of one rounded bit must change the penalty by exactly the STE gradient. Elements: the largest penalty
    gradient of the top EBOPs kernel's fractional and integer bits (E/F projection ``kq/f``, ``kq/i``).
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from hgq.utils.minmax_trace import TrainingFlagWrapper  # noqa: PLC0415

    model, leaves = t.steps.model, t.steps.ebops_leaves
    t._set_beta(beta)  # noqa: SLF001
    ntv = list(t.state.ntv)
    for p in t._wrap_index[0]:  # noqa: SLF001
        ntv[p] = jnp.full(ntv[p].shape, 20.0, ntv[p].dtype)
    inputs = {k: v[:events] for k, v in batch.inputs.items()}

    # jitted: eagerly, Keras' _clear_losses matches terms by id() of concrete arrays, which Python reuses, so the
    # number of returned terms can change between two eager calls (seen here); a trace fixes the list
    @jax.jit
    def terms(tv):
        with leaves.enabled(True):
            _, _, losses = model.stateless_call(tv, ntv, inputs, training=TrainingFlagWrapper("tracing"), return_losses=True)
        return leaves.split_losses(losses)[0]

    @jax.jit
    @jax.grad
    def grad_penalty(tv):
        return sum(terms(tv), jnp.zeros((), jnp.float32))

    def change(tv, base):
        # per leaf term in float64: the float32 total (~600 at beta 8.9e-10) cannot resolve a one-bit change (~3e-5)
        return float(sum(np.float64(a) - np.float64(b) for a, b in zip(terms(tv), base, strict=True)))

    tv0 = list(t.state.tv)
    grad = grad_penalty(tv0)
    base = terms(tv0)
    p0 = float(sum(np.float64(b) for b in base))
    out = []
    by_pen = np.bincount(np.repeat(np.arange(len(idx)), sizes), weights=pen.mean(0))
    kernel = max((k for k, p in enumerate(paths) if "_proj_kq/" in p and p.endswith("/f")), key=lambda k: by_pen[k])
    integer = [k for k, p in enumerate(paths) if p == paths[kernel][: -len("/f")] + "/i"]
    for k in [kernel, *integer]:
        n = idx[k]
        g = np.asarray(grad[n], np.float64).ravel()
        pos = int(np.argmax(g))
        row = {"path": paths[k], "element": pos, "ste_grad": float(g[pos]), "ebops_at_frozen_i": p0 / beta}
        for delta in (1.0, -1.0):
            tv = list(tv0)
            flat = np.asarray(tv[n]).ravel().copy()
            flat[pos] += delta
            tv[n] = jnp.asarray(flat.reshape(tv[n].shape))
            row[f"finite_diff_{delta:+g}_bit"] = change(tv, base) / delta
        row["relative_error"] = max(abs(row[f"finite_diff_{d:+g}_bit"] - row["ste_grad"]) for d in (1.0, -1.0)) / abs(row["ste_grad"])
        out.append(row)
        print("STE check", json.dumps(row), flush=True)
    return out


def onecycle(steps: int, lo: float, hi: float, end: float, pct: float):
    """``optim.OneCycleLR`` values at optimizer steps 0..steps-1 (numpy)."""
    import numpy as np  # noqa: PLC0415

    p1, p2 = pct * steps - 1, steps - 1
    n = np.arange(steps, dtype=np.float64)
    rise = hi + (lo - hi) / 2 * (np.cos(np.pi * n / p1) + 1)
    fall = end + (hi - end) / 2 * (np.cos(np.pi * (n - p1) / (p2 - p1)) + 1)
    return np.where(n <= p1, rise, fall)


def cmd_ste(args) -> int:
    """Only the ``ste_check`` of ``grads`` (its tensors chosen by the penalty gradient of one batch)."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    if args.mutant == "halfgrad":  # control: same EBOPs value, half its gradient (a surrogate that the check must catch)
        from hgq.quantizer.quantizer import Quantizer  # noqa: PLC0415

        orig = Quantizer.bits_

        def bits_(self, shape):
            b = orig(self, shape)
            return 0.5 * b + jax.lax.stop_gradient(0.5 * b)

        Quantizer.bits_ = bits_
    _, t, train, _ = setup(args, num_train=4 * args.batch)
    model = t.steps.model
    bits = bit_variables(model)
    idx = [n for n, _ in bits]
    sizes = [int(np.prod(model.trainable_variables[i].shape)) for i in idx]
    batch = next(iter(train.epoch(0)))
    t._set_beta(args.beta)  # noqa: SLF001
    g_on, _ = t.steps.grads(t.state, (batch.inputs, batch.targets), ebops_on=True)
    g_off, _ = t.steps.grads(t.state, (batch.inputs, batch.targets), ebops_on=False)
    pen = np.concatenate([np.asarray(g_on[i] - g_off[i], np.float64).ravel() for i in idx])[None]
    rows = ste_check(t, batch, args.beta, [p for _, p in bits], idx, sizes, pen)
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
    return 0 if all(r["relative_error"] < 1e-2 for r in rows) else 1


def cmd_replay(args) -> int:
    """Keras AdamW (``m``, ``v``, ``alpha = lr sqrt(1 - b2^t) / (1 - b1^t)``, step ``alpha m / (sqrt(v) + eps)``; no weight
    decay: the quantizer group) on the measured bitwidth gradients of ``grads``, per setting of the grid.

    Gradients at the init are drawn at random per step; the penalty (``pen`` at ``beta x mult``) on every ``every``-th
    step; global clipping at ``--clip`` with each batch's measured norm, the penalty's part rescaled
    (``|g_on|^2 = |g_off|^2 + 2 m <task, pen> + m^2 |pen|^2``: the penalty is zero outside the bitwidths). Up to
    ``--per-tensor`` elements of each tensor (stratified), so the per-tensor means (``bits_*_mean``) are unbiased. Out:
    ``bits_{f,b,i}_mean`` drift (mean over tensors of the mean element drift, as ``BitwidthMonitor``), the linearised
    EBOPs change ``sum dEBOPs/dx dx`` (``dEBOPs/dx = pen / beta``) and task-loss change ``sum mean(task) dx``.
    """
    import numpy as np  # noqa: PLC0415

    d = np.load(args.npz)
    paths, owner = list(d["paths"]), d["owner"]
    beta0, norm_off = float(d["beta"]), d["norm_off"].astype(np.float64)
    rng = np.random.default_rng(0)

    def stratum(k):
        e = np.nonzero(owner == k)[0]
        return e if e.size <= args.per_tensor else rng.choice(e, args.per_tensor, replace=False)

    sub = np.concatenate([stratum(k) for k in range(len(paths))])
    weight = (np.bincount(owner, minlength=len(paths)) / np.maximum(np.bincount(owner[sub], minlength=len(paths)), 1))[owner[sub]]
    task_all, pen_all = d["task"], d["pen"]
    dot = np.einsum("bj,bj->b", task_all.astype(np.float64), pen_all.astype(np.float64))
    pp = np.einsum("bj,bj->b", pen_all.astype(np.float64), pen_all.astype(np.float64))
    task, pen, o = task_all[:, sub].astype(np.float64), pen_all[:, sub].astype(np.float64), owner[sub]
    del task_all, pen_all
    debops, mu_t = pen.mean(0) / beta0, task.mean(0)
    cnt = np.maximum(np.bincount(o, minlength=len(paths)), 1)
    kinds = {s: [k for k, p in enumerate(paths) if p.endswith("/" + s)] for s in ("f", "b", "i")}
    rows = []
    for steps in args.steps:
        lr = onecycle(steps, args.lr_initial, args.lr_max, args.lr_end, args.pct_start)
        order = rng.integers(0, task.shape[0], steps)
        for every, mult, scale, eps, clip in itertools.product(args.every, args.beta_mult, args.lr_scale, args.eps, args.clip):
            n_on = np.sqrt(norm_off**2 + 2 * mult * dot + mult**2 * pp)
            c_on = np.minimum(1, clip / n_on) if clip > 0 else np.ones_like(n_on)
            c_off = np.minimum(1, clip / norm_off) if clip > 0 else np.ones_like(norm_off)
            m, v, x = np.zeros(sub.size), np.zeros(sub.size), np.zeros(sub.size)
            for s in range(steps):
                b = order[s]
                g = (task[b] + mult * pen[b]) * c_on[b] if s % every == 0 else task[b] * c_off[b]
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                x -= lr[s] * scale * math.sqrt(1 - 0.999 ** (s + 1)) / (1 - 0.9 ** (s + 1)) * m / (np.sqrt(v) + eps)
            per = np.bincount(o, weights=x, minlength=len(paths)) / cnt
            row = {
                "steps": steps,
                "every": every,
                "beta_logged": beta0 * mult,
                "lr_scale": scale,
                "eps": eps,
                "clip": clip,
                **{f"bits_{k}_mean_drift": float(per[v_].mean()) for k, v_ in kinds.items() if v_},
                "d_ebops_lin": float((weight * debops * x).sum()),
                "d_task_lin": float((weight * mu_t * x).sum()),
                "max_move": float(lr.sum() * scale),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps({"npz": args.npz, "elements": int(sub.size), "rows": rows}, indent=1) + "\n")
    return 0


def cmd_honest(args) -> int:
    """The stages of the module docstring; the train batches: N_max calibration batches, then ``--running`` others."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    n_cal = max(args.cal_batches)
    need = n_cal + args.running
    _, t, train, val = setup(args, num_train=int(need * args.batch * 1.05) + 2 * args.batch, num_val=int(args.val_batches * args.batch))
    model = t.steps.model
    t._set_beta(1.0)  # noqa: SLF001  (EBOPs read-out = penalty)
    i_pos, _ = t._wrap_index  # noqa: SLF001
    decay = sorted({float(np.asarray(v.value)) for v in model.non_trainable_variables if v.path.endswith("i_decay_speed")})
    out = {"config": args.config, "init": args.init, "batch": args.batch, "i_decay_speed": decay, "stages": {}}

    def evaluate(ntv, name, **extra):
        res = t.validate(val, state=t.state._replace(ntv=ntv), stage="v")
        out["stages"][name] = {"loss": res["v/loss"], "bits_i_mean": res["v/bits_i_mean"], **extra}
        print(f"RESULT {name}: loss {res['v/loss']:.4f} {extra}", flush=True)

    # (a) as in training: from the integer bits reset (no trace of the spec's i0 left, as after ~1000 steps of the pilot),
    # --running training-mode forwards on other train batches: i = max(i - i_decay_speed, the batch's minimal i)
    step = jax.jit(lambda tv, ntv, x: model.stateless_call(tv, ntv, x, training=True)[1])
    ntv = list(t.state.ntv)
    for p in i_pos:
        ntv[p] = jnp.full(ntv[p].shape, -1e9, ntv[p].dtype)
    batches = train.epoch(0)
    for n, batch in enumerate(batches):
        if n >= need:
            break
        if n >= n_cal:
            ntv = step(t.state.tv, ntv, batch.inputs)
    # (b) traced over the first N train batches (the calibrated validation); with margins m > 0 for the largest N: every
    # traced integer bit + m (range x 2^m, EBOPs up)
    for n in sorted(args.cal_batches):
        ntv_cal, ebops_cal = t.calibrated_ntv(train, n)
        evaluate(ntv_cal, f"calibrated_{n}", ebops=ebops_cal, events=n * args.batch)
    for m in args.margins:
        wide = list(ntv_cal)
        for p in i_pos:
            wide[p] = wide[p] + m
        evaluate(wide, f"calibrated_{max(args.cal_batches)}_margin_{m}", ebops=ebops_at(t, wide))
    if args.running:
        evaluate(ntv, f"training_running_max_{args.running}", ebops=ebops_at(t, ntv))
    # (c) the training-mode forward on the val events: no overflow at all (a WRAP lane only rounds in training), i.e.
    # what train/loss measures
    fwd = jax.jit(lambda tv, ntv, x: model.stateless_call(tv, ntv, x, training=True)[0])
    tot, events = 0.0, 0
    for batch in val.epoch(0):
        outputs = fwd(t.state.tv, t.state.ntv, batch.inputs)
        n = batch.num_real
        outputs = jax.tree_util.tree_map(lambda x, n=n: x[:n], outputs)
        task, _ = t._tail(outputs, {k: v[:n] for k, v in batch.targets.items()})  # noqa: SLF001
        tot += float(task) * n
        events += n
    out["stages"]["training_mode_on_val"] = {"loss": tot / events, "events": events}
    print(f"RESULT training_mode_on_val: loss {tot / events:.4f}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def ebops_at(t, ntv) -> float:
    """EBOPs at max(the stored integer bits of ``ntv``, the first calibration batch's): HGQ2 computes EBOPs only in a
    forward that also sets the integer bits; a tracing forward keeps max(stored, batch).
    """
    import numpy as np  # noqa: PLC0415

    batch_inputs = t._cal_inputs[0] if t._cal_inputs else None  # noqa: SLF001
    if batch_inputs is None:
        return float("nan")
    _, penalty = t._calibrate(t.state.tv, ntv, batch_inputs)  # noqa: SLF001
    return float(np.asarray(penalty)) / t.beta


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="23_ws_diag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("grads", "honest", "ste"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="the init's config.yaml (22_warmstart.py init)")
        p.add_argument("--init", required=True, help="the converted init.npz")
        p.add_argument("--batch", type=int, default=128)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--run-dir", required=True, help="scratch run directory of the Trainer (nothing is trained)")
        p.add_argument("--out", required=True)
    sub.choices["ste"].add_argument("--beta", type=float, required=True)
    sub.choices["ste"].add_argument("--mutant", choices=["none", "halfgrad"], default="none")
    g = sub.choices["grads"]
    g.add_argument("--batches", type=int, default=16)
    g.add_argument("--beta", type=float, required=True, help="the logged (active-step) beta")
    g.add_argument("--every", type=int, default=8, help="PeriodicEBOPs every_n_steps")
    g.add_argument("--top", type=int, default=12)
    r = sub.add_parser("replay")
    r.add_argument("--npz", required=True, help="the grads output")
    r.add_argument("--steps", type=int, nargs="+", default=[2000])
    r.add_argument("--every", type=int, nargs="+", default=[8])
    r.add_argument("--beta-mult", type=float, nargs="+", default=[1.0], help="x the grads --beta")
    r.add_argument("--lr-scale", type=float, nargs="+", default=[20.0])
    r.add_argument("--eps", type=float, nargs="+", default=[1e-8])
    r.add_argument("--clip", type=float, nargs="+", default=[0.1], help="global-norm clip (0: none)")
    r.add_argument("--per-tensor", type=int, default=400)
    r.add_argument("--lr-initial", type=float, default=1e-6)
    r.add_argument("--lr-max", type=float, default=5e-5)
    r.add_argument("--lr-end", type=float, default=1e-6)
    r.add_argument("--pct-start", type=float, default=0.05)
    r.add_argument("--out", required=True)
    h = sub.choices["honest"]
    h.add_argument("--val-batches", type=int, default=8)
    h.add_argument("--cal-batches", type=int, nargs="+", default=[1, 4, 16, 64])
    h.add_argument("--running", type=int, default=250, help="training-mode forwards for the running integer bits (a); 0: skip")
    h.add_argument("--margins", type=int, nargs="*", default=[], help="integer-bit margins on the largest calibration")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command != "replay" and os.environ.get("KERAS_BACKEND") != "jax":
        raise SystemExit("KERAS_BACKEND must be jax")
    return {"grads": cmd_grads, "replay": cmd_replay, "honest": cmd_honest, "ste": cmd_ste}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
