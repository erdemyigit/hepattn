#!/usr/bin/env python
r"""Checks before a QAT warm start from a float Lightning CLIC checkpoint (keras_native / JAX path). CPU is enough.

    PYTHONPATH=src python polaris/21_warmstart_check.py float-eval --config RUN/config.yaml --ckpt K --val V --out w1.json
    PYTHONPATH=src python polaris/21_warmstart_check.py parity --config RUN/config.yaml --ckpt K --val V \
        --qat-config polaris_hgq.yaml --out w2.json
    PYTHONPATH=src python polaris/21_warmstart_check.py qinit --config RUN/config.yaml --ckpt K --val V \
        --qat-config polaris_hgq.yaml --out w3.json

``--config`` is the config the run saved (``SaveConfig``); the checks use a CPU copy of it (``<out>.float.yaml``): the
matcher's GPU ``device_solver`` is dropped and the solver is ``scipy`` (exact, as ``lap1015``; ties may break
differently). ``--qat-config`` is the quantized target (``03_make_config.py`` with ``HGQ_SOURCE_CONFIG``).

``float-eval``  The torch model on the first ``--num-events`` validation events (before the reader's node / particle
                cuts, as ``data.num_val``), batches of ``--batch`` (the run's val batch: the class loss is normalised
                per batch). ``val/loss`` is the batch-size-weighted mean of the per-batch total, as Lightning
                aggregates ``self.log`` in validation; every matched loss term likewise. ``--precision fp32 bf16``
                (bf16: ``torch.autocast`` on the CPU over forward and loss, as Lightning's bf16-mixed step).
``parity``      Gate: torch (fp32) vs the converted native model on real events, (a) the float native target and
                (b) the QAT target with every HGQ2 Quantizer the identity (float mode):
                per output max |diff| <= ``--k-ulp`` x and median |diff| <= ``--k-q50`` x those of torch's own 1-ulp
                spread, each at least ``--rtol-floor`` x max |value| (the 1-ulp spread: every nonzero float input
                moved by one ulp, the model's rounding sensitivity at these events), attention-mask flips <=
                ``--flip-frac``, matched loss terms within 1e-4 + 1e-3 |ref|
                (``test_org_model.py``). Each ``--mutant`` (one native variable changed after the conversion) must
                fail the gate.
``qinit``       The QAT target at the converted weights: matched loss, quantized vs float mode, at the loaded
                integer bits, in a first training step's forward (integer bits fitted to the batch) and after an
                integer-bit trace on other events; EBOPs (``penalty / beta`` at beta 1; per leaf the stored
                ``_ebops``, whose sum must equal the total) for the last two (``cmd_qinit``).

Data option ``--cossin repo|fixed``: ``fixed`` puts the topo clusters' ``cos(phi)`` / ``sin(phi)`` into node features
3 / 4 (org PR #5); ``repo`` keeps this branch's reader (both columns hold ``topo_phi``). The layout is asserted.
"""

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

# node_features column order of CLICDataset.load_event (pflow_data.py, the node_features dict)
COL_PHI, COL_COS, COL_SIN, COL_IS_TOPO = 2, 3, 4, 26
SENTINEL = 1e30
# parity gate per float output: max |native - torch| <= max(K_ULP x torch's own 1-ulp spread, RTOL_FLOOR x max |torch|).
# A trained model amplifies rounding: one ulp of the inputs moves its final mask logits by ~1e-4 of their range (measured),
# so a fixed relative tolerance would either be loose for the early layers or fail a correct port in the late ones.
# The median |diff| catches a coherent shift (one changed weight moves every event) that hides under the max: a correct
# port's median equals the 1-ulp median within 1.0x (measured, 32 events), and it is a stable statistic over ~1e6 slots.
K_ULP = 10.0
K_Q50 = 4.0
RTOL_FLOOR = 1e-6
FLIP_FRAC = 1e-4
LOSS_ATOL, LOSS_RTOL = 1e-4, 1e-3


# ----------------------------------------------------------------------------------------------- config and data


def cpu_config(config: list[str], out: Path) -> Path:
    """The saved config with the matcher on the CPU (``device_solver`` dropped, ``scipy``), written to ``out``."""
    import yaml  # noqa: PLC0415

    from hepattn.keras_native import config as nconfig  # noqa: PLC0415

    sources = [Path(p) for p in config]
    cfg = nconfig.read_yaml(sources)
    matcher = cfg["model"]["model"]["init_args"]["matcher"]["init_args"]
    matcher.pop("device_solver", None)
    matcher.update(default_solver="scipy", adaptive_solver=False)
    # relative scale-dict paths resolved as load_config does for the source (its directory, then the packaged configs)
    dirs = nconfig._search_dirs(sources)  # noqa: SLF001
    cfg["model"] = nconfig._resolve_scale_dicts(cfg["model"], dirs)  # noqa: SLF001
    cfg["data"] = nconfig._resolve_scale_dicts(cfg["data"], dirs)  # noqa: SLF001
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def source_matcher(config: list[str]) -> dict:
    """The matcher settings the run was trained with (the eval uses scipy on the CPU instead)."""
    from hepattn.keras_native import config as nconfig  # noqa: PLC0415

    return nconfig.read_yaml([Path(p) for p in config])["model"]["model"]["init_args"]["matcher"].get("init_args", {})


def torch_model(config: Path, ckpt: str, n_jobs: int):
    from hepattn.experiments.clic.convert_to_native import instantiate_torch_model, read_lightning_state, source_model_section  # noqa: PLC0415

    entry = copy.deepcopy(source_model_section([config]))
    entry["init_args"]["matcher"]["init_args"].update(parallel_solver=n_jobs > 1, n_jobs=max(1, n_jobs))
    model = instantiate_torch_model(entry)
    state, info = read_lightning_state(ckpt)
    model.load_state_dict(state, strict=True)
    return model.eval(), info


def val_dataset(rc, val_path: str, num_events: int):
    from hepattn.experiments.clic.native_data import make_dataset  # noqa: PLC0415

    return make_dataset(dataclasses.replace(rc.data, valid_path=val_path), "val", dummy_data=False, num_events=num_events)


def apply_cossin(inputs: dict, variant: str) -> dict:
    """Node features with the topo ``cos(phi)`` / ``sin(phi)`` columns as ``variant`` says (module docstring).

    Asserts that this branch's reader puts ``topo_phi`` in both columns (else the feature layout changed).
    """
    import torch  # noqa: PLC0415

    f = inputs["node_features"]
    topo = f[..., COL_IS_TOPO] == 1
    for col in (COL_COS, COL_SIN):
        assert torch.equal(f[..., col][topo], f[..., COL_PHI][topo]), f"column {col} of topo nodes is not topo_phi: reader layout changed"
    if variant == "repo":
        return inputs
    f = f.clone()
    f[..., COL_COS] = torch.where(topo, inputs["node_cosphi"], f[..., COL_COS])
    f[..., COL_SIN] = torch.where(topo, inputs["node_sinphi"], f[..., COL_SIN])
    return {**inputs, "node_features": f}


def flatten(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        out.update(flatten(value, f"{prefix}{key}.") if isinstance(value, dict) else {f"{prefix}{key}": value})
    return out


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def env_info() -> dict:
    import torch  # noqa: PLC0415

    return {"host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__, "threads": torch.get_num_threads()}


# ----------------------------------------------------------------------------------------------- W1 float-eval


def cmd_float_eval(args) -> int:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    cfg = cpu_config(args.config, out.with_suffix(".float.yaml"))
    rc = load_config([cfg])
    model, info = torch_model(cfg, args.ckpt, args.n_jobs)
    dataset = val_dataset(rc, args.val, args.num_events)
    report = {
        "checkpoint": {"path": args.ckpt, "sha256": sha256(args.ckpt), **info},
        "val": {"path": args.val, "sha256": sha256(args.val) if args.hash_data else None, "num_events_read": args.num_events, "events": len(dataset)},
        "batch": args.batch,
        "cossin": args.cossin,
        "matcher": {"solver": "scipy", "n_jobs": args.n_jobs, "source": source_matcher(args.config)},
        "env": env_info(),
        "results": {},
    }
    for precision in args.precision:
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
        sizes, rows, t0 = [], [], time.time()
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs = apply_cossin(inputs, args.cossin)
                ctx = torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()
                with ctx:
                    outputs = model(inputs)
                    _, _, losses = model.loss(outputs, targets)
                sizes.append(int(inputs["node_features"].shape[0]))
                rows.append({k: float(v) for k, v in flatten(losses).items()})
                print(f"{precision} batch {len(sizes)}: {sizes[-1]} events, loss {sum(rows[-1].values()):.5f}, {time.time() - t0:.0f} s", flush=True)
        w = np.asarray(sizes, dtype=np.float64) / sum(sizes)
        terms = {k: float(sum(wi * r[k] for wi, r in zip(w, rows, strict=True))) for k in rows[0]}
        per_batch = [sum(r.values()) for r in rows]
        report["results"][precision] = {
            "val/loss": float(np.dot(w, per_batch)),
            "per_batch_loss": per_batch,
            "batch_sizes": sizes,
            "terms": terms,
            "per_layer": {layer: sum(v for k, v in terms.items() if k.startswith(layer + ".")) for layer in sorted({k.split(".")[0] for k in terms})},
            "seconds": time.time() - t0,
        }
        print(f"RESULT {precision} val/loss {report['results'][precision]['val/loss']:.5f}", flush=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


# ----------------------------------------------------------------------------------------------- W2 parity


@contextlib.contextmanager
def float_mode():
    """Every HGQ2 ``Quantizer`` the identity (G4's float mode, ``19_g4_compare.py cmd_native_forward``)."""
    from hgq.quantizer import Quantizer  # noqa: PLC0415

    original = Quantizer.call
    Quantizer.call = lambda self, x, training=None: x  # noqa: ARG005
    try:
        yield
    finally:
        Quantizer.call = original


def forward_errors(ref: dict, got: dict) -> dict:
    """Per output: max / median |diff| and max |ref| over valid slots (|ref| < 1e30; the sentinel must match exactly);
    booleans: flips.

    Raises:
        KeyError: If the output keys differ.
    """
    import numpy as np  # noqa: PLC0415

    if set(ref) != set(got):
        raise KeyError(f"output keys differ: {sorted(set(ref) ^ set(got))}")
    out = {}
    for key, value in ref.items():
        value, other = np.asarray(value), np.asarray(got[key])
        if value.dtype == bool:
            out[key] = {"flips": int((value != other).sum()), "n": int(value.size)}
            continue
        keep = np.abs(value) < SENTINEL
        diff = np.abs(value[keep].astype(np.float64) - other[keep].astype(np.float64))
        out[key] = {
            "max_abs": float(diff.max()),
            "q50_abs": float(np.median(diff)),
            "ref_max_abs": float(np.abs(value[keep]).max()),
            "sentinel_equal": bool(np.array_equal(value[~keep], other[~keep])),
        }
        out[key]["rel"] = out[key]["max_abs"] / max(out[key]["ref_max_abs"], 1e-12)
    return out


def loss_errors(ref: dict, got: dict) -> dict:
    out = {}
    for key, value in ref.items():
        diff = abs(got[key] - value)
        out[key] = {"ref": value, "got": got[key], "diff": diff, "ok": bool(diff <= LOSS_ATOL + LOSS_RTOL * abs(value))}
    return out


def output_tol(ulp: dict, k_ulp: float, k_q50: float, rtol_floor: float) -> dict:
    """Per float output, (max, median) |diff| allowed: k x the torch 1-ulp spread's, at least rtol_floor x max |value|."""
    return {
        k: (max(k_ulp * v["max_abs"], rtol_floor * v["ref_max_abs"]), max(k_q50 * v["q50_abs"], rtol_floor * v["ref_max_abs"]))
        for k, v in ulp.items()
        if "max_abs" in v
    }


def gate(fwd: dict, losses: dict | None, tol: dict, flip_frac: float) -> dict:
    floats = {k: v for k, v in fwd.items() if "rel" in v}
    flips = {k: v for k, v in fwd.items() if "flips" in v}
    ratio = {k: max(v["max_abs"] / tol[k][0], v["q50_abs"] / tol[k][1]) for k, v in floats.items()}
    worst = max(ratio, key=ratio.get)
    res = {
        "worst_output": worst,
        "worst_rel": floats[worst]["rel"],
        "worst_over_tol": ratio[worst],
        "worst_max_over_tol": max(v["max_abs"] / tol[k][0] for k, v in floats.items()),
        "worst_q50_over_tol": max(v["q50_abs"] / tol[k][1] for k, v in floats.items()),
        "outputs_ok": all(ratio[k] <= 1 and v["sentinel_equal"] for k, v in floats.items()),
        "mask_flips": sum(v["flips"] for v in flips.values()),
        "masks_ok": all(v["flips"] <= flip_frac * v["n"] for v in flips.values()),
    }
    if losses is not None:
        res["losses_ok"] = all(v["ok"] for v in losses.values())
        res["worst_loss_term"] = max(losses, key=lambda k: losses[k]["diff"] / (LOSS_ATOL + LOSS_RTOL * abs(losses[k]["ref"])))
        res["worst_loss_diff"] = losses[res["worst_loss_term"]]["diff"]
    res["ok"] = res["outputs_ok"] and res["masks_ok"] and res.get("losses_ok", True)
    return res


def real_batch(rc, args):
    """The first ``--parity-events`` events (after the reader's cuts) as torch tensors and as native numpy arrays."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from hepattn.keras_native.data import _cast  # noqa: PLC0415, PLC2701  (the driver's own int64 -> int32 cast)

    dataset = val_dataset(rc, args.val, args.num_events)
    inputs_t, targets_t = next(iter(DataLoader(dataset, batch_size=args.parity_events, shuffle=False)))
    inputs_t = apply_cossin(inputs_t, args.cossin)
    inputs_n = {k: _cast(k, v.numpy()) for k, v in inputs_t.items()}
    targets_n = {k: _cast(k, v.numpy()) for k, v in targets_t.items()}
    return inputs_t, targets_t, inputs_n, targets_n


def torch_outputs_and_losses(model, inputs_t, targets_t):
    import torch  # noqa: PLC0415

    with torch.inference_mode():
        outputs = model(inputs_t)
        flat_out = {k: v.numpy() for k, v in flatten(outputs).items()}
        _, _, losses = model.loss(outputs, dict(targets_t))
    return flat_out, {k: float(v) for k, v in flatten(losses).items()}


def native_outputs_and_losses(rc, nmodel, inputs_n, targets_n):
    import numpy as np  # noqa: PLC0415

    from hepattn.keras_native import objective as O  # noqa: PLC0415

    outputs = nmodel(inputs_n, training=False)
    _, losses, _ = O.matched_loss(outputs, targets_n, rc.task_specs, n_jobs=1)
    return {k: np.asarray(v) for k, v in flatten(outputs).items()}, {k: float(np.asarray(v)) for k, v in flatten(losses).items()}


def apply_mutant(nmodel, spec: str) -> str:
    """``elem:<path end>:<rel>`` scales the largest-|value| element of the one variable whose path ends so by 1 + rel;
    ``tensor:<path end>:<rel>`` scales the whole variable. Returns the variable path.

    Raises:
        SystemExit: If no variable or more than one matches.
    """
    import numpy as np  # noqa: PLC0415

    kind, sub, rel = spec.split(":")
    hits = [v for v in nmodel.variables if v.path.endswith(sub)]
    if len(hits) != 1:
        raise SystemExit(f"mutant {spec}: {len(hits)} variables match {[v.path for v in hits][:6]}")
    value = np.array(hits[0].value, dtype=np.float64)
    if kind == "elem":
        idx = np.unravel_index(np.argmax(np.abs(value)), value.shape)
        value[idx] *= 1 + float(rel)
    elif kind == "tensor":
        value *= 1 + float(rel)
    else:
        raise SystemExit(f"mutant kind {kind!r}: elem or tensor")
    hits[0].assign(value.astype(hits[0].dtype))
    return hits[0].path


def cmd_parity(args) -> int:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from hepattn.experiments.clic.convert_to_native import convert  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    cfg = cpu_config(args.config, out.with_suffix(".float.yaml"))
    rc = load_config([cfg])
    tmodel, info = torch_model(cfg, args.ckpt, 1)
    inputs_t, targets_t, inputs_n, targets_n = real_batch(rc, args)
    ref_out, ref_loss = torch_outputs_and_losses(tmodel, inputs_t, targets_t)
    # 1-ulp spread of the torch model itself: every nonzero float input moved by one ulp towards +inf (zeros stay:
    # padded slots, and features that are 0 by construction, e.g. topo pt)
    ulp_in = {
        k: torch.from_numpy(np.where(v.numpy() != 0, np.nextafter(v.numpy(), np.inf), v.numpy())) if v.is_floating_point() else v
        for k, v in inputs_t.items()
    }
    ulp_out, _ = torch_outputs_and_losses(tmodel, ulp_in, targets_t)
    report = {
        "checkpoint": {"path": args.ckpt, "sha256": sha256(args.ckpt), **info},
        "events": int(inputs_t["node_features"].shape[0]),
        "cossin": args.cossin,
        "k_ulp": args.k_ulp,
        "k_q50": args.k_q50,
        "rtol_floor": args.rtol_floor,
        "flip_frac": args.flip_frac,
        "loss_tol": [LOSS_ATOL, LOSS_RTOL],
        "env": env_info(),
        "torch_1ulp": forward_errors(ref_out, ulp_out),
        "targets": {},
    }
    tol = output_tol(report["torch_1ulp"], args.k_ulp, args.k_q50, args.rtol_floor)
    report["output_tol"] = tol
    targets = [("float", [cfg])] + ([("qat_float_mode", args.qat_config)] if args.qat_config else [])
    ok = True
    for name, target in targets:
        mode = float_mode if name == "qat_float_mode" else contextlib.nullcontext
        entry = {}
        for mutant in [None, *args.mutant]:
            rc_t, nmodel, conv = convert([cfg], args.ckpt, target_config=target, verify=False)
            path = apply_mutant(nmodel, mutant) if mutant else None
            with mode():
                got_out, got_loss = native_outputs_and_losses(rc_t, nmodel, inputs_n, targets_n)
            fwd, losses = forward_errors(ref_out, got_out), loss_errors(ref_loss, got_loss)
            res = {"gate": gate(fwd, losses, tol, args.flip_frac), "outputs": fwd, "losses": losses}
            if mutant:
                res["variable"] = path
                entry.setdefault("mutants", {})[mutant] = res
            else:
                entry["base"] = res
                entry["conversion"] = {
                    k: conv[k]
                    for k in ("native_variables", "native_written", "native_kept_initial", "torch_tensors_read", "dropped_rows", "config_hash")
                }
        entry["base_ok"] = entry["base"]["gate"]["ok"]
        entry["mutants_all_fail"] = all(not m["gate"]["ok"] for m in entry.get("mutants", {}).values())
        ok &= entry["base_ok"] and entry["mutants_all_fail"]
        report["targets"][name] = entry
        print(f"RESULT {name}: base {entry['base']['gate']}", flush=True)
        for m, r in entry.get("mutants", {}).items():
            g = r["gate"]
            print(
                f"RESULT {name}: mutant {m} -> ok={g['ok']} outputs_ok={g['outputs_ok']} ({g['worst_output']} {g['worst_over_tol']:.1f}x tol;"
                f" max {g['worst_max_over_tol']:.1f}x, median {g['worst_q50_over_tol']:.1f}x) losses_ok={g['losses_ok']}"
                f" ({g['worst_loss_term']} diff {g['worst_loss_diff']:.2e})",
                flush=True,
            )
    report["ok"] = bool(ok)
    print(f"RESULT parity ok={ok}", flush=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------- W3 qinit


def leaf_ebops_report(values, leaves) -> dict:
    """Total, top 10 leaves and the sum per top-level block of per-leaf EBOPs."""
    per_leaf = sorted(((float(e), leaf.path) for e, leaf in zip(values, leaves, strict=True)), reverse=True)
    groups: dict[str, float] = {}
    for e, path in per_leaf:
        key = "/".join(path.split("/")[:2])
        groups[key] = groups.get(key, 0.0) + e
    return {"stored_sum": float(sum(e for e, _ in per_leaf)), "top10": per_leaf[:10], "by_block": dict(sorted(groups.items(), key=lambda kv: -kv[1]))}


def wrap_quantizers(model) -> list:
    """The WRAP-mode quantizers whose integer bits HGQ2 traces (``_i_decay_speed``; ``minmax_trace._reset_minmax``)."""
    return [q for q in model._flatten_layers(include_self=False, recursive=True) if hasattr(q, "_i_decay_speed") and q.trainable]  # noqa: SLF001


def cmd_qinit(args) -> int:
    """W3 (module docstring): stages on the same eval events, each next to float mode (every Quantizer the identity).

    - ``init``: the converted QAT model as ``--init-from`` loads it (spec ``i0`` / ``f0``), eval mode.
    - ``train_mode``: the forward of a first training step (``training=True``: HGQ2 sets every WRAP quantizer's
      integer bits to the batch's minimal ones, ``i_decay_speed`` inf); EBOPs = penalty / beta at beta 1 (what the
      controller reads at step 1), per leaf the stored ``_ebops``.
    - ``traced``: integer bits traced over ``--cal-events`` other events (``hgq.utils.minmax_trace``: reset, then
      ``training='tracing'``), then eval mode; EBOPs per leaf from the last tracing call.
    """
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from hgq.utils.minmax_trace import TrainingFlagWrapper, _reset_minmax  # noqa: PLC0415, PLC2701  (what trace_minmax runs)

    from hepattn.experiments.clic.convert_to_native import convert  # noqa: PLC0415
    from hepattn.keras_native import objective as O  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415
    from hepattn.keras_native.ebops import UINT32_MAX, EbopsLeaves  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    cfg = cpu_config(args.config, out.with_suffix(".float.yaml"))
    rc = load_config([cfg])
    n_eval = args.parity_events
    all_args = argparse.Namespace(**{**vars(args), "parity_events": n_eval + args.cal_events})
    _, _, inputs_all, targets_all = real_batch(rc, all_args)
    inputs_n = {k: v[:n_eval] for k, v in inputs_all.items()}
    targets_n = {k: v[:n_eval] for k, v in targets_all.items()}
    # --cal-include-eval: the eval events are traced too (control: separates unseen-event wrap from a tracing defect)
    cal = {k: (v if args.cal_include_eval else v[n_eval:]) for k, v in inputs_all.items()}
    rc_q, nq, conv = convert([cfg], args.ckpt, target_config=args.qat_config, verify=False)
    leaves = EbopsLeaves(nq)
    wraps = wrap_quantizers(nq)
    report = {
        "checkpoint": {"path": args.ckpt, "sha256": sha256(args.ckpt)},
        "qat_config": args.qat_config,
        "config_hash": conv["config_hash"],
        "renorm_lane": conv.get("renorm_lane"),
        "env": env_info(),
        "eval_events": n_eval,
        "cal_events": int(cal["node_features"].shape[0]),
        "ebops_leaves": len(leaves),
        "wrap_quantizers": len(wraps),
        "stages": {},
    }

    def record(stage: str, q_out, q_loss, f_out, f_loss, **extra) -> None:
        report["stages"][stage] = {
            "loss_quant": sum(q_loss.values()),
            "loss_float_mode": sum(f_loss.values()),
            "terms_quant": q_loss,
            "terms_float_mode": f_loss,
            "quant_vs_float_outputs": forward_errors(f_out, q_out),
            **extra,
        }
        print(f"RESULT {stage}: loss quantized {sum(q_loss.values()):.5f} float-mode {sum(f_loss.values()):.5f} ({n_eval} events)", flush=True)

    with float_mode():
        f_out, f_loss = native_outputs_and_losses(rc_q, nq, inputs_n, targets_n)

    # init: as loaded
    q_out, q_loss = native_outputs_and_losses(rc_q, nq, inputs_n, targets_n)
    record("init", q_out, q_loss, f_out, f_loss)

    # train_mode: the first training step's forward, beta 1 for the EBOPs read-out
    tv = [v.value for v in nq.trainable_variables]
    ntv = leaves.set_beta([v.value for v in nq.non_trainable_variables], 1.0)
    t_out, ntv_after, qlosses = nq.stateless_call(tv, ntv, inputs_n, training=True, return_losses=True)
    pen, _ = leaves.split_losses(qlosses)
    total = float(sum(np.asarray(p, dtype=np.float64) for p in pen))
    stored = leaves.stored_ebops(ntv_after).astype(np.float64)
    _, t_loss, _ = O.matched_loss(t_out, targets_n, rc_q.task_specs, n_jobs=1)
    t_out = {k: np.asarray(v) for k, v in flatten(t_out).items()}
    t_loss = {k: float(np.asarray(v)) for k, v in flatten(t_loss).items()}
    ebops_train = {"total": total, "stored_saturated": int((stored >= UINT32_MAX).sum()), **leaf_ebops_report(stored, leaves.leaves)}
    record("train_mode", t_out, t_loss, f_out, f_loss, ebops=ebops_train)
    print(
        f"RESULT train_mode EBOPs {total:.4e} (stored sum {ebops_train['stored_sum']:.4e}, saturated {ebops_train['stored_saturated']})", flush=True
    )

    # traced: integer bits from the calibration events, then eval mode
    _reset_minmax(nq)
    for start in range(0, report["cal_events"], args.cal_batch):
        nq({k: v[start : start + args.cal_batch] for k, v in cal.items()}, training=TrainingFlagWrapper("tracing"))
    untraced = [q.path for q in wraps if float(np.min(np.asarray(q._i.value))) <= -1e8]  # noqa: SLF001
    stored = np.asarray([float(np.asarray(leaf._ebops.value)) for leaf in leaves.leaves])  # noqa: SLF001
    ebops_traced = {"stored_saturated": int((stored >= UINT32_MAX).sum()), **leaf_ebops_report(stored, leaves.leaves)}
    q_out, q_loss = native_outputs_and_losses(rc_q, nq, inputs_n, targets_n)
    record("traced", q_out, q_loss, f_out, f_loss, ebops=ebops_traced, untraced_quantizers=untraced)
    print(
        f"RESULT traced EBOPs {ebops_traced['stored_sum']:.4e} (saturated {ebops_traced['stored_saturated']});"
        f" untraced WRAP quantizers {len(untraced)}",
        flush=True,
    )
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


# ----------------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="21_warmstart_check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("float-eval", "parity", "qinit"):
        p = sub.add_parser(name)
        p.add_argument("--config", action="append", required=True, help="the run's saved config.yaml (repeat to merge)")
        p.add_argument("--ckpt", required=True)
        p.add_argument("--val", required=True, help="validation ROOT file")
        p.add_argument("--num-events", type=int, default=5000, help="events read before the reader's cuts (data.num_val)")
        p.add_argument("--cossin", choices=["repo", "fixed"], default="repo")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--out", required=True)
        if name == "float-eval":
            p.add_argument("--batch", type=int, default=2048)
            p.add_argument("--precision", nargs="+", choices=["fp32", "bf16"], default=["fp32"])
            p.add_argument("--n-jobs", type=int, default=8, help="matcher threads")
            p.add_argument("--workers", type=int, default=0)
            p.add_argument("--hash-data", action="store_true")
        else:
            p.add_argument("--parity-events", type=int, default=32)
            p.add_argument("--qat-config", action="append", help="quantized target config (03_make_config.py output)")
        if name == "qinit":
            p.add_argument("--cal-events", type=int, default=256, help="events (after the eval events) for the integer-bit trace")
            p.add_argument("--cal-batch", type=int, default=32)
            p.add_argument("--cal-include-eval", action="store_true", help="trace the eval events as well (a control)")
        if name == "parity":
            p.add_argument("--k-ulp", type=float, default=K_ULP)
            p.add_argument("--k-q50", type=float, default=K_Q50)
            p.add_argument("--rtol-floor", type=float, default=RTOL_FLOOR)
            p.add_argument("--flip-frac", type=float, default=FLIP_FRAC)
            p.add_argument("--mutant", action="append", default=[], help="elem|tensor:<variable path end>:<rel>")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ["KERAS_BACKEND"] != "jax":
        raise SystemExit("KERAS_BACKEND must be jax")
    return {"float-eval": cmd_float_eval, "parity": cmd_parity, "qinit": cmd_qinit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
