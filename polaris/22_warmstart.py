r"""QAT warm start of the keras_native (JAX) driver from a float Lightning checkpoint: init, smoke, pilot arms, report.

Stages (``22_warmstart.pbs`` runs them on Polaris, ``WS_STAGE``):

``init``    The QAT config (``03_make_config.py`` with ``HGQ_SOURCE_CONFIG`` = the float run's saved config, fp32, one
            device, ``--weight-b0``, the PID setpoints; mini-epochs of ``--epoch-steps`` optimizer steps with a
            validation each, as G4), the converted checkpoint (``convert_to_native --target-config``) and the parity
            gate on the CPU (``21_warmstart_check.py parity``, with one mutant that must fail), then ``manifest.json``.
            The optimizer is not in the model hash, so every arm warm-starts from the same ``init.npz``.
            ``--sat-cal-batches N`` also builds a second model whose datalane quantizers saturate (``SAT``: trainable
            integer bits, no wrap; ``HGQ_DATALANE_OVERFLOW``), converts the checkpoint into it and seeds its integer
            and sign bits from the WRAP model's, traced over the first N train micro-batches (``seed-sat``):
            ``init_sat.npz``, used by the arms with ``"model": "sat"``.
``run``     One arm (``ARMS``: optimizer, peak LR, weight decay, quantizer LR scale, epsilon and L1, BetaPID settings, the
            model) with ``main_jax fit --init-from`` (``--resume last`` when the arm has a checkpoint); ``--max-steps``
            caps it (the smoke). With the init's ``--val-calibration N`` every validation is repeated with the WRAP
            integer bits traced over N train micro-batches (``val_cal/*``). ``DONE`` at K.
``report``  Per arm: the first and last logged train rows, every validation (loss, EBOPs, beta, bits), ms per
            micro-batch. ``--out`` JSON.

ws4 (LayerNorm -> DyT, then QAT; options of ``init`` / ``init-qat``): ``--norm DyT`` (every norm DyT; the converter
replaces the checkpoint's non-affine LayerNorms with DyT at its init, ``--ln-to-dyt``, and the torch parity gate is
skipped: the models differ by design), ``--datalane-axis last`` (one datalane bitwidth per channel, ``HGQ_DATALANE_AXIS``),
``--attn-scale-frac-bits F`` (the score scale rounded to F fractional bits), ``--float`` (the float model: no quant spec,
no beta / EBOPs callbacks; for the float fine-tune arms ``ws4-ft-*``).
``init-qat`` The QAT init from a float run of this script (``--from-tag``, ``--from-arm``: its ``last`` checkpoint; ``auto``:
            the DONE arm with the lowest last ``val/loss``): the QAT config (as ``init``, from the same
            ``--source-config``), whose model section must equal the float run's plus the quant spec, and the float
            weights copied into it bit for bit (``transfer``: ``convert_to_native.port_native_state``; the quantizer
            state starts at the spec's init), then ``manifest.json`` as ``init`` writes it.

    python polaris/22_warmstart.py init --root R --tag T --source-config RUN/config.yaml --ckpt RUN/x.ckpt --data-dir D \
        --model-ebops 4.8e10 --target-ebops 2.9e10 --init-beta 8.5e-11
    python polaris/22_warmstart.py run --root R --tag T --arm adamw [--max-steps 60 --run-name smoke]
    python polaris/22_warmstart.py report --root R --tag T --out logs/ws_T.json

Beta values in ``ARMS`` are what BetaPID writes and ``metrics.csv`` logs (``train/quant_beta``); the EBOPs term acts on
every 8th optimizer step only (PeriodicEBOPs), so the time-averaged pull is beta / 8, and ``--init-beta`` of the init is
in those time-averaged units (``03_make_config.py`` multiplies it by ``HGQ_EBOPS_EVERY``).
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("g4", REPO / "polaris/19_g4_compare.py")
g4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g4)

# Arms: the optimizer recipe; the model and the converted init are shared (optimizer, lrs_config and the beta callbacks
# are not in the model hash), except "model": "sat" (the init's SAT model). "pid": BetaPID init_args (logged units);
# "every": PeriodicEBOPs every_n_steps (the time-averaged pull is the logged beta / every).
ADAMW = {"optimizer": "AdamW", "max": 5e-5, "weight_decay": 1e-4}
# ws2: the clipped bitwidth gradients (~1e-10) sit below AdamW's epsilon 1e-8 (polaris/23_ws_diag.py grads), so the
# quantizer group gets epsilon 1e-12; the pilot ended at beta 8.9e-10 (train/quant_beta of adamw-q20, ws1)
WS2 = {**ADAMW, "quantizer_lr_scale": 20.0, "quantizer_epsilon": 1e-12}
ARMS = {
    "adamw": ADAMW,  # the G4-validated recipe
    "adamw-q20": {**ADAMW, "quantizer_lr_scale": 20.0},
    "lion-6e5": {"optimizer": "Lion", "max": 6.2e-5, "weight_decay": 0.1},  # the small float reference run's recipe at 1/10 of its peak LR
    "lion-6e4": {"optimizer": "Lion", "max": 6.2e-4, "weight_decay": 0.1},  # the small float reference run's recipe
    "ws2-a": {**WS2, "pid": {"init_beta": 8.9e-10}},
    "ws2-b": {**WS2, "pid": {"init_beta": 2.7e-9}},
    "ws2-c": {**WS2, "pid": {"init_beta": 8.9e-10, "i": 2.0}},
    # the EBOPs term on every step at ws2-a's time-averaged pull: with eps 1e-12 a penalty on 1 step in 8 inflates
    # AdamW's v: the replay (23_ws_diag.py replay) moves EBOPs 2.3x further with the term on every step
    "ws2-e1": {**WS2, "every": 1, "pid": {"init_beta": 8.9e-10 / 8}},
    "ws2-sat": {**WS2, "pid": {"init_beta": 8.9e-10}, "model": "sat"},
}
# ws3: a loss-vs-EBOPs scan from the ws2-e1 recipe (EBOPs term on every step, bitwidth epsilon 1e-12), with padded
# tokens zeroed before the WRAP lanes. In ws2 the EBOPs went below the 2.9e10 target and kept falling with beta at the
# PID floor: HGQ2's bit L1 (MonoL1 1e-8, at ~lr steps with epsilon 1e-12) set the end point. The axis is therefore the
# BetaPID target, one decade apart from ws2-e1's end (1.6e10) down to one VU13P fully unrolled (1.728e6 CLB LUTs, AMD
# DS890 v4.10 Table 15; 1 EBOP ~ 1 LUT, Chang Sun 2026-11-02), with the L1 at HGQ2's default. Beta starts where the
# EBOPs pull per bit equals the per-batch task-gradient noise of the median f (23_ws_diag.py grads: logged 9.8e-9 at
# every 8, i.e. 1.2e-9 per step) and may rise to 1e-5 (HGQ2 penalty per bit = beta x the partner bits, median ~570).
WS3 = {**WS2, "every": 1}


def _target(target: float, **pid) -> dict:
    return {"pid": {"target_ebops": target, "init_beta": 1e-9, "max_beta": 1e-5, "min_beta": 1e-12, **pid}}


ARMS.update({
    "ws3-t9": {**WS3, **_target(1.6e9)},
    "ws3-t8": {**WS3, **_target(1.6e8)},
    "ws3-t7": {**WS3, **_target(1.6e7)},
    "ws3-t6": {**WS3, **_target(1.6e6)},
    # Chang's bitwidth LR scale 3-5 (here 4), at the two nearest targets: an Adam-type step moves a bit by at most
    # ~lr x scale, so 4 allows ~5x less bitwidth change than 20 in the same steps
    "ws3-t9-q4": {**WS3, "quantizer_lr_scale": 4.0, **_target(1.6e9)},
    "ws3-t8-q4": {**WS3, "quantizer_lr_scale": 4.0, **_target(1.6e8)},
    # control: HGQ2's bit L1 off (lrs_config.quantizer_l1 0), beta alone
    "ws3-t8-l0": {**WS3, "quantizer_l1": 0.0, **_target(1.6e8)},
    "ws3-t5e8": {**WS3, **_target(5e8)},  # half a decade between t9 and t8
})  # fmt: skip
# ws4: the float fine-tune of the LN -> DyT model (float arms: no beta), then the ws3 scan on the DyT model with one
# datalane bitwidth per channel and the rounded attention scale. Fine-tune: AdamW at 2x / 6x / 1x ws2's peak LR
# (5e-5), weight decay 1e-4; Lion at 1/10 of the float reference run's peak (6.2e-4, weight decay 0.1).
ARMS.update({
    "ws4-ft-a": {**ADAMW, "max": 1e-4},
    "ws4-ft-b": {**ADAMW, "max": 3e-4},
    "ws4-ft-c": ADAMW,
    "ws4-ft-d": {"optimizer": "Lion", "max": 6.2e-5, "weight_decay": 0.1},
    "ws4-t9": {**WS3, **_target(1.6e9)},
    "ws4-t8": {**WS3, **_target(1.6e8)},
    "ws4-t7": {**WS3, **_target(1.6e7)},
    "ws4-t6": {**WS3, **_target(1.6e6)},
})  # fmt: skip
FLOAT_DROPPED_CALLBACKS = ("BetaPID", "BetaScheduler", "PeriodicEBOPs", "EBOPsMonitor", "BitwidthMonitor")
COLUMNS = (
    "train/loss", "train/ebops", "val/loss", "val/ebops", "val_cal/loss", "val_cal/ebops", "train/quant_beta", "train/pid_ebops",
    "val/bits_f_mean", "val/bits_b_mean", "val/bits_i_mean", "time/ms_per_micro_batch",
)  # fmt: skip


def ws4_env(args) -> dict[str, str]:
    """The ``03_make_config.py`` variables of the ws4 options that are set."""
    env = {}
    if getattr(args, "norm", None):
        env["HGQ_NORM"] = args.norm
    if getattr(args, "datalane_axis", None):
        env["HGQ_DATALANE_AXIS"] = args.datalane_axis
    if getattr(args, "attn_scale_frac_bits", None) is not None:
        env["HGQ_ATTN_SCALE_FRAC_BITS"] = str(args.attn_scale_frac_bits)
    return env


def floatify(cfg: dict) -> dict:
    """The float twin of a generated QAT config: no quant spec, no beta or EBOPs callbacks (``FLOAT_DROPPED_CALLBACKS``)."""
    cfg["model"]["model"]["init_args"].pop("quant", None)
    t = cfg["trainer"]
    t["callbacks"] = [cb for cb in t["callbacks"] if not cb.get("class_path", "").endswith(FLOAT_DROPPED_CALLBACKS)]
    return cfg


def make_config(args, init_dir: Path, name: str = "config", **extra_env: str) -> Path:
    import yaml  # noqa: PLC0415

    env = {
        "HGQ_SOURCE_CONFIG": str(Path(args.source_config).resolve()),
        "HGQ_MODEL_EBOPS": str(args.model_ebops),
        "HGQ_TARGET_EBOPS": str(args.target_ebops),
        "HGQ_INIT_BETA": str(args.init_beta),
        "HGQ_WEIGHT_B0": str(args.weight_b0),
        "HGQ_PRECISION": "32",
        "HGQ_DEVICES": "1",
        "HGQ_BATCH": str(args.batch),
        "HGQ_ACCUM": str(args.accum),
        "HGQ_PID_WARMUP_EPOCHS": str(args.pid_warmup_epochs),
        "DATA_DIR": args.data_dir,
        "CKPT_DIR": str(init_dir / "unused_ckpts"),
        **ws4_env(args),
        **extra_env,
    }
    path = g4.generate_config(init_dir / f"generated_{name}.yaml" if name != "config" else init_dir / "generated.yaml", **env)
    cfg = yaml.safe_load(path.read_text())
    s = {
        "k": args.k,
        "epoch_steps": args.epoch_steps,
        "accum": args.accum,
        "log_every": args.log_every,
        "workers": args.workers,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "ckpt_every": args.ckpt_every,
        "pid_warmup_epochs": args.pid_warmup_epochs,
    }
    cfg = g4.g4_overrides(cfg, s, init_dir / "unused_ckpts")
    if getattr(args, "float", False):
        cfg = floatify(cfg)
    out = init_dir / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def cmd_init(args) -> int:
    root = Path(args.root) / args.tag
    init_dir = root / "init"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        raise SystemExit(f"{manifest_path} is complete; a new init needs a new --tag")
    init_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, init_dir)
    check_onecycle(config, args.k)
    log = init_dir / "init.log"
    cpu = g4.env_with_src(KERAS_BACKEND="jax", JAX_PLATFORMS="cpu")
    convert = [args.jax_python, "-m", "hepattn.experiments.clic.convert_to_native", "--config", args.source_config, "--ckpt", args.ckpt]
    convert += ["--target-config", config, "--out", init_dir / "init.npz", "--report", init_dir / "convert.json", "--overwrite"]
    if args.norm:
        convert += ["--ln-to-dyt"]
    if g4.run_logged(convert, cpu, log) != 0:
        raise SystemExit(f"conversion failed: {log}")
    conv = json.loads((init_dir / "convert.json").read_text())
    if args.norm or args.float:
        # the torch parity gate compares the torch model with the converted float and QAT targets: a DyT target differs
        # from the LayerNorm source by design, and a float config has no QAT target
        code, report = 0, {"ok": None, "targets": {}, "skipped": "norm replaced or float target"}
    else:
        parity = [args.jax_python, REPO / "polaris/21_warmstart_check.py", "parity", "--config", args.source_config, "--ckpt", args.ckpt]
        parity += ["--val", f"{args.data_dir}/val_clic_fix.root", "--num-events", "200", "--parity-events", "32", "--threads", "8"]
        parity += ["--qat-config", config, "--mutant", "tensor:register_tokens:1e-3", "--out", init_dir / "parity.json"]
        code = g4.run_logged(parity, cpu, log)
        report = json.loads((init_dir / "parity.json").read_text())
    manifest = {
        "complete": code == 0,
        "git": g4.git_head(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(config),
        "native_init": str(init_dir / "init.npz"),
        "source": {"config": args.source_config, "ckpt": args.ckpt, "ckpt_sha256": conv["checkpoint"]["sha256"]},
        "settings": {k: v for k, v in vars(args).items() if k not in {"func", "command"}},
        "config_hash": conv["config_hash"],
        "renorm_lane": conv.get("renorm_lane"),
        "ln_to_dyt": len(conv.get("ln_to_dyt") or []),
        "parity_ok": report["ok"],
        "parity_worst": {t: e["base"]["gate"] for t, e in report["targets"].items()},
        "files_sha256": {p.name: g4.sha256(p) for p in (config, init_dir / "init.npz", init_dir / "parity.json") if p.is_file()},
    }
    if args.sat_cal_batches and manifest["complete"]:
        manifest["sat"] = make_sat(args, init_dir, config, log)
        manifest["complete"] = manifest["sat"]["ok"]
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(json.dumps({"complete": manifest["complete"], "parity_ok": manifest["parity_ok"], "config_hash": manifest["config_hash"]}))
    return 0 if manifest["complete"] else 1


def best_arm(root: Path) -> str:
    """The DONE arm of ``root`` with the lowest last ``val/loss`` (``metrics.csv``).

    Raises:
        SystemExit: If no arm is DONE.
    """
    scored = []
    for arm_dir in sorted(p for p in root.iterdir() if (p / "DONE").is_file() and (p / "metrics.csv").is_file()):
        rows = [r for r in g4.read_rows(arm_dir / "metrics.csv") if math.isfinite(r.get("val/loss", math.nan))]
        if rows:
            scored.append((rows[-1]["val/loss"], arm_dir.name))
    if not scored:
        raise SystemExit(f"no DONE arm with a val/loss in {root}")
    return min(scored)[1]


def model_section_without_quant(config: Path) -> dict:
    import yaml  # noqa: PLC0415

    init_args = dict(yaml.safe_load(Path(config).read_text())["model"]["model"]["init_args"])
    init_args.pop("quant", None)
    return init_args


def cmd_init_qat(args) -> int:
    """``init-qat`` (module docstring): QAT config, transfer of the float run's last (or ``--from-ckpt``) weights, manifest.

    Raises:
        SystemExit: For a complete manifest of ``--tag``, ``--float``, or a QAT model section that is not the float run's.
    """
    root = Path(args.root) / args.tag
    init_dir = root / "init"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        raise SystemExit(f"{manifest_path} is complete; a new init needs a new --tag")
    if args.float:
        raise SystemExit("init-qat builds the QAT model: no --float")
    src_root = Path(args.root) / args.from_tag
    arm = best_arm(src_root) if args.from_arm == "auto" else args.from_arm
    float_config = src_root / arm / "arm_config.yaml"
    if args.from_ckpt:
        float_ckpt = Path(args.from_ckpt)
    else:
        last = json.loads((src_root / arm / "ckpts" / "last.json").read_text())
        float_ckpt = src_root / arm / "ckpts" / last["file"]
    init_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, init_dir)
    check_onecycle(config, args.k)
    if model_section_without_quant(config) != model_section_without_quant(float_config):
        raise SystemExit(
            f"the QAT model section ({config}) is not the float run's ({float_config}) plus a quant spec: check --norm / --attn-scale-frac-bits"
        )
    cmd = [args.jax_python, REPO / "polaris/22_warmstart.py", "transfer", "--float-config", float_config, "--float-ckpt", float_ckpt]
    cmd += ["--qat-config", config, "--out", init_dir / "init.npz", "--report", init_dir / "transfer.json"]
    code = g4.run_logged(cmd, g4.env_with_src(KERAS_BACKEND="jax", JAX_PLATFORMS="cpu"), init_dir / "init.log")
    conv = json.loads((init_dir / "transfer.json").read_text()) if code == 0 else {}
    manifest = {
        "complete": code == 0,
        "git": g4.git_head(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(config),
        "native_init": str(init_dir / "init.npz"),
        "source": {"float_tag": args.from_tag, "float_arm": arm, "float_config": str(float_config), "float_ckpt": str(float_ckpt),
                   "float_ckpt_sha256": g4.sha256(float_ckpt)},
        "settings": {k: v for k, v in vars(args).items() if k not in {"func", "command"}},
        "config_hash": conv.get("config_hash"),
        "transfer": conv.get("report"),
        "parity_ok": None,
        "files_sha256": {p.name: g4.sha256(p) for p in (config, init_dir / "init.npz") if p.is_file()},
    }  # fmt: skip
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(json.dumps({"complete": manifest["complete"], "float_arm": arm, "config_hash": manifest["config_hash"]}))
    return 0 if manifest["complete"] else 1


def cmd_transfer(args) -> int:
    """(called by ``init-qat``) The float checkpoint's variables into the QAT model of ``--qat-config``, bit for bit.

    Raises:
        SystemExit: If a copied value differs from its source (a check of the written model).
    """
    import numpy as np  # noqa: PLC0415

    from hepattn.experiments.clic.convert_to_native import port_native_state, sample_batch, write_native_checkpoint  # noqa: PLC0415
    from hepattn.keras_native import checkpoint as nckpt  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    rc_f, rc_q = load_config([args.float_config]), load_config([args.qat_config])
    inputs, _ = sample_batch(rc_q, 2)
    fmodel, qmodel = rc_f.build_model(), rc_q.build_model()
    fmodel(inputs, training=False)
    qmodel(inputs, training=False)
    (tv, ntv, _), meta = nckpt.load_checkpoint(args.float_ckpt, fmodel, None, expect_config_hash=rc_f.model_hash)
    report = port_native_state(fmodel, (tv, ntv), qmodel)
    report.update(float_ckpt=str(args.float_ckpt), float_step=meta.get("step"), float_config_hash=rc_f.model_hash)
    # bit-for-bit check of the copy on the written model: every float value equals the QAT variable of its key
    fkeys, qkeys = nckpt.variable_keys(fmodel), nckpt.variable_keys(qmodel)
    qvalues = {qkeys[id(v)]: np.asarray(v.value) for v in qmodel.variables}
    fvars = [*fmodel.trainable_variables, *fmodel.non_trainable_variables]
    report["bitwise_equal"] = all(np.array_equal(np.asarray(x), qvalues[fkeys[id(v)]]) for v, x in zip(fvars, [*tv, *ntv], strict=True))
    if not report["bitwise_equal"]:
        raise SystemExit("transfer: a copied value differs")
    write_native_checkpoint(args.out, rc_q, qmodel, report, overwrite=True)
    Path(args.report).write_text(json.dumps({"config_hash": rc_q.model_hash, "report": report}, indent=1, default=str) + "\n")
    print("RESULT " + json.dumps(report, default=str), flush=True)
    return 0


def make_sat(args, init_dir: Path, wrap_config: Path, log: Path) -> dict:
    """The SAT-datalane model: config, conversion (``init_sat_spec.npz``), integer bits seeded from the WRAP model.

    Raises:
        SystemExit: If the conversion fails.
    """
    config = make_config(args, init_dir, name="config_sat", HGQ_DATALANE_OVERFLOW="SAT")
    cpu = g4.env_with_src(KERAS_BACKEND="jax", JAX_PLATFORMS="cpu")
    spec = init_dir / "init_sat_spec.npz"
    convert = [args.jax_python, "-m", "hepattn.experiments.clic.convert_to_native", "--config", args.source_config, "--ckpt", args.ckpt]
    convert += ["--target-config", config, "--out", spec, "--report", init_dir / "convert_sat.json", "--overwrite"]
    if g4.run_logged(convert, cpu, log) != 0:
        raise SystemExit(f"SAT conversion failed: {log}")
    seed = [args.jax_python, REPO / "polaris/22_warmstart.py", "seed-sat", "--wrap-config", wrap_config, "--wrap-init", init_dir / "init.npz"]
    seed += ["--sat-config", config, "--sat-init", spec, "--n", str(args.sat_cal_batches), "--out", init_dir / "init_sat.npz"]
    seed += ["--report", init_dir / "seed_sat.json", "--work-dir", init_dir / "seed_sat_run"]
    code = g4.run_logged(seed, g4.env_with_src(KERAS_BACKEND="jax"), log)
    report = json.loads((init_dir / "seed_sat.json").read_text()) if code == 0 else {}
    print(json.dumps({"sat_seed": report.get("summary")}), flush=True)
    return {
        "ok": code == 0 and bool(report.get("ok")),
        "config": str(config),
        "native_init": str(init_dir / "init_sat.npz"),
        "config_hash": report.get("config_hash"),
        "seed": report.get("summary"),
    }


def cmd_seed_sat(args) -> int:
    """``init_sat.npz``: the SAT model at the converted weights with each datalane quantizer's integer bits (trainable
    here) and sign bits set to the WRAP model's, traced over the first ``--n`` train micro-batches
    (``loop.Trainer.calibrated_ntv``). Variables are matched by their checkpoint key (auto-numbered names stripped).
    Report: the counts, the matched loss and EBOPs of one training-mode forward of each model on the first calibration
    batch (gate: losses within 1e-3 relative).
    """
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from hepattn.experiments.clic.main_jax import ClicSource  # noqa: PLC0415
    from hepattn.keras_native import checkpoint as nckpt  # noqa: PLC0415
    from hepattn.keras_native import loop  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415
    from hepattn.keras_native.ebops import EbopsLeaves  # noqa: PLC0415
    from hepattn.keras_native.objective import matched_loss  # noqa: PLC0415

    rc = load_config([args.wrap_config])
    train = ClicSource(rc, "train", seed=42)
    sample = next(iter(train.epoch(0)))
    t = loop.Trainer(rc, loop.LoopSettings.from_run_config(rc, args.work_dir), sample=sample, train_batches=train.num_batches)
    t.warm_start(args.wrap_init)
    t._set_beta(1.0)  # noqa: SLF001  (EBOPs read-out = penalty)
    ntv, ebops_wrap_traced = t.calibrated_ntv(train, args.n)
    wrap = t.steps.model
    keys = nckpt.variable_keys(wrap)
    i_pos, k_pos = t._wrap_index  # noqa: SLF001
    # by position, not by name: a repeated key ends in "#n" (checkpoint._numbered)
    role = {keys[id(wrap.non_trainable_variables[p])]: r for r, ps in (("i", i_pos), ("k", k_pos)) for p in ps}
    traced = {keys[id(wrap.non_trainable_variables[p])]: np.asarray(ntv[p]) for p in (*i_pos, *k_pos)}
    untraced = [k for k, r in role.items() if r == "i" and float(traced[k].min()) <= -1e8]

    rc_s = load_config([args.sat_config])
    model = rc_s.build_model()
    model(sample.inputs, training=False)
    (tv, ntv_s, _), meta = nckpt.load_checkpoint(args.sat_init, model, None, expect_config_hash=rc_s.model_hash)
    keys_s = nckpt.variable_keys(model)
    tv, ntv_s, set_i, still_wrap, set_k = list(tv), list(ntv_s), 0, 0, 0
    for group, variables in ((tv, model.trainable_variables), (ntv_s, model.non_trainable_variables)):
        for n, var in enumerate(variables):
            key = keys_s[id(var)]
            if role.get(key) == "i":  # SAT: trainable; a quantizer that stays WRAP (not a datalane): set from the data anyway
                group[n] = np.clip(traced[key], -23, 23).astype(group[n].dtype)  # kif datalane ic = MinMax(-23, 23)
                set_i, still_wrap = set_i + var.trainable, still_wrap + (not var.trainable)
            elif role.get(key) == "k":
                group[n] = traced[key].astype(group[n].dtype)
                set_k += 1
    meta = {**meta, "seeded_from": {"wrap_init": str(args.wrap_init), "calibration_batches": args.n}}
    nckpt.save_checkpoint(args.out, model, None, (tv, ntv_s, []), meta, overwrite=True)

    def forward(m, tv_, ntv_):
        leaves = EbopsLeaves(m)

        @jax.jit  # eagerly, Keras' _clear_losses matches EBOPs terms by id() of concrete arrays (reused ids)
        def run(tv_, ntv_):
            with leaves.enabled(True):
                out, _, losses = m.stateless_call(tv_, ntv_, sample.inputs, training=True, return_losses=True)
            return matched_loss(out, sample.targets, rc.task_specs, n_jobs=rc.matcher_n_jobs)[0], leaves.split_losses(losses)[0]

        loss, terms = run(tv_, leaves.set_beta(ntv_, 1.0))
        return float(loss), float(sum(np.asarray(p, np.float64) for p in terms))

    # the first train batch is a calibration batch: inside the SAT ranges, so the SAT forward is the WRAP one without
    # overflow (WRAP lanes only round in training) and the losses must agree. Both at the traced bits: the kbi WRAP
    # quantizers (biases) keep b, so their stored i (the spec's i0 in the converted init) sets their fraction bits
    loss_sat, ebops_sat = forward(model, [jax.numpy.asarray(x) for x in tv], ntv_s)
    loss_wrap, ebops_wrap = forward(wrap, t.state.tv, ntv)
    summary = {
        "wrap_quantizers": len(i_pos), "untraced": len(untraced), "set_i_sat": set_i, "set_i_still_wrap": still_wrap, "set_k": set_k,
        "loss_train_mode_wrap": loss_wrap, "loss_train_mode_sat": loss_sat, "ebops_wrap": ebops_wrap, "ebops_sat": ebops_sat,
        "ebops_wrap_traced": ebops_wrap_traced,
    }  # fmt: skip
    ok = set_i > 0 and set_i + still_wrap == len(i_pos) == set_k and not untraced and abs(loss_sat - loss_wrap) <= 1e-3 * abs(loss_wrap)
    Path(args.report).write_text(json.dumps({"ok": ok, "config_hash": rc_s.model_hash, "summary": summary, "untraced": untraced}, indent=1) + "\n")
    print(json.dumps({"ok": ok, **summary}), flush=True)
    return 0 if ok else 1


def check_onecycle(config: Path, total_steps: int) -> None:
    """OneCycle needs pct_start x total steps > 1 (``optim.OneCycleLR``, as torch); fail before any job time is spent.

    Raises:
        SystemExit: If the run is too short for the config's ``pct_start``.
    """
    import yaml  # noqa: PLC0415

    pct = float(yaml.safe_load(config.read_text())["model"]["lrs_config"]["pct_start"])
    if pct * total_steps <= 1:
        raise SystemExit(f"pct_start {pct} x {total_steps} steps <= 1: OneCycle needs more steps (> {1 / pct:.0f})")


def arm_model(manifest: dict, arm: str) -> dict:
    """``{"config", "native_init"}`` of the model an arm trains (the init's, or its SAT model).

    Raises:
        SystemExit: For a SAT arm when the init built no SAT model.
    """
    if ARMS[arm].get("model") == "sat":
        if "sat" not in manifest:
            raise SystemExit(f"arm {arm} needs the SAT model: run the init with --sat-cal-batches")
        return manifest["sat"]
    return manifest


def arm_config(manifest: dict, arm: str, arm_dir: Path) -> Path:
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(Path(arm_model(manifest, arm)["config"]).read_text())
    recipe = ARMS[arm]
    cfg["model"]["optimizer"] = recipe["optimizer"]
    lrs = cfg["model"]["lrs_config"]
    lrs["max"], lrs["weight_decay"] = recipe["max"], recipe["weight_decay"]
    for key in ("quantizer_lr_scale", "quantizer_epsilon", "quantizer_l1"):
        if key in recipe:
            lrs[key] = recipe[key]
    for key, name, update in (("pid", "BetaPID", recipe.get("pid")), ("every", "PeriodicEBOPs", {"every_n_steps": recipe.get("every")})):
        if key not in recipe:
            continue
        found = [cb for cb in cfg["trainer"]["callbacks"] if cb.get("class_path", "").endswith(name)]
        if len(found) != 1:
            raise SystemExit(f"arm {arm} sets {name} arguments but the config has {len(found)} {name} callbacks")
        found[0].setdefault("init_args", {}).update(update)
    out = arm_dir / "arm_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def cmd_run(args) -> int:
    root = Path(args.root) / args.tag
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit("the init is not complete (manifest.json): run the init stage first")
    arm_dir = root / (args.run_name or args.arm)
    arm_dir.mkdir(parents=True, exist_ok=True)
    k = args.max_steps or manifest["settings"]["k"]
    if (arm_dir / "DONE").is_file():
        print(json.dumps({"arm": args.arm, "done": True}))
        return 0
    config = arm_config(manifest, args.arm, arm_dir)
    check_onecycle(config, k)
    cmd = [args.jax_python, "-m", "hepattn.experiments.clic.main_jax", "fit", "--config", config, "--run-dir", arm_dir]
    cmd += ["--seed", "42", "--matmul-precision", "highest", "--compile-cache", args.compile_cache]
    cmd += ["--ckpt-every-n-steps", str(manifest["settings"]["ckpt_every"])]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.num_val:
        cmd += ["--num-val", str(args.num_val)]
    if manifest["settings"].get("val_calibration"):
        cmd += ["--val-calibration", str(manifest["settings"]["val_calibration"])]
    last = arm_dir / "ckpts" / "last.json"
    cmd += ["--resume", "last"] if last.is_file() else ["--init-from", arm_model(manifest, args.arm)["native_init"]]
    t0 = time.time()
    code = g4.run_logged(cmd, g4.env_with_src(KERAS_BACKEND="jax"), arm_dir / "logs" / f"{os.environ.get('PBS_JOBID', 'local')}.log")
    end = g4.progress_step(arm_dir)
    if code == 0 and end >= k:
        (arm_dir / "DONE").write_text(f"{end}\n")
    segment = {"job": os.environ.get("PBS_JOBID", "local"), "end_step": end, "wall_s": time.time() - t0, "returncode": code}
    with (arm_dir / "segments.jsonl").open("a") as f:
        f.write(json.dumps(segment) + "\n")
    print(json.dumps({"arm": args.arm, **segment}))
    return code


def pick(row: dict) -> dict:
    return {c: row[c] for c in COLUMNS if c in row and not (isinstance(row[c], float) and math.isnan(row[c]))}


def cmd_report(args) -> int:
    root = Path(args.root) / args.tag
    out = {"tag": args.tag, "arms": {}}
    for arm_dir in sorted(p for p in root.iterdir() if (p / "steps.csv").is_file()):
        steps, metrics = g4.read_rows(arm_dir / "steps.csv"), g4.read_rows(arm_dir / "metrics.csv")
        ms = [r["time/ms_per_micro_batch"] for r in steps if not math.isnan(r.get("time/ms_per_micro_batch", math.nan))]
        # updates the non-finite guard skipped (cumulative; None: guard off or a run before the guard)
        skips = [r["train/nonfinite_skipped"] for r in [*steps, *metrics] if not math.isnan(r.get("train/nonfinite_skipped", math.nan))]
        out["arms"][arm_dir.name] = {
            "steps_logged": len(steps),
            "first_train": {"step": steps[0].get("step"), **pick(steps[0])} if steps else None,
            "last_train": {"step": steps[-1].get("step"), **pick(steps[-1])} if steps else None,
            "validations": [{"step": r.get("step"), "epoch": r.get("epoch"), **pick(r)} for r in metrics],
            "median_ms_per_micro_batch": statistics.median(ms) if ms else None,
            "nonfinite_train_loss": sum(1 for r in steps if not math.isfinite(r.get("train/loss", 0.0))),
            "nonfinite_skipped": int(max(skips)) if skips else None,
            "done": (arm_dir / "DONE").is_file(),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for name, a in out["arms"].items():
        last_val = a["validations"][-1] if a["validations"] else {}
        print(
            f"{name:10s} steps {a['last_train']['step'] if a['last_train'] else 0:>6} first train/loss "
            f"{(a['first_train'] or {}).get('train/loss', float('nan')):.4f} last val/loss {last_val.get('val/loss', float('nan')):.4f} "
            f"val/ebops {last_val.get('val/ebops', float('nan')):.3e} ms/micro-batch {a['median_ms_per_micro_batch']}"
            f" nonfinite {a['nonfinite_train_loss']} skipped {a['nonfinite_skipped']}"
        )
    return 0


def _init_options(p) -> None:
    """Options shared by ``init`` and ``init-qat`` (the generated config)."""
    p.add_argument("--source-config", required=True, help="the float Lightning run's saved config.yaml")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-ebops", type=float, required=True, help="EBOPs of the QAT model at the converted weights")
    p.add_argument("--target-ebops", type=float, required=True)
    p.add_argument("--init-beta", type=float, required=True)
    p.add_argument("--weight-b0", type=int, default=12)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--k", type=int, default=2000, help="optimizer steps per arm")
    p.add_argument("--epoch-steps", type=int, default=250, help="optimizer steps per mini-epoch (one validation each)")
    p.add_argument("--pid-warmup-epochs", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--num-train", type=int, default=400000, help="train events read (-1: the whole file)")
    p.add_argument("--num-val", type=int, default=5000)
    p.add_argument("--val-calibration", type=int, default=0, help="train micro-batches traced for val_cal/* (0: none)")
    p.add_argument("--norm", choices=["DyT"], help="ws4: every norm DyT (a LayerNorm checkpoint converts with --ln-to-dyt)")
    p.add_argument("--datalane-axis", choices=["last"], help="ws4: one datalane bitwidth per index of the last axis")
    p.add_argument("--attn-scale-frac-bits", type=int, help="ws4: the attention scale rounded to this many fractional bits")
    p.add_argument("--float", action="store_true", help="ws4: the float model (no quant spec, no beta callbacks)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="22_warmstart", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "init-qat", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--tag", required=True)
        if name in {"init", "init-qat", "run"}:
            p.add_argument("--jax-python", default=sys.executable)
    seed = sub.add_parser("seed-sat", help="(called by init) the SAT model's integer bits from a traced WRAP model")
    for key in ("--wrap-config", "--wrap-init", "--sat-config", "--sat-init", "--out", "--report", "--work-dir"):
        seed.add_argument(key, required=True)
    seed.add_argument("--n", type=int, required=True, help="train micro-batches traced")
    tr = sub.add_parser("transfer", help="(called by init-qat) float checkpoint -> QAT init")
    for key in ("--float-config", "--float-ckpt", "--qat-config", "--out", "--report"):
        tr.add_argument(key, required=True)
    for name in ("init", "init-qat"):
        _init_options(sub.choices[name])
    init = sub.choices["init"]
    init.add_argument("--ckpt", required=True, help="its Lightning checkpoint")
    init.add_argument("--sat-cal-batches", type=int, default=0, help="also build the SAT-datalane model, seeded from N batches")
    iq = sub.choices["init-qat"]
    iq.add_argument("--from-tag", required=True, help="the float run's tag (an init with --float)")
    iq.add_argument("--from-arm", default="auto", help="its arm; auto: the DONE arm with the lowest last val/loss")
    iq.add_argument("--from-ckpt", help="a checkpoint of that arm instead of its last one")
    run = sub.choices["run"]
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--run-name", help="run directory name (default: the arm)")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--num-val", type=int)
    run.add_argument("--compile-cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    sub.choices["report"].add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    commands = {"init": cmd_init, "init-qat": cmd_init_qat, "transfer": cmd_transfer, "run": cmd_run, "report": cmd_report, "seed-sat": cmd_seed_sat}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
