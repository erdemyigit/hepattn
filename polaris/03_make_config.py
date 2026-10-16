#!/usr/bin/env python
"""Generate the Polaris HGQ2 training config from the repo's CLIC configs.

Run from the repo root, inside the venv:
    python 03_make_config.py

Writes ./polaris_hgq.yaml. The model is the org float Linformer model (org/main ac0da8f):
configs/base.yaml with configs/linformer.yaml merged over it, with the network swapped for
KerasMaskFormer, the HGQ2 quantization spec added, and the matcher solver set to scipy.
Everything Polaris-specific (data paths, checkpoint dir, batch size, precision) is set here
so the repo's committed configs stay clean.

Env overrides:
    HGQ_ATTN       linformer | torch              (default linformer: base.yaml + linformer.yaml;
                                                   torch: base.yaml alone, dense attention, no sorting)
    HGQ_BATCH      micro-batch per GPU            (default 32)
    HGQ_ACCUM      gradient accumulation steps    (default 4)
    HGQ_CONFIG_OUT output path                    (default ./polaris_hgq.yaml)
    HGQ_DEVICES    GPUs for one DDP run           (default 4; validated run 7598515)
    HGQ_PRECISION  bf16-mixed | 32                (default bf16-mixed)
    HGQ_BETA_MODE  pid | schedule                 (default pid)
    HGQ_BETA_END   final EBOPs penalty weight     (default 4.0e-15, schedule mode only; NOT re-derived for the
                                                   org model, see docs/hgq/BETA_SETPOINTS.md)
    HGQ_TARGET_EBOPS   EBOPs setpoint for pid mode    (default 1.5e11; PROVISIONAL, docs/hgq/BETA_SETPOINTS.md)
    HGQ_INIT_BETA      starting beta for pid mode     (default 8.7e-11; PROVISIONAL)
    HGQ_MAX_BETA       pid beta ceiling               (default 3.3e-8; PROVISIONAL)
    HGQ_MIN_BETA       pid beta floor, > 0            (default 2.3e-13; PROVISIONAL)
    HGQ_PID_I          pid integral gain              (default 0.5)
    HGQ_BETA_START     schedule-mode ramp start, > 0  (default 1e-3 x HGQ_BETA_END)
    HGQ_PID_WARMUP_EPOCHS  epochs before pid engages  (default 2)
    HGQ_EPOCHS     max epochs                     (default 20)
    HGQ_WARMUP_EPOCHS  beta ramp length, in epochs (default 1.0)
    DATA_DIR       CLIC ROOT directory            (default $HGQ_DATA or ./data/clic)
    CKPT_DIR       checkpoint output directory    (default ./hgq_ckpts)
"""

import math
import os
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent  # polaris/ -> repo root
CONFIGS = REPO / "src/hepattn/experiments/clic/configs"
SRC = CONFIGS / "base.yaml"
OVERLAY = CONFIGS / "linformer.yaml"
DST = pathlib.Path(os.environ.get("HGQ_CONFIG_OUT", REPO / "polaris_hgq.yaml"))
if not SRC.exists() or not OVERLAY.exists():
    raise SystemExit(f"cannot find {SRC} or {OVERLAY} — run this from the repo (it lives in polaris/)")
ATTN = os.environ.get("HGQ_ATTN", "linformer").lower()
if ATTN not in {"linformer", "torch"}:
    raise SystemExit(f"HGQ_ATTN must be 'linformer' or 'torch', got {ATTN!r}")

DATA_DIR = os.environ.get("DATA_DIR", os.environ.get("HGQ_DATA", str(REPO / "data/clic")))
CKPT_DIR = os.environ.get("CKPT_DIR", str(REPO / "hgq_ckpts"))
SCALE = str(REPO / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml")

BATCH = int(os.environ.get("HGQ_BATCH", "32"))
ACCUM = int(os.environ.get("HGQ_ACCUM", "4"))
PRECISION = os.environ.get("HGQ_PRECISION", "bf16-mixed")
BETA_MODE = os.environ.get("HGQ_BETA_MODE", "pid").lower()
if BETA_MODE not in {"pid", "schedule"}:
    raise SystemExit(f"HGQ_BETA_MODE must be 'pid' or 'schedule', got {BETA_MODE!r}")
BETA_END = float(os.environ.get("HGQ_BETA_END", "4.0e-15"))

# --- pid mode -------------------------------------------------------------------
# Closed-loop control of beta against a fixed EBOPs setpoint, as in Laatu, Sun et al.
# (arXiv:2510.24784), which trains every model to 350k EBOPs. Open-loop scheduling
# needs the right beta known in advance; we do not have it, and calibrating it from
# the reported total_ebops overstated the loss slope by 10,410x.
#
# PROVISIONAL setpoints for the org/main model (masked Linformer k=64), re-derived on 2026-10-16
# in docs/hgq/BETA_SETPOINTS.md (S5); confirm them with a Polaris beta scan before a sweep.
# The pre-retarget values (1.0e15 / 2.6e-10 / 1.0e-7) belonged to a model whose 1.6755e15 EBOPs were
# 99.98% decoder softmax tables that beta never reached, and INIT_BETA came from a scan at the old LR.
#
# MODEL_EBOPS: total EBOPs of this generated model at its initial bitwidths (exact torch-backend sum over
# every HGQ2 leaf, CLIC dummy events: 2.509e11 / 2.512e11 / 2.512e11; the native JAX read-out agrees;
# Polaris G2 on real events: 2.513e11). The old model measured 1.6755e15.
MODEL_EBOPS = 2.511e11
OLD_MODEL_EBOPS = 1.6755e15
# Setpoint: the documented 40%-cut rule of the old setpoint (1.0e15 / 1.6755e15 = 0.597), applied to
# MODEL_EBOPS. Deliberately NOT the hardware budget (an XCVU13P holds ~2.15e6 EBOPs): a setpoint that far
# away saturates the controller on its first epoch. It asks whether closed-loop pressure moves the cost.
TARGET_EBOPS = float(os.environ.get("HGQ_TARGET_EBOPS", "1.5e11"))
# Starting beta: keeps the old calibrated ratio of penalty gradient to task gradient on the quantizer
# (bitwidth) variables: 2.6e-10 x s_old / s_new = 2.6e-10 x 0.333 (per-batch 0.303-0.357), with
# s = |grad_q(EBOPs penalty per unit beta)| / |grad_q(task loss)|, each model as its own callbacks
# apply beta, on the same 3 x 2 CLIC dummy events (old model: commit 37ff127).
INIT_BETA = float(os.environ.get("HGQ_INIT_BETA", "8.7e-11"))
# Safety valve: the old 2.585 decades of headroom above INIT_BETA (1.0e-7 / 2.6e-10). An unresponsive
# process variable winds the integral up; saturating the ceiling is informative, not a failure.
MAX_BETA = float(os.environ.get("HGQ_MAX_BETA", "3.3e-8"))
# Floor: the same 2.585 decades below INIT_BETA. It must be > 0: the JAX driver reads EBOPs as
# penalty / beta (KERAS_JAX_PLAN.md section 3.6). The old default was 0.
MIN_BETA = float(os.environ.get("HGQ_MIN_BETA", "2.3e-13"))
# HGQ2's default integral gain of 2e-3 is tuned for hundred-epoch runs. Gain for roughly one decade per
# ten epochs at the initial error log10(MODEL_EBOPS / TARGET_EBOPS) = log10(2.511e11 / 1.5e11) = 0.224:
# 1 / (10 x 0.224) = 0.45, rounded to 0.5 (the old model's error was 0.224 too, by the same 40% rule).
PID_I = float(os.environ.get("HGQ_PID_I", "0.5"))
PID_WARMUP_EPOCHS = int(os.environ.get("HGQ_PID_WARMUP_EPOCHS", "2"))
EPOCHS = int(os.environ.get("HGQ_EPOCHS", "20"))

# Evaluate the EBOPs resource penalty every N steps rather than every step. Measured
# on an A100: EBOPs bookkeeping is 16.1% of step time, and EBOPs itself drifts 0.03%
# over 14k steps — so sampling it costs essentially nothing. Set to 1 to disable.
EBOPS_EVERY = int(os.environ.get("HGQ_EBOPS_EVERY", "8"))
# Skipped steps contribute no penalty, so the time-averaged pressure would drop by
# 1/N. Scale beta_end by N so the user-facing beta_end keeps its meaning.
BETA_END_EFFECTIVE = BETA_END * EBOPS_EVERY
# Same 1/N argument applies to the controller's beta: it would find the right value on
# its own, but starting and bounding it in the same units keeps every beta in this file
# comparable to the BetaScheduler runs.
INIT_BETA_EFFECTIVE = INIT_BETA * EBOPS_EVERY
MAX_BETA_EFFECTIVE = MAX_BETA * EBOPS_EVERY
MIN_BETA_EFFECTIVE = MIN_BETA * EBOPS_EVERY
# schedule mode: the ramp starts above 0, so that every step has an EBOPs read-out (penalty / beta)
BETA_START_EFFECTIVE = float(os.environ.get("HGQ_BETA_START", str(1e-3 * BETA_END))) * EBOPS_EVERY

# Beta ramp length. This was hardcoded at 42000 optimizer steps, which at 7769
# steps/epoch meant beta only reached full strength at epoch 5.4 — a diagnostic sweep
# had to run for days before it could show anything. Scale it to epochs instead.
# TRAIN_EVENTS is measured: 31075 micro-batches x 32 at batch 32, i.e. after the
# loader drops the ~1% of events exceeding the node/particle caps.
TRAIN_EVENTS = 994_400
STEPS_PER_EPOCH = math.ceil(TRAIN_EVENTS / (BATCH * ACCUM))
WARMUP_EPOCHS = float(os.environ.get("HGQ_WARMUP_EPOCHS", "1.0"))
WARMUP_STEPS = max(1, round(WARMUP_EPOCHS * STEPS_PER_EPOCH))

# No rescaling for ACCUM: Lightning divides each micro-batch loss by accumulate_grad_batches
# (lightning/pytorch/loops/optimization/automatic.py), so the accumulated gradient is the
# mean over BATCH x ACCUM events and clipping applies to that mean. An earlier version
# divided LR and multiplied the clip by ACCUM, which ran every job up to 2026-10-14 at
# LR 1.25e-5 / clip 0.4. tests/polaris/test_grad_accumulation.py pins this.
# Reference point: 5e-5 at effective batch 128 (matches the group's float baseline).
LR_MAX = 5.0e-5
CLIP = 0.1


def merge(base: dict, overlay: dict) -> dict:
    """Overlay ``overlay`` on ``base``: dicts merge key by key, every other value is replaced.

    The org overlay repeats every base key of the dicts it touches (decoder_layer_config), so this gives the same
    effective config as the LightningCLI's replace-a-plain-dict rule (docs/hgq/RETARGET_ORG_MAIN.md section 1.1).
    """
    out = dict(base)
    for key, value in overlay.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


cfg = yaml.safe_load(SRC.read_text())
if ATTN == "linformer":
    cfg = merge(cfg, yaml.safe_load(OVERLAY.read_text()))

cfg["name"] = "CLIC_pflow_HGQ_polaris" if ATTN == "linformer" else "CLIC_pflow_HGQ_dense_polaris"

d = cfg["data"]
d["train_path"] = f"{DATA_DIR}/train_clic_fix.root"
d["valid_path"] = f"{DATA_DIR}/val_clic_fix.root"
# NOTE: test_clic_common_infer.root contains -9999 sentinel indices that crash the
# loader; point test at the val file until that is patched.
d["test_path"] = f"{DATA_DIR}/val_clic_fix.root"
d["scale_dict_path"] = SCALE
d["batch_size"] = BATCH
d["num_workers"] = 8
d["num_val"] = 5000

t = cfg["trainer"]
# DDP is now validated on this stack (run 7598515, after the keras-device fix in
# lightning_module_hgq.setup): two-rank fit completes, checkpoint has 7095 tensors /
# 6176 quantizer vars / 0 non-finite, and DDP reaches val/loss 40.29 against the
# single-GPU 39.61 -- a 1.72% difference on 10 batches with different sharding.
#
# LR is deliberately NOT rescaled for devices. Lightning's DDP AVERAGES gradients across
# ranks, so the gradient magnitude at each optimizer step is unchanged; what grows is the
# number of samples behind it (devices x BATCH x ACCUM), i.e. less gradient noise. That
# permits a larger LR but does not require one, so raising it is left as a tuning
# decision rather than smuggled in with a throughput change. ACCUM behaves the same way:
# Lightning averages over accumulated micro-batches too.
DEVICES = int(os.environ.get("HGQ_DEVICES", "4"))
t["devices"] = DEVICES
# Pin the strategy. Lightning auto-selects ddp for devices>1, but the validation ran with
# it explicit, and an implicit choice is one more thing that can change under us.
if DEVICES > 1:
    t["strategy"] = "ddp"
t["accelerator"] = "gpu"
t["max_epochs"] = EPOCHS
t["precision"] = PRECISION
t["accumulate_grad_batches"] = ACCUM
t["gradient_clip_val"] = CLIP
t["gradient_clip_algorithm"] = "norm"
t["log_every_n_steps"] = 200

# Drop the Compile callback: torch.compile is measured NET-SLOWER (0.77x) on this
# stack because the Keras-torch dispatch layer breaks the graph at every layer.
cbs = [c for c in t["callbacks"] if "Compile" not in c.get("class_path", "")]
# The repo's Checkpoint callback monitors val/loss, which never fires mid-epoch. On a
# walltime-limited cluster that loses everything since the last epoch boundary, so add
# a plain step-interval ModelCheckpoint with save_last for resumable chaining.
cbs.append({
    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
    "init_args": {
        "dirpath": CKPT_DIR,
        "every_n_train_steps": 1000,
        "save_last": True,
        "save_top_k": 1,
        "monitor": None,
    },
})
t["callbacks"] = cbs

# Comet is the repo default; compute nodes have no outbound network, so force it offline.
# NOTE: trainer.logger must stay a SINGLE entry. The repo's CLI hardcodes
# `trainer.logger.init_args.offline_directory` and links `name` into
# `trainer.logger.init_args.name`; a list of loggers makes jsonargparse replace the list
# with a bare Namespace and the run dies at instantiation. Readable on-disk metrics come
# from the MetricsCsvWriter callback below instead.
if isinstance(t.get("logger"), dict):
    t["logger"].setdefault("init_args", {})["online"] = False

m = cfg["model"]
m["optimizer"] = "AdamW"
m["lrs_config"]["max"] = LR_MAX

net = m["model"]["init_args"]
net_cls = m["model"]
net_cls["class_path"] = "hepattn.keras.maskformer.KerasMaskFormer"

# KerasMaskFormer takes the encoder as a plain dict (it builds the layers inside the HGQ2 config
# scopes) and passes the model dim itself.
encoder = dict(net["encoder"]["init_args"])
if encoder.pop("dim") != net["dim"]:
    raise SystemExit("encoder dim differs from the model dim")
if ATTN == "torch":
    # base.yaml asks for flash-varlen, which KerasEncoder coerces to the same dense maths; say so explicitly
    encoder["attn_type"] = "torch"
net["encoder"] = encoder

# lap1015_late (base.yaml) is not importable here: 01_setup_env.sh installs with --no-install-project, so
# `import lap1015` is an empty namespace package and the first match fails. scipy finds the same optimum
# (ties may break differently) and is what every Polaris run before the org/main merge used.
net["matcher"]["init_args"]["default_solver"] = "scipy"

# HGQ2 quantization spec. q_type/place are SELECTORS in HGQ2 scopes; the type is
# chosen via default_q_type.
net["quant"] = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 0.0},
}

# BetaPID and BetaScheduler both write every quantized layer's _beta, and BetaScheduler
# writes per batch -- i.e. after BetaPID's epoch-start write -- so exactly one of them
# may be registered. BetaPID.setup() raises if both are; drop the other here so the
# config is never generated in that state.
t["callbacks"] = [c for c in t["callbacks"] if not c.get("class_path", "").endswith(("BetaScheduler", "BetaPID"))]
if BETA_MODE == "pid":
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.BetaPID",
        "init_args": {
            "target_ebops": TARGET_EBOPS,
            "init_beta": INIT_BETA_EFFECTIVE,
            "i": PID_I,
            "warmup_epochs": PID_WARMUP_EPOCHS,
            "max_beta": MAX_BETA_EFFECTIVE,
            "min_beta": MIN_BETA_EFFECTIVE,
        },
    })
else:
    # Open-loop ramp: kept for reproducing the earlier sweeps.
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.BetaScheduler",
        "init_args": {"beta_start": BETA_START_EFFECTIVE, "beta_end": BETA_END_EFFECTIVE, "warmup_steps": WARMUP_STEPS},
    })
if not any("EBOPsMonitor" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({"class_path": "hepattn.keras.callbacks.EBOPsMonitor"})
# The learned bitwidths are the only direct readout of whether QAT is doing anything.
# Without this, an inert beta sweep looks healthy on every other metric.
if not any("BitwidthMonitor" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({"class_path": "hepattn.keras.callbacks.BitwidthMonitor"})
# Must come after BitwidthMonitor: Lightning runs callbacks in order, and this writes
# whatever is in callback_metrics at that moment.
if not any("MetricsCsvWriter" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.MetricsCsvWriter",
        "init_args": {"path": f"{CKPT_DIR}/metrics.csv"},
    })
if EBOPS_EVERY > 1 and not any("PeriodicEBOPs" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.PeriodicEBOPs",
        "init_args": {"every_n_steps": EBOPS_EVERY},
    })

for task in net["tasks"]["init_args"]["modules"]:
    ia = task.get("init_args", {})
    if "scale_dict_path" in ia:
        ia["scale_dict_path"] = SCALE

DST.write_text(yaml.safe_dump(cfg, sort_keys=False))

print(f"WROTE {DST}")
print(f"  model      {'base.yaml + linformer.yaml (masked linformer, phi sort)' if ATTN == 'linformer' else 'base.yaml (dense attention)'}")
print(f"  data       {DATA_DIR}")
print(f"  ckpts      {CKPT_DIR}")
print(f"  batch      {BATCH} x accum {ACCUM} x devices {DEVICES} = effective {BATCH * ACCUM * DEVICES}")
print(f"  precision  {PRECISION}")
print(f"  lr max     {LR_MAX:.3e}   clip {CLIP}   (Lightning averages over accum: no rescaling)")
print(f"  epochs     {EPOCHS}")
if BETA_MODE == "pid":
    print(f"  beta       PID -> target {TARGET_EBOPS:.3e} EBOPs (model at init: {MODEL_EBOPS:.4e}; PROVISIONAL, BETA_SETPOINTS.md)")
    if ATTN != "linformer":
        print("             WARNING: the setpoints are derived for the linformer model; re-derive them for dense attention")
    print(
        f"             init {INIT_BETA_EFFECTIVE:.2e}  ceiling {MAX_BETA_EFFECTIVE:.2e}  floor {MIN_BETA_EFFECTIVE:.2e}"
        f"  i={PID_I}  warmup {PID_WARMUP_EPOCHS} epoch(s)"
    )
    headroom = math.log10(MAX_BETA_EFFECTIVE / INIT_BETA_EFFECTIVE)
    err = math.log10(MODEL_EBOPS / TARGET_EBOPS)
    print(
        f"             {headroom:.2f} decades of headroom; at a constant error of {err:.3f}"
        f" the ceiling is reached in ~{headroom / (PID_I * err):.0f} epochs"
    )
else:
    print(f"  beta_end   {BETA_END:.2e}  (open-loop schedule)")
    print(f"  beta ramp  {WARMUP_STEPS} steps = {WARMUP_EPOCHS:g} epoch(s) at {STEPS_PER_EPOCH} steps/epoch")
if EBOPS_EVERY > 1:
    scaled = f"beta_end scaled to {BETA_END_EFFECTIVE:.2e}" if BETA_MODE != "pid" else "pid betas scaled by 8"
    print(f"  EBOPs      every {EBOPS_EVERY} steps ({scaled} so time-averaged pressure is unchanged); ~16% faster steps")
