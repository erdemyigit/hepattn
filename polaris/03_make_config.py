#!/usr/bin/env python
"""Generate the Polaris HGQ2 training config from the repo's base.yaml.

Run from the repo root, inside the venv:
    python 03_make_config.py

Writes ./polaris_hgq.yaml. Everything Polaris-specific (data paths, checkpoint dir,
batch size, precision) is set here so the repo's committed configs stay clean.

Env overrides:
    HGQ_BATCH      micro-batch per GPU            (default 32)
    HGQ_ACCUM      gradient accumulation steps    (default 4  -> effective 128)
    HGQ_PRECISION  bf16-mixed | 32                (default bf16-mixed)
    HGQ_BETA_END   final EBOPs penalty weight     (default 4.0e-15)
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
SRC = REPO / "src/hepattn/experiments/clic/configs/base.yaml"
DST = REPO / "polaris_hgq.yaml"
if not SRC.exists():
    raise SystemExit(f"cannot find {SRC} — run this from the repo (it lives in polaris/)")

DATA_DIR = os.environ.get("DATA_DIR", os.environ.get("HGQ_DATA", str(REPO / "data/clic")))
CKPT_DIR = os.environ.get("CKPT_DIR", str(REPO / "hgq_ckpts"))
SCALE = str(REPO / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml")

BATCH = int(os.environ.get("HGQ_BATCH", "32"))
ACCUM = int(os.environ.get("HGQ_ACCUM", "4"))
PRECISION = os.environ.get("HGQ_PRECISION", "bf16-mixed")
BETA_END = float(os.environ.get("HGQ_BETA_END", "4.0e-15"))
EPOCHS = int(os.environ.get("HGQ_EPOCHS", "20"))

# Evaluate the EBOPs resource penalty every N steps rather than every step. Measured
# on an A100: EBOPs bookkeeping is 16.1% of step time, and EBOPs itself drifts 0.03%
# over 14k steps — so sampling it costs essentially nothing. Set to 1 to disable.
EBOPS_EVERY = int(os.environ.get("HGQ_EBOPS_EVERY", "8"))
# Skipped steps contribute no penalty, so the time-averaged pressure would drop by
# 1/N. Scale beta_end by N so the user-facing beta_end keeps its meaning.
BETA_END_EFFECTIVE = BETA_END * EBOPS_EVERY

# Beta ramp length. This was hardcoded at 42000 optimizer steps, which at 7769
# steps/epoch meant beta only reached full strength at epoch 5.4 — a diagnostic sweep
# had to run for days before it could show anything. Scale it to epochs instead.
# TRAIN_EVENTS is measured: 31075 micro-batches x 32 at batch 32, i.e. after the
# loader drops the ~1% of events exceeding the node/particle caps.
TRAIN_EVENTS = 994_400
STEPS_PER_EPOCH = math.ceil(TRAIN_EVENTS / (BATCH * ACCUM))
WARMUP_EPOCHS = float(os.environ.get("HGQ_WARMUP_EPOCHS", "1.0"))
WARMUP_STEPS = max(1, round(WARMUP_EPOCHS * STEPS_PER_EPOCH))

# Losses are mean-reduced, so Lightning SUMS the accumulated micro-batch gradients.
# Keep each optimizer step invariant to ACCUM: divide LR, multiply the clip threshold.
# Reference point: 5e-5 at effective batch 128 (matches the group's float baseline).
LR_MAX = 5.0e-5 / ACCUM
CLIP = 0.1 * ACCUM

with open(SRC) as f:
    cfg = yaml.safe_load(f)

cfg["name"] = "CLIC_pflow_HGQ_polaris"

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
t["devices"] = 1  # DDP over the Keras-torch/HGQ2 stack is NOT validated — see README
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

# Quantized encoder/decoder with the linformer attention used by the group's baseline.
net["encoder"] = {
    "num_layers": 6,
    "attn_type": "linformer",
    "hybrid_norm": True,
    "value_residual": True,
    "num_register_tokens": 8,
    "attn_kwargs": {"num_heads": 16, "linformer_seq_len": 256, "linformer_proj_dim": 256},
}
net["decoder"]["decoder_layer_config"]["attn_kwargs"].update({
    "attn_type": "linformer",
    "linformer_seq_len": 256,
    "linformer_proj_dim": 256,
})

# HGQ2 quantization spec. q_type/place are SELECTORS in HGQ2 scopes; the type is
# chosen via default_q_type.
net["quant"] = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 0.0},
}

# Ramp the EBOPs penalty in only after the task loss settles.
for c in t["callbacks"]:
    if c.get("class_path", "").endswith("BetaScheduler"):
        c["init_args"].update({"beta_start": 0.0, "beta_end": BETA_END_EFFECTIVE, "warmup_steps": WARMUP_STEPS})
        break
else:
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.BetaScheduler",
        "init_args": {"beta_start": 0.0, "beta_end": BETA_END_EFFECTIVE, "warmup_steps": WARMUP_STEPS},
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

with open(DST, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"WROTE {DST}")
print(f"  data       {DATA_DIR}")
print(f"  ckpts      {CKPT_DIR}")
print(f"  batch      {BATCH} x accum {ACCUM} = effective {BATCH * ACCUM}")
print(f"  precision  {PRECISION}")
print(f"  lr max     {LR_MAX:.3e}   clip {CLIP}   (invariant to accum)")
print(f"  beta_end   {BETA_END:.2e}   epochs {EPOCHS}")
print(f"  beta ramp  {WARMUP_STEPS} steps = {WARMUP_EPOCHS:g} epoch(s) at {STEPS_PER_EPOCH} steps/epoch")
if EBOPS_EVERY > 1:
    print(
        f"  EBOPs      every {EBOPS_EVERY} steps (beta_end scaled to {BETA_END_EFFECTIVE:.2e}"
        f" so time-averaged pressure is unchanged); ~16% faster steps"
    )
