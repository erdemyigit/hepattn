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
    HGQ_EPOCHS     max epochs                     (default 50)
    DATA_DIR       CLIC ROOT directory            (default $HGQ_DATA or ./data/clic)
    CKPT_DIR       checkpoint output directory    (default ./hgq_ckpts)
"""

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
EPOCHS = int(os.environ.get("HGQ_EPOCHS", "50"))

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
if isinstance(t.get("logger"), dict):
    t["logger"].setdefault("init_args", {})["online"] = False  # no outbound net on compute nodes

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
        c["init_args"].update({"beta_start": 0.0, "beta_end": BETA_END, "warmup_steps": 42000})
        break
else:
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.BetaScheduler",
        "init_args": {"beta_start": 0.0, "beta_end": BETA_END, "warmup_steps": 42000},
    })
if not any("EBOPsMonitor" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({"class_path": "hepattn.keras.callbacks.EBOPsMonitor"})

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
