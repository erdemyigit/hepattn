# ruff: noqa: ARG005
"""Linformer PR #8 decoder layer (depth 0): native (ported) vs org torch, q and valid kv max |diff|; q_sa core kind."""

import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from tests.keras_native import test_pr8_layers as T
from tests.keras_native.porting import perturb_affine_params

from hepattn.models.decoder import MaskFormerDecoderLayer

cfg = T.decoder_cfg(0)
cfg["attn_kwargs"] = {**cfg["attn_kwargs"], "attn_type": "linformer", "linformer_seq_len": T.DEC_N, "linformer_k": 8}
torch.manual_seed(20)
t = MaskFormerDecoderLayer(**cfg).eval()
perturb_affine_params(t, 20)
q, kv, am, km = T.decoder_inputs()
with torch.no_grad():
    qt, kt = t(*(torch.from_numpy(a) for a in (q, kv)), attn_mask=torch.from_numpy(am), kv_mask=torch.from_numpy(km))
orig = T.decoder_cfg
T.decoder_cfg = lambda depth, **k: cfg
n = T.native_decoder_layer(t, 0)
qn, kn = n(q, kv, attn_mask=am, kv_mask=km)
print(
    "q_sa core",
    n.q_sa.fn.core_kind,
    "| q max",
    float(np.abs(qt.numpy() - np.asarray(qn)).max()),
    "| kv valid max",
    float(np.abs((kt.numpy() - np.asarray(kn))[km]).max()),
)
