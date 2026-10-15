"""Relative error of the quantized renorm factor (1-bit mask lane x energy table -> rsqrt LUT) vs masked_linformer._rescale,
at the deployment renorm precision, n = seq_len = 160, k = 64, by number of valid keys."""
import sys

sys.path.insert(0, sys.argv[1])
import masked_linformer as ml
import numpy as np
import torch

import qml_layers as ql
from hepattn.keras_native.factory import LayerFactory

N, K, Q = 160, 64, 64
rng = np.random.default_rng(3)
E = rng.uniform(-(K**-0.5), K**-0.5, (N, K)).astype("float32")
if len(sys.argv) > 2:  # E on the grid of the deployment weight quantizer (kbi b0=8 i0=2: step 2^-(b0-i0))
    step = 2.0 ** -int(sys.argv[2])
    E = (np.round(E / step) * step).astype("float32")
    print(f"E quantized to step {step}: {np.mean(E == 0):.1%} of entries are 0, min nonzero |E| {np.abs(E[E != 0]).min():.4f}")
fac = LayerFactory({"weight": {"default_q_type": "kbi", "b0": 8, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
                    "table": {"default_q_type": "kif", "i0": 2, "f0": 10}})
for prec in ({}, {"lut_in_f": 16, "lut_out_f": 12, "w_f": 16}):
    ql.RENORM_PREC.update({"lut_in_f": 12, "lut_out_i": 7, "lut_out_f": 8, "w_f": 12, **prec})
    with fac.scopes():
        leaves = ql.Leaves(fac)
        kept, lut = leaves.mask_einsum("bqn,znk->bqk", "kept"), leaves.lut(ql.rsqrt_floor, "lut")
    kept.build([(None, Q, N), (1, N, K)])
    lut.build((None, Q * K))
    W = E**2 / (E**2).sum(0)
    print("renorm precision", dict(ql.RENORM_PREC))
    for n_valid in (1, 2, 5, 20, 80, 160):
        mask = np.zeros((1, Q, N), bool)
        for q in range(Q):
            mask[0, q, rng.choice(N, n_valid, replace=False)] = True
        r = np.asarray(ql.rescale(kept, lut, mask, W, training=False))
        ref = ml._rescale(torch.tensor(E), torch.tensor(mask), 1e-8).numpy()
        rel = np.abs(r / ref - 1)
        nz = ref < 1e6  # columns whose valid E entries are not all exactly 0 (r irrelevant there: (S*M)E is 0)
        rel = rel[nz]
        print(f"  valid keys {n_valid:4d}: exact r in [{ref.min():6.2f}, {ref[nz].max():8.2f}]  rel err median {np.median(rel):.2e}  max {rel.max():.2e}")
