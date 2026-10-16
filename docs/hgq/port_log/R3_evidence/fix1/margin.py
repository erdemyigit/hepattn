# ruff: noqa: E501  (mutant source strings are exact one-line matches)
"""Measure plan 5.8's margin on the R3 whole-model fixture: valid mask logits vs logit(0.1)."""

import math

import numpy as np
import tests.keras_native.test_pr8_layers as t
from tests.keras_native.golden.mf_cases import clic_dummy_batch

inputs, _ = clic_dummy_batch()
out_t, out_n = t.run_pair(*t.org_dense_pair(inputs), inputs)
thr = math.log(0.1 / 0.9)
print("keys with attn_mask:", sorted(k for k in out_t if k.endswith("attn_mask")))
for k in sorted(out_t):
    if k.endswith("mask.pflow_node_logit"):
        v = out_t[k]
        keep = v > -1e30
        tol = t.E2E["atol"] + t.E2E["rtol"] * np.abs(v[keep]).max()
        d = np.abs(v[keep] - thr)
        err = np.abs(v[keep].astype(np.float64) - np.asarray(out_n[k])[keep]).max()
        print(
            f"{k}: n={keep.sum()} tol={tol:.2e} min|logit-logit(0.1)|={d.min():.3e} ratio={d.min() / tol:.2f} n_within_10tol={(d < 10 * tol).sum()} maxerr={err:.2e}"
        )
