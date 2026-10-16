"""Effect of jax_default_matmul_precision on the quantized org test model forward (CPU): default vs highest vs bfloat16.

Run from the worktree root (PYTHONPATH=src:.). Prints max |diff| of every float output vs 'highest' and the mask flips.
"""
# ruff: noqa: B023
# Evidence probe for docs/hgq/port_log/S5.md (scratch style: plain open(), short names, Keras/HGQ2 internals).

import json

import jax
import numpy as np
from tests.keras_native.golden.cases import QUANT_SPECS
from tests.keras_native.golden.mf_cases import clic_dummy_batch
from tests.keras_native.mf_golden_utils import build_mf, flatten

inputs, _ = clic_dummy_batch(num_events=2)
res = {}
for spec in ("high", "low"):
    model = build_mf("org", QUANT_SPECS[spec], inputs)
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    outs = {}
    for prec in ("highest", "default", "bfloat16", "tensorfloat32"):

        def fwd(tv, ntv, x, prec=prec):
            with jax.default_matmul_precision(prec):
                return model.stateless_call(tv, ntv, x, training=False)[0]

        outs[prec] = flatten(jax.jit(fwd)(tv, ntv, inputs))
    ref = outs["highest"]
    for prec in ("default", "bfloat16", "tensorfloat32"):
        diffs, flips = [], 0
        for k, r in ref.items():
            c = np.asarray(outs[prec][k])
            r = np.asarray(r)
            if r.dtype == bool:
                flips += int((c != r).sum())
            else:
                keep = np.abs(r) < 1e30
                diffs.append(float(np.abs(c[keep].astype(np.float64) - r[keep]).max()) if keep.any() else 0.0)
        res[f"{spec}/{prec}"] = {"max_abs": max(diffs), "mask_flips": flips}
print(json.dumps(res))
