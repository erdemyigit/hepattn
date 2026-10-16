# ruff: noqa: SLF001
"""Measure jit vs eager max |diff| of the native org model (dense attention, dim 32) over 3 phi permutations."""

import sys

import jax
import numpy as np

sys.path.insert(0, sys.argv[1])
from tests.keras_native import test_pr8_layers as T
from tests.keras_native.golden.mf_cases import clic_dummy_batch
from tests.keras_native.test_maskformer import flatten_torch

batch, _ = clic_dummy_batch()
m = T.native_from_section(T.with_dense_attention(T.org_model_section(dim=32)))
m(batch, training=False)
tv = [v.value for v in m.trainable_variables]
ntv = [v.value for v in m.non_trainable_variables]


@jax.jit
def fwd(tv, ntv, inp):
    out, _ = m.stateless_call(tv, ntv, inp, training=False)
    return flatten_torch(out)


rng = np.random.default_rng(4)
worst = {}
for _ in range(3):
    inp = dict(batch)
    inp["node_phi"] = np.stack([rng.permutation(r) for r in batch["node_phi"]]).astype(np.float32)
    e = flatten_torch(m(inp, training=False))
    j = fwd(tv, ntv, inp)
    for k in e:
        a, b = np.asarray(e[k]), np.asarray(j[k])
        if a.dtype == bool:
            worst[k] = max(worst.get(k, 0), int((a != b).sum()))
            continue
        keep = np.abs(a) < 1e30
        worst[k] = max(worst.get(k, 0.0), float(np.abs(a[keep].astype(np.float64) - b[keep]).max()) if keep.any() else 0.0)
for k, v in sorted(worst.items()):
    print(f"{k:45s} {v}")
print("cache", fwd._cache_size())
