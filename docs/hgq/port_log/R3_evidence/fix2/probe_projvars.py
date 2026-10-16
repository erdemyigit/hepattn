# ruff: noqa: E741
"""List native Linformer projection variables (E/F, tails) of the org model on the merged tree; core kind per attention."""

import collections
import sys

sys.path.insert(0, sys.argv[1])
from tests.keras_native import test_pr8_layers as T
from tests.keras_native.golden.mf_cases import clic_dummy_batch

m = T.native_from_section(T.org_model_section())
inputs, _ = clic_dummy_batch()
m(inputs, training=False)
c = collections.Counter()
for v in m.variables:
    p = v.path
    if any(s in p for s in ("_E_proj/", "_F_proj/", "_E/tail", "_F/tail")) or p.endswith(("_E/tail", "_F/tail")):
        key = "/".join(p.split("/")[-2:])
        part = next((s for s in ("encoder", "q_ca", "q_sa", "kv_ca") if s in p), "?")
        c[part, key.split("_", 2)[-1] if False else p.split("/")[-2][-6:] + "/" + p.split("/")[-1], tuple(v.shape)] += 1
for k, n in sorted(c.items()):
    print(k, n)
for i, l in enumerate(m.decoder.layers[:1]):
    print("dec", i, {p: getattr(l, p).fn.core_kind for p in ("q_ca", "q_sa", "kv_ca")})
print("enc0", m.encoder.layers[0].attn.fn.core_kind if hasattr(m.encoder.layers[0], "attn") else [a for a in dir(m.encoder.layers[0]) if "attn" in a])
