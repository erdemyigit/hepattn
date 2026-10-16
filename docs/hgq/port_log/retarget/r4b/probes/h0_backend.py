"""Quantized cores at 4 call sites x specs, same weights/inputs from numpy seeds; saves outputs per backend."""
import os, sys
import numpy as np
backend = os.environ["KERAS_BACKEND"]
if backend == "torch":
    import hepattn.keras
    hepattn.keras.set_keras_default_device("cpu")
import keras
from hepattn.keras_native import masked_linformer as nml
from hepattn.keras_native.factory import LayerFactory
HEADS, HD, K = 16, 16, 64
SITES = {"q_ca": (150, 160, 160, "attend"), "kv_ca": (160, 150, 160, "attend"), "enc": (168, 168, 168, "project"),
         "enc_short": (40, 40, 168, "project"), "q_sa": (150, 150, 160, "project_unmasked")}
SPECS = {
 "high": {"weight": {"default_q_type": "kbi", "b0": 20, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 8, "f0": 16}, "table": {"default_q_type": "kif", "i0": 2, "f0": 16}},
 "prod": {"weight": {"default_q_type": "kbi", "b0": 8, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8}, "table": {"default_q_type": "kif", "i0": 2, "f0": 10}},
}
B = int(os.environ.get("B", 2))
res = {}
for spec_name, spec in SPECS.items():
    for site, (nq, n, seq, kind) in SITES.items():
        rng = np.random.default_rng(7)
        q = rng.standard_normal((B, HEADS, nq, HD)).astype(np.float32)
        k = rng.standard_normal((B, HEADS, n, HD)).astype(np.float32)
        v = rng.standard_normal((B, HEADS, n, HD)).astype(np.float32)
        kv_mask = np.ones((B, n), bool); kv_mask[0, -7:] = False
        if B > 1: kv_mask[1, -1:] = False
        pk, pv = (rng.uniform(-(K**-0.5), K**-0.5, (seq, K)).astype(np.float32) for _ in range(2))
        factory = LayerFactory(spec)
        with factory.scopes():
            if kind == "attend":
                core = nml.AttendCore(factory, HEADS, HD, nq, n, seq, K, name="c")
                mask = (rng.random((B, nq, n)) < 0.4) & kv_mask[:, None]
            else:
                core = nml.ProjectCore(factory, HEADS, HD, nq, n, seq, K, masked=kind == "project", name="c")
                mask = kv_mask if kind == "project" else None
        core.build()
        core.E.assign_full(pk); core.F.assign_full(pv)
        for training in (False, True):
            out = keras.ops.convert_to_numpy(core(q, k, v, mask, training=training))
            res[f"{spec_name}.{site}.{int(training)}"] = np.asarray(out)
        # float reference (same weights) for the quantization-effect control
        fcore = (nml.AttendCore(LayerFactory(), HEADS, HD, nq, n, seq, K, name="f") if kind == "attend"
                 else nml.ProjectCore(LayerFactory(), HEADS, HD, nq, n, seq, K, masked=kind == "project", name="f"))
        fcore.build(); fcore.E.assign_full(pk); fcore.F.assign_full(pv)
        res[f"float.{site}"] = np.asarray(keras.ops.convert_to_numpy(fcore(q, k, v, mask)))
np.savez(sys.argv[1], **res)
print("saved", len(res), "jax" in sys.modules)
