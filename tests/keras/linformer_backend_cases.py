"""Gate H0 backend-consistency cases: the keras_native masked-Linformer cores, quantized, at the org call sites.

Backend neutral (numpy and ``hepattn.keras_native`` only): ``make_linformer_backend_golden.py`` runs these cases
under ``KERAS_BACKEND=jax`` and stores the outputs; ``test_linformer_parity.py`` runs them under the torch backend of
the hybrid and compares (docs/hgq/RETARGET_ORG_MAIN.md section 4, gate H0). Weights and inputs come from numpy seeds,
so both processes see the same values; ``inputs_digest`` pins that.

Shapes are the production call sites of ``configs/linformer.yaml`` (q_len, kv_len, linformer_seq_len) with k = 64
and head_dim 16; 2 heads instead of 16 keep the golden file small (the core treats heads as a batch axis).
"""

import hashlib

import keras
import numpy as np

from hepattn.keras_native import masked_linformer as nml
from hepattn.keras_native.factory import LayerFactory

B, HEADS, HEAD_DIM, K = 2, 2, 16, 64
SITES = {  # (q_len, kv_len, linformer_seq_len, core)
    "q_ca": (150, 160, 160, "attend"),
    "kv_ca": (160, 150, 160, "attend"),  # renorm reads the tail rows 150-159
    "enc": (168, 168, 168, "project"),
    "enc_short": (40, 40, 168, "project"),  # tail rows in the renorm
    "q_sa": (150, 150, 160, "project_unmasked"),
}
SPECS = {
    # tests/keras_native/golden/cases.py "high"
    "high": {
        "weight": {"default_q_type": "kbi", "b0": 20, "i0": 2},
        "datalane": {"default_q_type": "kif", "i0": 8, "f0": 16},
        "table": {"default_q_type": "kif", "i0": 2, "f0": 16},
    },
    # the production spec of polaris/03_make_config.py and pflow_hgq.yaml
    "prod": {
        "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
        "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
        "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    },
}
MODES = {"eval": False, "train": True}


def case_inputs(site: str) -> dict:
    """q, k, v (B, H, S, Dh), E/F (seq_len, k) at masked_linformer's init range, and the mask of the site's core."""
    q_len, kv_len, seq_len, core = SITES[site]
    rng = np.random.default_rng(7)
    out = {
        name: rng.standard_normal((B, HEADS, length, HEAD_DIM)).astype(np.float32) for name, length in (("q", q_len), ("k", kv_len), ("v", kv_len))
    }
    out["E"], out["F"] = (rng.uniform(-(K**-0.5), K**-0.5, (seq_len, K)).astype(np.float32) for _ in range(2))
    kv_mask = np.ones((B, kv_len), bool)
    kv_mask[0, -7:] = False
    kv_mask[1, -1:] = False
    if core == "attend":
        out["mask"] = (rng.random((B, q_len, kv_len)) < 0.4) & kv_mask[:, None]
        out["mask"][0, 3] = False  # an all-False query row
    elif core == "project":
        out["mask"] = kv_mask
    else:
        out["mask"] = None
    return out


def inputs_digest() -> str:
    h = hashlib.sha256()
    for site in SITES:
        for name, value in sorted(case_inputs(site).items()):
            if value is not None:
                h.update(name.encode())
                h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def build_core(site: str, spec: dict | None):
    """The site's core from ``spec`` (None: float), with E/F loaded from ``case_inputs``."""
    q_len, kv_len, seq_len, core = SITES[site]
    factory = LayerFactory(spec)
    with factory.scopes():
        if core == "attend":
            layer = nml.AttendCore(factory, HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, name=f"{site}_core")
        else:
            layer = nml.ProjectCore(factory, HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, masked=core == "project", name=f"{site}_core")
    layer.build()
    x = case_inputs(site)
    layer.E.assign_full(x["E"])
    layer.F.assign_full(x["F"])
    return layer


def run_case(site: str, spec: dict | None, training: bool) -> np.ndarray:
    x = case_inputs(site)
    layer = build_core(site, spec)
    return np.asarray(keras.ops.convert_to_numpy(layer(x["q"], x["k"], x["v"], x["mask"], training=training)))


def key(spec_name: str, site: str, mode: str) -> str:
    return f"{spec_name}.{site}.{mode}"
