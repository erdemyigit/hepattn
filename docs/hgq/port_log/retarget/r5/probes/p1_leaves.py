import os, sys
import keras, numpy as np
from hgq.layers import QDense, QEinsum, QSoftmax, QEinsumDense, QUnaryFunctionLUT
from hgq.quantizer import Quantizer
from hepattn.keras_native.masked_linformer import SeqProjection
B = keras.backend.backend()
spec = {"weight": {"default_q_type": "kbi", "b0": 20, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 8, "f0": 16}, "table": {"default_q_type": "kif", "i0": 2, "f0": 16}}
if B == "torch":
    import torch
    from hepattn.keras import set_keras_default_device; set_keras_default_device("cpu")
    from hepattn.keras.attention import KerasAttention as A
    from hepattn.keras.factory import LayerFactory
    f = LayerFactory(spec)
    with f.scopes():
        a = A(32, num_heads=16, attn_type="linformer", linformer_seq_len=160, linformer_k=64, qkv_norm=True, norm="LayerNorm", factory=f, name="attn")
    q = torch.randn(2, 160, 32); k = torch.randn(2, 150, 32); m = torch.rand(2, 160, 150) > 0.3
    a.eval(); a(q, k=k, attn_mask=m)
    mods = [m for m in a.modules() if isinstance(m, keras.layers.Layer)]
else:
    from hepattn.keras_native.attention import Attention as A
    from hepattn.keras_native.factory import LayerFactory
    f = LayerFactory(spec)
    with f.scopes():
        a = A(32, num_heads=16, attn_type="linformer", linformer_seq_len=160, linformer_k=64, qkv_norm=True, norm="LayerNorm", factory=f, name="attn")
        q = np.random.randn(2, 160, 32).astype("f4"); k = np.random.randn(2, 150, 32).astype("f4"); m = np.random.rand(2, 160, 150) > 0.3
        a(q, k=k, attn_mask=m)
    mods = list(a._flatten_layers(include_self=False, recursive=True))
for l in mods:
    own = [v.name for v in l._trainable_variables + l._non_trainable_variables] if hasattr(l, "_trainable_variables") else []
    print(type(l).__name__, l.name, len(l.variables), own)
