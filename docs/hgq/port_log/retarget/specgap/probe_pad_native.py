"""keras_native LinformerAttention: same two checks."""
import numpy as np, keras
from hepattn.keras_native.attention import build_attention
rng = np.random.default_rng(0)
a = build_attention(256, attn_type="linformer", num_heads=16, linformer_seq_len=168, linformer_proj_dim=64, name="a")
x = rng.normal(size=(2, 168, 256)).astype("float32"); m = np.arange(168)[None] < np.array([[100], [150]])
x2 = x.copy(); x2[~m] = rng.normal(size=(int((~m).sum()), 256)) * 50
d = np.abs(np.asarray(a(x, kv_mask=m)) - np.asarray(a(x2, kv_mask=m)))[m].max()
print("native self-attn: max |dOut| at valid rows after perturbing padded rows =", float(d))
c = build_attention(256, attn_type="linformer", num_heads=16, linformer_seq_len=160, linformer_proj_dim=64, name="c")
q = rng.normal(size=(1, 150, 256)).astype("float32"); kv = rng.normal(size=(1, 160, 256)).astype("float32")
am = rng.random((1, 150, 160)) < 0.1
try:
    c(q, k=kv, v=kv, attn_mask=am); print("native masked CA k=64: ran")
except Exception as e:
    print("native masked CA k=64:", type(e).__name__, e)
