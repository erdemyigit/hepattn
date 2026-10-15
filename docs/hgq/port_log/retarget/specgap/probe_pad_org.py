"""org/main Attention(linformer): do valid-row outputs depend on padded tokens? (self-attn, kv_mask) and per-query mask exactness."""
import torch
from hepattn.models.attention import Attention
torch.manual_seed(0)
a = Attention(256, num_heads=16, attn_type="linformer", linformer_seq_len=168, linformer_k=64, qkv_norm=True, norm="FastLayerNorm").eval()
x = torch.randn(2, 168, 256); m = torch.arange(168)[None] < torch.tensor([[100], [150]])
x2 = x.clone(); x2[~m] = torch.randn(int((~m).sum()), 256) * 50
with torch.no_grad():
    d = (a(x, kv_mask=m) - a(x2, kv_mask=m))[m].abs().max()
print("org self-attn: max |dOut| at valid rows after perturbing padded rows =", float(d))
# masked cross-attention: a key outside query i's mask must not affect query i
c = Attention(256, num_heads=16, attn_type="linformer", linformer_seq_len=160, linformer_k=64, qkv_norm=True, norm="FastLayerNorm").eval()
q = torch.randn(1, 150, 256); kv = torch.randn(1, 160, 256)
am = torch.rand(1, 150, 160) < 0.1; am[0, 0] = False; am[0, 0, :5] = True
kv2 = kv.clone(); kv2[0, 50] += 10.0
with torch.no_grad():
    o1 = c(q, k=kv, v=kv, attn_mask=am); o2 = c(q, k=kv2, v=kv2, attn_mask=am)
print("org masked CA: |dOut| query0 (key50 not in its mask) =", float((o1 - o2)[0, 0].abs().max()), "; queries that see key50 change:", bool((o1 - o2)[0, am[0, :, 50]].abs().max() > 0))
