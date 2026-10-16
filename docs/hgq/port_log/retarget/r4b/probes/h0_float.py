"""H0 float: native cores under KERAS_BACKEND=torch vs masked_linformer, same fixtures as test_masked_linformer.py."""
import sys
import numpy as np, torch, torch.nn.functional as F
import hepattn.keras  # pins torch backend
hepattn.keras.set_keras_default_device("cpu")
import keras
import masked_linformer as ml
from hepattn.keras_native import masked_linformer as nml
from hepattn.keras_native.factory import LayerFactory
assert "jax" not in sys.modules
DIM, HEADS, HEAD_DIM, K, B = 256, 16, 16, 64, 2
SHAPES = {"q_ca": (150, 160, 160, "attend"), "kv_ca": (160, 150, 160, "attend"), "enc": (168, 168, 168, "project"),
          "enc_short": (40, 40, 168, "project"), "q_sa": (150, 150, 160, "project_unmasked")}
def t(x): return None if x is None else torch.from_numpy(np.asarray(x))
def proj(seq_len, seed):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-(K**-0.5), K**-0.5, (seq_len, K)).astype(np.float32) for _ in range(2))
for shape in ("q_ca", "kv_ca"):
    q_len, kv_len, seq_len, _ = SHAPES[shape]
    rng = np.random.default_rng(1)
    q = rng.standard_normal((B, HEADS, q_len, HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32)
    kv_mask = np.ones((B, kv_len), bool); kv_mask[0, -7:] = False; kv_mask[1, -1:] = False
    attn_mask = (rng.random((B, q_len, kv_len)) < 0.5) & kv_mask[:, None]; attn_mask[0, 1] = False
    k = np.where(kv_mask[:, None, :, None], k, np.nan).astype(np.float32)
    v = np.where(kv_mask[:, None, :, None], v, np.nan).astype(np.float32)
    pk, pv = proj(seq_len, 2)
    core = nml.AttendCore(LayerFactory(), HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, name="att"); core.build()
    core.E.assign_full(pk); core.F.assign_full(pv)
    tq, tk, tv, tpk, tpv = (torch.tensor(a, requires_grad=True) for a in (q, k, v, pk, pv))
    ref = ml.attend(tq, tk, tv, tpk, tpv, t(attn_mask))
    cot = rng.standard_normal(ref.shape).astype(np.float32)
    (ref * t(cot)).sum().backward()
    hq, hk, hv = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
    out = core(hq, hk, hv, t(attn_mask))
    (out * t(cot)).sum().backward()
    print(type(out), out.dtype)
    fwd = (out.detach() - ref.detach()).abs().max().item()
    def full(p):
        rows = [p.proj.kernel.value.grad]
        if p.tail is not None: rows.append(p.tail.value.grad)
        return torch.cat(rows, 0)
    pad = ~t(kv_mask)[:, None, :, None].expand_as(tk)
    g = {"q": (hq.grad, tq.grad), "k": (hk.grad[~pad], tk.grad[~pad]), "v": (hv.grad[~pad], tv.grad[~pad]), "E": (full(core.E), tpk.grad), "F": (full(core.F), tpv.grad)}
    print(shape, "fwd", fwd, "finite", torch.isfinite(out).all().item(), {n: float((a - b).abs().max()) for n, (a, b) in g.items()},
          "padgrad", float(hk.grad[pad].abs().max()), float(hv.grad[pad].abs().max()))
for shape in ("enc", "enc_short", "q_sa"):
    q_len, kv_len, seq_len, kind = SHAPES[shape]
    masked = kind == "project"
    rng = np.random.default_rng(3)
    q, k, v = (rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32) for _ in range(3))
    kv_mask = None
    if masked:
        kv_mask = np.ones((B, kv_len), bool); kv_mask[0, -9:] = False; kv_mask[1, -1:] = False
        k = np.where(kv_mask[:, None, :, None], k, np.nan).astype(np.float32)
        v = np.where(kv_mask[:, None, :, None], v, np.nan).astype(np.float32)
    pk, pv = proj(seq_len, 4)
    core = nml.ProjectCore(LayerFactory(), HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, masked=masked, name="prj"); core.build()
    core.E.assign_full(pk); core.F.assign_full(pv)
    with torch.no_grad():
        out = core(t(q), t(k), t(v), t(kv_mask))
        tm = t(kv_mask)
        ref = F.scaled_dot_product_attention(t(q), ml.project(t(k), t(pk), tm), ml.project(t(v), t(pv), tm))
    print(shape, "project", float((out - ref).abs().max()), torch.isfinite(out).all().item())
print("jax loaded:", "jax" in sys.modules)
