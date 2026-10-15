"""Quantized masked-linformer prototype: float-mode parity, high-spec parity, prod-spec jit/grad/EBOPs.

Usage: KERAS_BACKEND=jax PYTHONPATH=<hepattn-keras-jax>/src:. python run_quant.py <masked_linformer_dir> [small|prod]
"""

import math
import sys

sys.path.insert(0, sys.argv[1])
SHAPE = sys.argv[2] if len(sys.argv) > 2 else "small"

import jax  # noqa: E402
import keras  # noqa: E402
import masked_linformer as ml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hgq.layers.core.base import QLayerBase  # noqa: E402

from hepattn.keras_native.attention import Attention  # noqa: E402
from hepattn.keras_native.factory import LayerFactory  # noqa: E402
from hepattn.keras_native.linformer import LinformerAttention  # noqa: E402
import qml_layers  # noqa: E402
from qml_layers import AttendCore, ExplicitCore, Leaves, LinAttention, ProjectCore  # noqa: E402

PROD_SPEC = {  # src/hepattn/experiments/clic/configs/pflow_hgq.yaml quant block, beta0 = 1 so losses == EBOPs
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0},
}
HIGH_SPEC = {  # tests/keras_native/golden/cases.py "high"
    "weight": {"default_q_type": "kbi", "b0": 20, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 8, "f0": 16},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 16},
    "ebops": {"beta0": 1.0},
}
if SHAPE == "small":
    B, DIM, H, NQ, N, SEQ, K = 2, 32, 2, 12, 16, 20, 8
else:  # decoder q_ca of linformer.yaml: dim 256, 16 heads, 150 queries over 160 constituents, seq_len 160, k 64
    B, DIM, H, NQ, N, SEQ, K = 1, 256, 16, 150, 160, 160, 64
D = DIM // H
rng = np.random.default_rng(1)


def masks(b, nq, n):
    kv_mask = np.ones((b, n), bool)
    kv_mask[0, -max(1, n // 5) :] = False  # padding last
    attn_mask = rng.random((b, nq, n)) < 0.5
    attn_mask[0, 1] = False  # all-False query row
    return kv_mask, attn_mask


def report(name, val, tol):
    print(f"  {name:62s} {val:.3e}  [{'OK' if val <= tol else 'FAIL'}, tol {tol:.0e}]")
    return val <= tol


# ======================= 1. core parity vs masked_linformer (float and high-spec leaves) =================
def core_parity(spec, tol):
    factory = LayerFactory(spec)
    q = rng.standard_normal((B, H, NQ, D)).astype("float32")
    k = rng.standard_normal((B, H, N, D)).astype("float32")
    v = rng.standard_normal((B, H, N, D)).astype("float32")
    kv_mask, attn_mask = masks(B, NQ, N)
    eff = attn_mask & kv_mask[:, None]
    if spec is None:  # NaN in padded rows (float only: a quantizer would map NaN to garbage, fixed point has no NaN)
        for b in range(B):
            k[b, :, ~kv_mask[b]] = np.nan
            v[b, :, ~kv_mask[b]] = np.nan
    pk = rng.uniform(-(K**-0.5), K**-0.5, (SEQ, K)).astype("float32")
    pv = rng.uniform(-(K**-0.5), K**-0.5, (SEQ, K)).astype("float32")
    T = torch.tensor
    ok = True
    with factory.scopes():
        att = AttendCore(Leaves(factory), H, D, NQ, N, SEQ, K)
        prj = ProjectCore(Leaves(factory), H, D, NQ, N, SEQ, K)
    att.build()
    prj.build()
    for core in (att, prj):
        core.E.assign_full(pk)
        core.F.assign_full(pv)
    out = np.asarray(att(q, k, v, eff, training=False))
    ref = ml.attend(T(q), T(k), T(v), T(pk), T(pv), T(eff)).numpy()
    ok &= report("attend core vs masked_linformer.attend: max|diff|", float(np.abs(out - ref).max()), tol)
    zero = ~eff.any(-1)
    ok &= report("attend core: all-False query rows |out| max", float(np.abs(out.transpose(0, 2, 1, 3)[zero]).max()), 0.0 if spec is None else tol)
    out = np.asarray(prj(q, k, v, kv_mask, training=False))
    ref = F.scaled_dot_product_attention(T(q), ml.project(T(k), T(pk), T(kv_mask)), ml.project(T(v), T(pv), T(kv_mask))).numpy()
    ok &= report("project core vs project(k,v,kv_mask)+SDPA: max|diff|", float(np.abs(out - ref).max()), tol)
    out = np.asarray(prj(q, np.nan_to_num(k), np.nan_to_num(v), None, training=False))
    ref = F.scaled_dot_product_attention(T(q), ml.project(T(np.nan_to_num(k)), T(pk)), ml.project(T(np.nan_to_num(v)), T(pv))).numpy()
    ok &= report("project core, mask None (E[:n], no renorm): max|diff|", float(np.abs(out - ref).max()), tol)
    if spec is None:
        with factory.scopes():
            ex = ExplicitCore(Leaves(factory), H, D, NQ, N, SEQ, K)
            a0 = AttendCore(Leaves(factory), H, D, NQ, N, SEQ, K, renorm=False)
        ex.build()
        a0.build()
        for core in (ex, a0):
            core.E.assign_full(pk)
            core.F.assign_full(pv)
        kc, vc = np.nan_to_num(k), np.nan_to_num(v)
        ref = ml.attend(T(q), T(kc), T(vc), T(pk), T(pv), T(eff), renorm=False).numpy()
        ok &= report("explicit per-query form vs attend(renorm=False): max|diff|", float(np.abs(np.asarray(ex(q, kc, vc, eff)) - ref).max()), tol)
        ok &= report("AttendCore(renorm=False) vs attend(renorm=False): max|diff|", float(np.abs(np.asarray(a0(q, kc, vc, eff)) - ref).max()), tol)
    return ok


def core_ebops():
    """EBOPs of the attention cores alone (q/k/v already projected), prod spec, renorm off for attend vs explicit."""
    factory = LayerFactory(PROD_SPEC)
    q = rng.standard_normal((B, H, NQ, D)).astype("float32")
    k = rng.standard_normal((B, H, N, D)).astype("float32")
    kv_mask, attn_mask = masks(B, NQ, N)
    eff = attn_mask & kv_mask[:, None]
    res = {}
    for name, cls, kw in (("attend(renorm=False)", AttendCore, {"renorm": False}), ("explicit per-query", ExplicitCore, {})):
        with factory.scopes():
            core = cls(Leaves(factory), H, D, NQ, N, SEQ, K, **kw)
        core.build()
        core(q, k, k, eff, training=True)
        rows = {n_: float(sum(float(x) for x in leaf.losses)) for n_, leaf in hgq_leaves(core).items()}
        res[name] = rows
        print(f"  {name:22s} total {sum(rows.values()):.4e}: " + ", ".join(f"{n_.split('_', 1)[1]}={v_:.3e}" for n_, v_ in rows.items()))
    return res


# ======================= 2. EBOPs per leaf, jit + grad ======================================================
def hgq_leaves(module):
    out = {}
    for layer in module._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, QLayerBase) and not layer.name.endswith(("_exp_table", "_inv_table")):
            out[layer.name] = layer
    return out


def leaf_ebops(module):
    """Float EBOPs per leaf from the train-mode losses (beta0 = 1; each HGQ2 leaf adds ebops * beta), and the uint32
    variable for comparison (it truncates and saturates, docs/hgq/KERAS_JAX_PLAN.md 3.6)."""
    rows = {}
    for name, leaf in hgq_leaves(module).items():
        rows[name] = (float(sum(float(x) for x in leaf.losses)), int(leaf.ebops))
    return rows


def build_module(kind, factory):
    with factory.scopes():
        if kind in ("attend", "project"):
            mod = LinAttention(factory, DIM, H, NQ, N, SEQ, K, core=kind, name=kind)
            mod.build()
        elif kind == "helen":  # keras_native/linformer.py today: shared seqproj, mask in rank-k space
            mod = LinformerAttention(DIM, num_heads=H, seq_len=SEQ, k=K, factory=factory, name="helen")
            mod.build((None, NQ, DIM), (None, N, DIM))
        else:  # quadratic masked attention (keras_native Attention, torch-SDPA semantics)
            mod = Attention(DIM, num_heads=H, factory=factory, name=kind)
            mod.build((None, NQ, DIM), (None, N, DIM))
    return mod


def call_module(kind, mod, tv, ntv, x, training):
    q, kv, kv_mask, attn_mask = x
    if kind in ("attend", "project"):
        return mod.stateless_call(tv, ntv, q, kv, kv_mask=kv_mask, attn_mask=attn_mask, return_losses=True, training=training)
    if kind == "helen":
        return mod.stateless_call(tv, ntv, q, k=kv, v=kv, return_losses=True, training=training)
    if kind == "dense_nomask":
        return mod.stateless_call(tv, ntv, q, k=kv, return_losses=True, training=training)
    return mod.stateless_call(tv, ntv, q, k=kv, kv_mask=kv_mask, attn_mask=attn_mask, return_losses=True, training=training)


def macs(kind):  # noqa: PLR0911
    """Multiplies per event by class (dxd = data x data, dxp = data x parameter), q/k/v/out projections excluded."""
    hq = H * NQ
    if kind == "attend":
        return {"dxd": 2 * hq * N * D + 2 * hq * K, "dxp": 2 * hq * N * K, "mask x table": 2 * NQ * N * K}
    if kind == "project":
        return {"dxd": 2 * hq * K * D + 2 * H * K * D, "dxp": 2 * H * N * K * D, "mask x table": 2 * N * K}
    if kind == "helen":
        return {"dxd": 2 * hq * K * D, "dxp": 2 * N * DIM * K}
    return {"dxd": 2 * hq * N * D}  # dense and dense_nomask


def ebops_and_jit(kind):
    factory = LayerFactory(PROD_SPEC)
    mod = build_module(kind, factory)
    kv_mask, attn_mask = masks(B, NQ, N)
    x = (
        rng.standard_normal((B, NQ, DIM)).astype("float32"),
        rng.standard_normal((B, N, DIM)).astype("float32"),
        kv_mask,
        attn_mask,
    )
    q, kv, km, am = x
    if kind in ("attend", "project"):
        mod(q, kv, kv_mask=km, attn_mask=am, training=True)
    elif kind == "helen":
        mod(q, k=kv, v=kv, training=True)
    elif kind == "dense_nomask":
        mod(q, k=kv, training=True)
    else:
        mod(q, k=kv, kv_mask=km, attn_mask=am, training=True)
    rows = leaf_ebops(mod)

    def loss_fn(tv, ntv, x):
        out, new_ntv, losses = call_module(kind, mod, tv, ntv, x, True)
        qloss = sum(losses)
        return keras.ops.mean(keras.ops.square(out)) + 1e-12 * qloss, (new_ntv, qloss)

    step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    tv = [v.value for v in mod.trainable_variables]
    ntv = [v.value for v in mod.non_trainable_variables]
    (loss, (ntv2, qloss)), grads = step(tv, ntv, x)
    (_, (_, qloss2)), _ = step(tv, ntv2, x)  # second call: same shapes, must not retrace
    zero = [v.path for v, g in zip(mod.trainable_variables, grads, strict=True) if not np.any(np.asarray(g) != 0)]
    finite = all(np.all(np.isfinite(np.asarray(g))) for g in grads)
    return rows, float(qloss), step._cache_size(), len(tv), zero, finite, float(loss)


def main():
    ok = True
    print(f"shape={SHAPE}: B={B} dim={DIM} heads={H} head_dim={D} nq={NQ} n={N} seq_len={SEQ} k={K}")
    print("[1] float leaves vs masked_linformer (torch), NaN in padded k/v rows, all-False query rows")
    ok &= core_parity(None, 2e-5 if SHAPE == "small" else 1e-4)
    print("[2] HGQ2 leaves at the 'high' spec vs masked_linformer (no NaN; quantization error only)")
    print("    [2a] renorm LUT/energy lanes at the deployment precision", qml_layers.RENORM_PREC)
    core_parity(HIGH_SPEC, 1e-2)
    saved = dict(qml_layers.RENORM_PREC)
    qml_layers.RENORM_PREC.update(lut_in_f=19, lut_out_f=16, w_f=20)
    print("    [2b] renorm lanes raised (parity only; a 2^20-entry table exceeds hls4ml's 1e6 cap)", qml_layers.RENORM_PREC)
    ok &= core_parity(HIGH_SPEC, 1e-3)
    qml_layers.RENORM_PREC.update(saved)
    print("[3] cores only (q/k/v given), prod spec: implemented attend vs explicit per-query projection form")
    core_ebops()
    print("[4] prod spec (pflow_hgq.yaml): eager train-mode EBOPs per leaf, jit value_and_grad")
    summary = {}
    for kind in ("attend", "project", "helen", "dense", "dense_nomask"):
        rows, qloss, cache, ntv, zero, finite, loss = ebops_and_jit(kind)
        total = sum(r[0] for r in rows.values())
        summary[kind] = (total, rows)
        print(f"  --- {kind}: total EBOPs (float, eager) = {total:.4e}; jit losses sum = {qloss:.4e}; "
              f"jit cache size after 2 calls = {cache}; {ntv} trainables, zero-grad: {len(zero)} {zero}; grads finite={finite}")
        for name, (f, u) in sorted(rows.items(), key=lambda r: -r[1][0]):
            print(f"      {name:34s} {f:14.4e}  {100 * f / total:6.2f}%  (uint32 var {u})")
        m = macs(kind)
        print(f"      MACs/event (core, excl. q/k/v/out proj): {m}")
        ok &= total > 0 and finite and cache == 1
    base = summary["dense_nomask"][0]
    print("  EBOPs relative to quadratic attention WITHOUT a softmax mask (same q/k/v/out projections):")
    for kind, (total, _) in summary.items():
        print(f"      {kind:8s} {total:12.4e}  x{total / base:.3f}")
    print("ALL OK" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
