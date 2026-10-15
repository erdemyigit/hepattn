"""Float parity: qml_float (keras.ops, JAX) vs masked_linformer (torch) on padded inputs with all-False rows.

Usage: KERAS_BACKEND=jax [JAX_ENABLE_X64=1] python check_float.py <masked_linformer_dir> [float32|float64]
Padded k/v rows hold NaN, so a mask applied by multiply instead of where would show up as NaN.
"""

import sys

sys.path.insert(0, sys.argv[1])
DTYPE = sys.argv[2] if len(sys.argv) > 2 else "float32"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import masked_linformer as ml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import qml_float as qf  # noqa: E402

# mutants: deliberate bugs the checker must catch (non-vacuity evidence)
import os  # noqa: E402

MUTANT = os.environ.get("MUTANT", "")
if MUTANT == "mul_mask":  # mask by multiply instead of where: NaN padding leaks

    _real_ops = qf.ops

    class _Ops:
        def __getattr__(self, a):
            return getattr(_real_ops, a)

        @staticmethod
        def where(c, x, y):
            c = _real_ops.cast(c, "float32" if DTYPE == "float32" else "float64")
            return c * x + (1 - c) * y

    qf.ops = _Ops()
elif MUTANT == "full_sliced":  # numerator norm over E[:n] instead of all seq_len rows
    _orig = qf.rescale

    def _bad(proj, mask, eps=1e-8):
        return _orig(proj[: mask.shape[-1]], mask, eps)

    qf.rescale = _bad
elif MUTANT == "no_vmask":  # drop the final where(m, probs, 0) in attend
    _att = qf.attend

    def _bad_attend(q, k, v, pk, pv, am, renorm=True, eps=1e-8, rescale_fn=qf.rescale):
        return _att(q, k, v, pk, pv, am | ~am.any(-1, keepdims=True) * 0, renorm, eps, rescale_fn) if False else _no_vmask(q, k, v, pk, pv, am, renorm, eps, rescale_fn)

    def _no_vmask(q, k, v, pk, pv, am, renorm, eps, rescale_fn):
        o = qf.ops
        n = k.shape[-2]
        m = am[:, None]
        seen = o.expand_dims(o.any(m, axis=-2), -1)
        k, v = o.where(seen, k, 0.0), o.where(seen, v, 0.0)
        s = o.where(m, o.einsum("bhqd,bhnd->bhqn", q, k), 0.0) * q.shape[-1] ** -0.5
        s = o.einsum("bhqn,nk->bhqk", s, pk[:n]) * rescale_fn(pk, am, eps)[:, None]
        a = o.softmax(s, axis=-1) * rescale_fn(pv, am, eps)[:, None]
        return o.einsum("bhqn,bhnd->bhqd", o.einsum("bhqk,nk->bhqn", a, pv[:n]), v)

    qf.attend = _bad_attend

TDT = {"float32": torch.float32, "float64": torch.float64}[DTYPE]
rng = np.random.default_rng(0)
B, H, NQ, N, D, SEQ, K = 3, 4, 7, 11, 8, 13, 5

q = rng.standard_normal((B, H, NQ, D)).astype(DTYPE)
k = rng.standard_normal((B, H, N, D)).astype(DTYPE)
v = rng.standard_normal((B, H, N, D)).astype(DTYPE)
E = rng.uniform(-K**-0.5, K**-0.5, (SEQ, K)).astype(DTYPE)
Fp = rng.uniform(-K**-0.5, K**-0.5, (SEQ, K)).astype(DTYPE)
kv_mask = np.ones((B, N), bool)
kv_mask[0, -3:] = False  # padding last, as linformer.yaml
kv_mask[2, -1:] = False
kv_mask[1, :] = True
for b in range(B):  # NaN in padded rows: only where-masking keeps them out
    k[b, :, ~kv_mask[b]] = np.nan
    v[b, :, ~kv_mask[b]] = np.nan
attn_mask = rng.random((B, NQ, N)) < 0.6
attn_mask[1, 0] = False  # all-False query row
attn_mask[0, 2] = False
attn_mask[2, :, 4] = False  # a key no query sees
eff_mask = attn_mask & kv_mask[:, None]
assert (~eff_mask.any(-1)).sum() >= 2

w_att = rng.standard_normal((B, H, NQ, D)).astype(DTYPE)
w_prj = rng.standard_normal((B, H, K, D)).astype(DTYPE)


def tt(x, grad=False):
    t = torch.tensor(x, dtype=TDT)
    return t.requires_grad_(grad)


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all(), "non-finite values"
    return float(np.max(np.abs(a - b)))


def report(name, val, tol):
    status = "OK" if val <= tol else "FAIL"
    print(f"{name:58s} max|diff| = {val:.3e}  [{status}, tol {tol:.0e}]")
    return val <= tol


tol = 1e-5 if DTYPE == "float32" else 1e-12
ok = True

# ---- rescale: copy and LUT form vs torch -------------------------------------------------------
r_t = ml._rescale(tt(E), torch.tensor(eff_mask), 1e-8).detach().numpy()
ok &= report("rescale (op copy) vs _rescale, incl. all-False rows (rel)", float(np.max(np.abs(np.asarray(qf.rescale(E, eff_mask)) / r_t - 1))), tol)
lut_rel = np.max(np.abs(np.asarray(qf.rescale_lut_form(E, eff_mask)) / r_t - 1))
ok &= report("rescale LUT form rsqrt(M @ W) vs _rescale (relative)", lut_rel, tol)

# ---- attend: forward + grads ---------------------------------------------------------------------
tq, tk, tv, tE, tF = tt(q, True), tt(k, True), tt(v, True), tt(E, True), tt(Fp, True)
out_t = ml.attend(tq, tk, tv, tE, tF, torch.tensor(eff_mask))
(out_t * tt(w_att)).sum().backward()
g_t = [x.grad.numpy() for x in (tq, tk, tv, tE, tF)]

for label, rfn in (("op copy", qf.rescale), ("LUT form", qf.rescale_lut_form)):

    def f_att(q_, k_, v_, e_, f_, rfn=rfn):
        return jnp.sum(qf.attend(q_, k_, v_, e_, f_, eff_mask, rescale_fn=rfn) * w_att)

    out_j = qf.attend(q, k, v, E, Fp, eff_mask, rescale_fn=rfn)
    ok &= report(f"attend fwd ({label})", maxdiff(out_j, out_t.detach().numpy()), tol)
    zero_rows = ~eff_mask.any(-1)
    ok &= report(f"attend fwd ({label}): all-False rows exactly 0", float(np.abs(np.asarray(out_j).transpose(0, 2, 1, 3)[zero_rows]).max()), 0.0)
    g_j = jax.jit(jax.grad(f_att, argnums=(0, 1, 2, 3, 4)))(q, k, v, E, Fp)
    for nm, gj, gt in zip("q k v E F".split(), g_j, g_t, strict=True):
        ok &= report(f"attend grad d/d{nm} ({label})", maxdiff(gj, gt), tol * 10)
    gk = np.asarray(g_j[1])
    pad_grad = max(np.abs(gk[b, :, ~kv_mask[b]]).max(initial=0.0) for b in range(B))
    ok &= report(f"attend grad into padded keys exactly 0 ({label})", pad_grad, 0.0)

# ---- project: mask None, mask with/without renorm, NaN in padded rows ------------------------------
for renorm in (True, False):
    tk2, tE2 = tt(k, True), tt(E, True)
    pt = ml.project(tk2, tE2, torch.tensor(kv_mask), renorm=renorm)
    (pt * tt(w_prj)).sum().backward()
    pj = qf.project(k, E, kv_mask, renorm=renorm)
    ok &= report(f"project fwd (kv_mask, renorm={renorm})", maxdiff(pj, pt.detach().numpy()), tol)
    gk, gE = jax.grad(lambda k_, e_, r=renorm: jnp.sum(qf.project(k_, e_, kv_mask, renorm=r) * w_prj), argnums=(0, 1))(k, E)
    ok &= report(f"project grad d/dk (renorm={renorm})", maxdiff(gk, tk2.grad.numpy()), tol * 10)
    ok &= report(f"project grad d/dE (renorm={renorm})", maxdiff(gE, tE2.grad.numpy()), tol * 10)

k_clean = np.nan_to_num(k)
ok &= report("project fwd (mask None: E sliced, no rescale)", maxdiff(qf.project(k_clean, E, None), ml.project(tt(k_clean), tt(E), None).numpy()), tol)

# ---- org/main project path: project k/v with kv_mask, then unmasked SDPA ----------------------------
kp_t, vp_t = ml.project(tt(k), tt(E), torch.tensor(kv_mask)), ml.project(tt(v), tt(Fp), torch.tensor(kv_mask))
o_t = F.scaled_dot_product_attention(tt(q), kp_t, vp_t).numpy()
o_j = qf.sdpa(q, qf.project(k, E, kv_mask), qf.project(v, Fp, kv_mask))
ok &= report("project path: project(k,v,kv_mask) + SDPA", maxdiff(o_j, o_t), tol)

print("ALL OK" if ok else "SOME FAILED", f"({DTYPE})")
