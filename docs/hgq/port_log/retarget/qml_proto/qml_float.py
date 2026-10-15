"""Float keras.ops twin of masked_linformer.project / _rescale / attend (dce2e15).

Two rescale forms:
- ``rescale``: op-for-op copy of masked_linformer._rescale (full / sqrt(clamp(mask @ E^2, eps^2))).
- ``rescale_lut_form``: the form the quantized design uses, rsqrt(mask @ W) with the column-normalized
  energy W = E^2 / sum_rows(E^2); identical in exact arithmetic (full/kept = 1/sqrt(kept^2/full^2)).
"""

from keras import ops


def _promote(p):
    return p if ops.dtype(p) == "float64" else ops.cast(p, "float32")


def rescale(proj, mask, eps=1e-8):
    """(..., n) bool mask -> (..., k) column rescale ||E[:, j]|| / ||E[valid, j]||."""
    p = _promote(proj)
    p2 = ops.square(p)
    full = ops.sqrt(ops.sum(p2, axis=0))
    n = mask.shape[-1]
    kept = ops.sqrt(ops.maximum(ops.einsum("...n,nk->...k", ops.cast(mask, p.dtype), p2[:n]), eps**2))
    return full / kept


def energy_fraction(proj):
    """W[i, j] = E[i, j]^2 / sum_i' E[i', j]^2 over all seq_len rows (parameter-only)."""
    p2 = ops.square(_promote(proj))
    return p2 / ops.sum(p2, axis=0, keepdims=True)


def rescale_lut_form(proj, mask, eps=1e-8):
    """Same values as ``rescale``: rsqrt(max(mask @ W, eps^2 / full^2))."""
    p = _promote(proj)
    full2 = ops.sum(ops.square(p), axis=0)
    n = mask.shape[-1]
    kept2n = ops.einsum("...n,nk->...k", ops.cast(mask, p.dtype), energy_fraction(p)[:n])
    return ops.rsqrt(ops.maximum(kept2n, eps**2 / full2))


def project(t, proj, mask=None, renorm=True, eps=1e-8, rescale_fn=rescale):
    """``t`` (b, ..., n, d) -> (b, ..., k, d) along n; ``mask`` bool (b, n), True = real token."""
    n = t.shape[-2]
    if mask is None:
        return ops.einsum("...nd,nk->...kd", t, ops.cast(proj[:n], t.dtype))
    lead = (mask.shape[0],) + (1,) * (len(t.shape) - 3)
    out = ops.einsum("...nd,nk->...kd", ops.where(ops.reshape(mask, (*lead, n, 1)), t, 0.0), ops.cast(proj[:n], t.dtype))
    if renorm:
        out = out * ops.reshape(ops.cast(rescale_fn(proj, mask, eps), out.dtype), (*lead, -1, 1))
    return out


def attend(q, k, v, proj_k, proj_v, attn_mask, renorm=True, eps=1e-8, rescale_fn=rescale):
    """q (b, h, nq, d), k/v (b, h, n, d), attn_mask bool (b, nq, n) True = attend."""
    n = k.shape[-2]
    m = attn_mask[:, None]
    seen = ops.expand_dims(ops.any(m, axis=-2), -1)
    k, v = ops.where(seen, k, 0.0), ops.where(seen, v, 0.0)
    scores = ops.where(m, ops.einsum("bhqd,bhnd->bhqn", q, k), 0.0) * q.shape[-1] ** -0.5
    scores = ops.einsum("bhqn,nk->bhqk", scores, ops.cast(proj_k[:n], scores.dtype))
    if renorm:
        scores = scores * ops.cast(rescale_fn(proj_k, attn_mask, eps), scores.dtype)[:, None]
    attn = ops.softmax(scores, axis=-1)
    if renorm:
        attn = attn * ops.cast(rescale_fn(proj_v, attn_mask, eps), attn.dtype)[:, None]
    probs = ops.einsum("bhqk,nk->bhqn", attn, ops.cast(proj_v[:n], attn.dtype))
    return ops.einsum("bhqn,bhnd->bhqd", ops.where(m, probs, 0.0), v)


def sdpa(q, k, v):
    """Unmasked torch SDPA (the org/main project path calls self.attn(q, k, v) with no mask)."""
    scores = ops.einsum("bhqd,bhkd->bhqk", q * q.shape[-1] ** -0.5, k)
    return ops.einsum("bhqk,bhkd->bhqd", ops.softmax(scores, axis=-1), v)
