"""Quantizable masked-Linformer core: ``masked_linformer.attend`` / ``project`` (dce2e15) from factory leaves.

org/main ``Attention(attn_type="linformer")`` (``hepattn/models/attention.py:317-331``) calls one of two functions:
- ``attend(q, k, v, E, F, attn_mask)`` when an attention mask is given: query i sees ``E^T diag(M_i) K`` and
  ``F^T diag(M_i) V``, computed as ``(S * M) E`` and ``((a F^T) * M) v`` with ``S = q k^T`` (``AttendCore``);
- ``project(t, E, kv_mask)`` on k and v, then unmasked softmax attention over the k slots (``ProjectCore``).
With a mask, column j of E (F) is rescaled by ``||E[:, j]|| / ||E[valid, j]||`` (renorm); the numerator runs over all
``seq_len`` rows. Without a mask, ``project`` uses ``E[:n]`` and no rescale.

One graph, two leaf sets (``LayerFactory``): float leaves reproduce masked_linformer; HGQ2 leaves give the QAT model.

| Step | HGQ2 leaf |
|---|---|
| ``q k^T`` | ``QEinsum`` |
| ``S E[:n]``, ``a F[:n]^T`` (attend); ``E^T K``, ``F^T V`` (project) | ``QEinsumDense``, kernel ``E[:n]`` (``F[:n]``), capped fractional bits |
| ``x r_k``, ``x r_v`` | ``QEinsum`` |
| softmax over the k slots | ``QSoftmax``, never with a mask (a masked QSoftmax fills -1e9 and sizes its exp table for it) |
| ``M W``, ``W = E^2 / sum_rows E^2`` | ``QEinsum``: 1-bit mask lane, fixed-precision table lane |
| rsqrt | ``QUnaryFunctionLUT``, one homogeneous table, rank-2 input |
| ``p v`` (attend), ``a V'`` (project) | ``QEinsum`` |

Masks are selections (``keras.ops.where``), not multiplies: a NaN in a padded row cannot reach a valid output, and
selections cost no EBOPs.

Renorm precision (``renorm_bits``): the lane into the LUT and the table lane have ``2 * cap + 4`` fractional bits,
``cap`` the fractional-bit cap of E's and F's weight quantizers (``QuantSpec.seq_proj_frac_cap``, 7 by default:
a 2**19-entry table; hls4ml accepts at most 1e6 entries).

Tail rows ``E[n:]`` (``seq_len > n``) enter only the renorm numerator. A core without renorm has none: org's
unmasked ``project`` reads only ``E[:n]``, so the rows past n get no gradient there and are not created.
"""

import math

import keras
from hgq.quantizer import Quantizer
from keras import ops

from hepattn.keras_native.factory import LayerFactory, apply_softmax

RSQRT_FLOOR = 1e-16  # masked_linformer clamps kept^2 >= eps^2 = 1e-16; binds only on all-False rows, whose output is masked


def renorm_bits(frac_cap: int) -> dict[str, tuple[int, int]]:
    """Fixed ``(integer, fractional)`` bits of the renorm leaves for E/F weights with at most ``frac_cap`` fractional bits.

    The kept energy fraction ``x = M W`` lies in [0, 1], so the input lane has 1 integer bit. Its smallest nonzero
    value is ``2**-F`` (F = ``2 * frac_cap + 4``), so ``r = x**-0.5 <= 2**(frac_cap + 2)``: the table needs
    ``frac_cap + 3`` integer bits.
    """
    lane = 2 * frac_cap + 4
    return {"lane": (1, lane), "lut_out": (frac_cap + 3, 2 * frac_cap)}


def lut_entries(frac_cap: int) -> int:
    """Number of entries of the renorm rsqrt table for ``frac_cap``."""
    return 2 ** sum(renorm_bits(frac_cap)["lane"])


def rsqrt_floor(x):
    return ops.rsqrt(ops.maximum(x, RSQRT_FLOOR))


def _sub(name: str | None, part: str) -> str | None:
    return f"{name}_{part}" if name else None


class SeqProjection(keras.layers.Layer):
    """One (seq_len, k) sequence projection: rows [:n] are the kernel of a data x parameter einsum, rows [n:] the tail.

    The tail exists only with ``renorm`` and ``seq_len > n``. ``energy()`` gives ``W = E[:n]^2 / sum_all_rows E^2``
    from the quantized rows, so the renorm and the projection see the same E.
    """

    def __init__(self, factory: LayerFactory, equation: str, output_shape, n: int, seq_len: int, k: int, renorm: bool, name=None):
        super().__init__(name=name)
        self.n, self.seq_len, self.k = n, seq_len, k
        self.has_tail = renorm and seq_len > n
        bound = 1.0 / math.sqrt(k)  # masked_linformer.init_
        self._bound = bound
        cap = factory.quant.seq_proj_frac_cap if factory.quantize else None
        self.proj = factory.einsum_dense(
            equation, output_shape, name=_sub(name, "proj"), kernel_initializer=keras.initializers.RandomUniform(-bound, bound), max_frac_bits=cap
        )
        self.tail_q = None
        if factory.quantize and self.has_tail:
            self.tail_q = Quantizer(factory.capped_weight_config(cap), name=_sub(name, "tail_kq"))

    def build(self, input_shape) -> None:
        self.proj.build(input_shape)
        if tuple(self.proj.kernel.shape) != (self.n, self.k):
            raise ValueError(f"{self.name}: kernel {tuple(self.proj.kernel.shape)} != (n, k) = {(self.n, self.k)}")
        self.tail = None
        if self.has_tail:
            init = keras.initializers.RandomUniform(-self._bound, self._bound)
            self.tail = self.add_weight(name="tail", shape=(self.seq_len - self.n, self.k), initializer=init)
            if self.tail_q is not None:
                self.tail_q.build(self.tail.shape)

    @property
    def kernel(self):
        return self.proj.kernel

    def rows(self, training=None):
        """(head (n, k), tail (seq_len - n, k) or None) as the graph uses them (quantized in QAT)."""
        head, tail = self.proj.kernel, self.tail
        if hasattr(self.proj, "kq"):
            head = self.proj.kq(head, training=training)
        if tail is not None and self.tail_q is not None:
            tail = self.tail_q(tail, training=training)
        return head, tail

    def energy(self, training=None):
        head, tail = self.rows(training)
        full2 = ops.sum(ops.square(head), 0)
        if tail is not None:
            full2 = full2 + ops.sum(ops.square(tail), 0)
        return ops.square(head) / ops.maximum(full2, 1e-30)  # guard a column the quantizer set to 0

    def assign_full(self, proj) -> None:
        """Load a (seq_len, k) projection. Rows past n are dropped when this projection has no tail.

        Raises:
            ValueError: If ``proj`` is not (seq_len, k).
        """
        proj = keras.ops.convert_to_numpy(proj)
        if tuple(proj.shape) != (self.seq_len, self.k):
            raise ValueError(f"{self.name}: projection {tuple(proj.shape)} != (seq_len, k) = {(self.seq_len, self.k)}")
        self.proj.kernel.assign(proj[: self.n])
        if self.tail is not None:
            self.tail.assign(proj[self.n :])

    def full(self):
        """The (seq_len, k) float projection this layer holds; rows without a tail variable are absent (shape (n, k))."""
        head = self.proj.kernel
        return head if self.tail is None else ops.concatenate([head, self.tail], 0)

    def call(self, x, training=None):
        return self.proj(x, training=training)


def rescale(kept_layer, lut, mask, energy, training=None):
    """``masked_linformer._rescale`` as ``rsqrt(M W)``: bool (b, ..., n) -> (b, ..., k).

    The LUT sees a rank-2 (batch, features) input: QUnaryFunctionLUT on rank-3 inputs does not convert
    (docs/hgq/FPGA_STATUS.md, HGQ2 table-shape bug).
    """
    lead = tuple(mask.shape[1:-1])
    k = energy.shape[-1]
    kept = kept_layer([ops.cast(mask, energy.dtype), energy[None]], training=training)  # (b, ..., k)
    flat = ops.reshape(kept, (-1, math.prod(lead) * k))
    return ops.reshape(lut(flat, training=training), (-1, *lead, k))


def _make_renorm_leaves(core, factory: LayerFactory, kept_eq: str, mul_eq: str, name) -> None:
    """Per projection (k, v): mask x table einsum ``kept_*``, rsqrt table ``lut_*``, product ``mul_*`` with the scores."""
    # float leaves ignore the bits
    bits = renorm_bits(factory.quant.seq_proj_frac_cap) if factory.quantize else {"lane": (0, 0), "lut_out": (0, 0)}
    for part in ("k", "v"):
        setattr(core, f"kept_{part}", factory.mask_einsum(kept_eq, name=_sub(name, f"kept_{part}"), table_bits=bits["lane"]))
        setattr(core, f"lut_{part}", factory.lut(rsqrt_floor, name=_sub(name, f"rsqrt_{part}"), in_bits=bits["lane"], out_bits=bits["lut_out"]))
        setattr(core, f"mul_{part}", factory.einsum(mul_eq, name=_sub(name, f"mul_{part}")))


class AttendCore(keras.layers.Layer):
    """``masked_linformer.attend``: q (b, h, nq, d), k/v (b, h, n, d), attn_mask bool (b, nq, n) -> (b, h, nq, d).

    Fold key padding into ``attn_mask``. A query with an all-False row gets a zero output.
    """

    def __init__(self, factory: LayerFactory, heads: int, head_dim: int, nq: int, n: int, seq_len: int, k: int, renorm: bool = True, name=None):
        super().__init__(name=name)
        self.h, self.d, self.nq, self.n, self.seq_len, self.k, self.renorm = heads, head_dim, nq, n, seq_len, k, renorm
        self.qk = factory.einsum("bhqd,bhnd->bhqn", name=_sub(name, "qk"))
        self.E = SeqProjection(factory, "bhqn,nk->bhqk", (heads, nq, k), n, seq_len, k, renorm, name=_sub(name, "E"))
        self.F = SeqProjection(factory, "bhqk,nk->bhqn", (heads, nq, n), n, seq_len, k, renorm, name=_sub(name, "F"))
        self.softmax = factory.softmax(axis=-1, name=_sub(name, "softmax"))
        self.pv = factory.einsum("bhqn,bhnd->bhqd", name=_sub(name, "pv"))
        if renorm:
            _make_renorm_leaves(self, factory, "bqn,znk->bqk", "bhqk,bqk->bhqk", name)

    def build(self, input_shape=None) -> None:
        h, d, nq, n, k = self.h, self.d, self.nq, self.n, self.k
        self.qk.build([(None, h, nq, d), (None, h, n, d)])
        self.E.build((None, h, nq, n))
        self.F.build((None, h, nq, k))
        self.softmax.build((None, h, nq, k))
        self.pv.build([(None, h, nq, n), (None, h, n, d)])
        if self.renorm:
            for part in ("k", "v"):
                getattr(self, f"kept_{part}").build([(None, nq, n), (1, n, k)])
                getattr(self, f"lut_{part}").build((None, nq * k))
                getattr(self, f"mul_{part}").build([(None, h, nq, k), (None, nq, k)])

    def call(self, q, k, v, attn_mask, training=None):
        m = attn_mask[:, None]  # (b, 1, nq, n)
        # keys no query sees may hold NaN (padding); where-selecting the scores alone would let it into the matmuls
        seen = ops.expand_dims(ops.any(m, axis=-2), -1)
        k, v = ops.where(seen, k, 0.0), ops.where(seen, v, 0.0)
        s = ops.where(m, self.qk([q * self.d**-0.5, k], training=training), 0.0)
        s = self.E(s, training=training)
        if self.renorm:
            s = self.mul_k([s, rescale(self.kept_k, self.lut_k, attn_mask, self.E.energy(training), training)], training=training)
        a = apply_softmax(self.softmax, s, training=training)
        if self.renorm:
            a = self.mul_v([a, rescale(self.kept_v, self.lut_v, attn_mask, self.F.energy(training), training)], training=training)
        p = ops.where(m, self.F(a, training=training), 0.0)
        return self.pv([p, v], training=training)


class ProjectCore(keras.layers.Layer):
    """``project(k, E, kv_mask)``, ``project(v, F, kv_mask)``, then unmasked softmax attention over the k slots.

    q (b, h, nq, d), k/v (b, h, n, d), kv_mask bool (b, n) or None -> (b, h, nq, d). With ``masked=False`` the
    core has no renorm leaves and no tail, and a ``kv_mask`` raises: use it where org calls ``project`` without a
    mask (decoder query self-attention).
    """

    def __init__(
        self,
        factory: LayerFactory,
        heads: int,
        head_dim: int,
        nq: int,
        n: int,
        seq_len: int,
        k: int,
        renorm: bool = True,
        masked: bool = True,
        name=None,
    ):
        super().__init__(name=name)
        self.h, self.d, self.nq, self.n, self.seq_len, self.k = heads, head_dim, nq, n, seq_len, k
        self.masked = masked
        self.renorm = renorm and masked
        self.E = SeqProjection(factory, "bhnd,nk->bhkd", (heads, k, head_dim), n, seq_len, k, self.renorm, name=_sub(name, "E"))
        self.F = SeqProjection(factory, "bhnd,nk->bhkd", (heads, k, head_dim), n, seq_len, k, self.renorm, name=_sub(name, "F"))
        self.qk = factory.einsum("bhqd,bhkd->bhqk", name=_sub(name, "qk"))
        self.softmax = factory.softmax(axis=-1, name=_sub(name, "softmax"))
        self.av = factory.einsum("bhqk,bhkd->bhqd", name=_sub(name, "av"))
        if self.renorm:
            _make_renorm_leaves(self, factory, "bn,znk->bk", "bhkd,bk->bhkd", name)

    def build(self, input_shape=None) -> None:
        h, d, nq, n, k = self.h, self.d, self.nq, self.n, self.k
        self.E.build((None, h, n, d))
        self.F.build((None, h, n, d))
        self.qk.build([(None, h, nq, d), (None, h, k, d)])
        self.softmax.build((None, h, nq, k))
        self.av.build([(None, h, nq, k), (None, h, k, d)])
        if self.renorm:
            for part in ("k", "v"):
                getattr(self, f"kept_{part}").build([(None, n), (1, n, k)])
                getattr(self, f"lut_{part}").build((None, k))
                getattr(self, f"mul_{part}").build([(None, h, k, d), (None, k)])

    def call(self, q, k, v, kv_mask=None, training=None):
        if kv_mask is not None:
            if not self.masked:
                raise ValueError(f"{self.name}: built without mask support (masked=False), but a kv_mask was given")
            km = kv_mask[:, None, :, None]
            k, v = ops.where(km, k, 0.0), ops.where(km, v, 0.0)
        kp, vp = self.E(k, training=training), self.F(v, training=training)
        if kv_mask is not None and self.renorm:
            kp = self.mul_k([kp, rescale(self.kept_k, self.lut_k, kv_mask, self.E.energy(training), training)], training=training)
            vp = self.mul_v([vp, rescale(self.kept_v, self.lut_v, kv_mask, self.F.energy(training), training)], training=training)
        s = self.qk([q * self.d**-0.5, kp], training=training)
        a = apply_softmax(self.softmax, s, training=training)
        return self.av([a, vp], training=training)
