"""Prototype HGQ2 expression of masked_linformer.attend / project (+ renorm) for keras_native.

One graph, two leaf sets (as hepattn.keras_native.factory.LayerFactory): float leaves give the parity
reference, HGQ2 leaves give the QAT model. Every data-path multiply is an HGQ2 leaf:
  data x data     -> QEinsum          (q k^T, p v, rescale products, mask @ W)
  data x param    -> QEinsumDense     (S E, A F^T, k E, v F): E / F are the kernels, one weight quantizer each
  nonlinear       -> QSoftmax, QUnaryFunctionLUT (rsqrt of the renorm energy, homogeneous table)
Masks are selections (keras.ops.where), not multiplies; they carry no EBOPs.
"""

import math

import keras
from hgq.layers import QEinsum, QEinsumDense, QUnaryFunctionLUT
from hgq.quantizer import Quantizer, QuantizerConfig
from keras import ops

from hepattn.keras_native.attention import recombine_heads, separate_heads
from hepattn.keras_native.factory import LayerFactory, apply_softmax

# renorm-path precisions (not set by the QuantSpec scopes): LUT address / table output, energy-table lane
RENORM_PREC = {"lut_in_f": 12, "lut_out_i": 7, "lut_out_f": 8, "w_f": 12}
RSQRT_FLOOR = 1e-16  # masked_linformer clamps kept^2 >= eps^2 = 1e-16; only binds on all-False rows, whose output is masked


def rsqrt_floor(x):
    return ops.rsqrt(ops.maximum(x, RSQRT_FLOOR))


class Leaves:
    """LayerFactory plus the two leaf kinds this design adds (data x param einsum, unary LUT)."""

    def __init__(self, factory: LayerFactory):
        self.f = factory

    @property
    def quantize(self):
        return self.f.quantize

    def einsum(self, eq, name, iq_confs=None):
        if self.quantize and iq_confs is not None:
            return QEinsum(eq, iq_confs=iq_confs, name=name)
        return self.f.einsum(eq, name=name)

    def einsum_dense(self, eq, output_shape, name):
        cls = QEinsumDense if self.quantize else keras.layers.EinsumDense
        return cls(eq, output_shape, bias_axes=None, name=name)

    def lut(self, fn, name):
        if not self.quantize:
            return keras.layers.Activation(fn, name=name)
        # homogeneous unsigned input in [0, 2): a 2^13-entry table; hls4ml's UnaryLUT handler needs a homogeneous
        # power-of-two input range and allow_heterogeneous_table=False (converters/keras_v3/hgq2/unary_lut.py)
        p = RENORM_PREC
        iq = QuantizerConfig("kif", "datalane", k0=False, i0=1, f0=p["lut_in_f"], overflow_mode="SAT", homogeneous_axis=None, heterogeneous_axis=())
        oq = QuantizerConfig("kif", "table", k0=False, i0=p["lut_out_i"], f0=p["lut_out_f"])
        return QUnaryFunctionLUT(fn, iq_conf=iq, oq_conf=oq, allow_heterogeneous_input=False, allow_heterogeneous_table=False, name=name)

    def mask_einsum(self, eq, name):
        """Binary mask x energy table: the mask lane is a fixed 1-bit unsigned quantizer, the table per element."""
        if not self.quantize:
            return self.f.einsum(eq, name=name)
        m_conf = QuantizerConfig("kif", "datalane", k0=False, i0=1, f0=0, overflow_mode="SAT", trainable=False, homogeneous_axis=None, heterogeneous_axis=())
        w_conf = QuantizerConfig("kif", "weight", k0=False, i0=1, f0=RENORM_PREC["w_f"], overflow_mode="SAT", homogeneous_axis=())
        return QEinsum(eq, iq_confs=[m_conf, w_conf], name=name)


class SeqProjection(keras.layers.Layer):
    """One (seq_len, k) Linformer projection split as kernel rows [:n] (an EinsumDense kernel) + tail rows [n:].

    The tail only enters the renorm numerator ||E[:, j]|| over all seq_len rows (batch-padding invariant, as
    masked_linformer). ``energy()`` returns W = E^2 / sum_rows E^2 from the QUANTIZED rows, so the renorm and the
    projection see the same E.
    """

    def __init__(self, leaves: Leaves, eq: str, output_shape, n: int, seq_len: int, name: str):
        super().__init__(name=name)
        self.n, self.seq_len = n, seq_len
        self.dense = leaves.einsum_dense(eq, output_shape, name=f"{name}_proj")
        self.tail_q = Quantizer(QuantizerConfig("default", "weight"), name=f"{name}_tail_kq") if leaves.quantize else None

    def build_all(self, input_shape, k):
        self.dense.build(input_shape)
        assert tuple(self.dense.kernel.shape) == (self.n, k), self.dense.kernel.shape  # kernel spec 'nk' in every equation
        init = keras.initializers.RandomUniform(-(k**-0.5), k**-0.5)  # masked_linformer.init_
        self.dense.kernel.assign(init(self.dense.kernel.shape))
        self.tail = None
        if self.seq_len > self.n:
            self.tail = self.add_weight(name="tail", shape=(self.seq_len - self.n, k), initializer=init)
            if self.tail_q is not None:
                self.tail_q.build(self.tail.shape)

    def rows(self, training=None):
        """(head (n, k), tail (seq_len - n, k) or None) as used in the graph (quantized in QAT)."""
        head = self.dense.kernel
        if isinstance(self.dense, QEinsumDense):
            head = self.dense.kq(head, training=training)
        tail = self.tail
        if tail is not None and self.tail_q is not None:
            tail = self.tail_q(tail, training=training)
        return head, tail

    def energy(self, training=None):
        head, tail = self.rows(training)
        full2 = ops.sum(ops.square(head), 0)
        if tail is not None:
            full2 = full2 + ops.sum(ops.square(tail), 0)
        return ops.square(head) / ops.maximum(full2, 1e-30)  # guard a column the quantizer collapsed to 0

    def assign_full(self, proj):
        """Load a torch (seq_len, k) projection."""
        self.dense.kernel.assign(proj[: self.n])
        if self.tail is not None:
            self.tail.assign(proj[self.n :])

    def call(self, x, training=None):
        return self.dense(x, training=training)


def rescale(kept_layer, lut, mask, energy, training=None):
    """masked_linformer._rescale as rsqrt(mask @ W): (b, ..., n) bool -> (b, ..., k).

    The LUT sees a rank-2 (batch, features) tensor: QUnaryFunctionLUT on rank-3 inputs does not convert
    (docs/hgq/FPGA_STATUS.md, HGQ2 table-shape bug).
    """
    lead = tuple(mask.shape[1:-1])
    kept2n = kept_layer([ops.cast(mask, "float32"), energy[None]], training=training)  # (b, ..., k)
    flat = ops.reshape(kept2n, (-1, math.prod(lead) * energy.shape[-1]))
    return ops.reshape(lut(flat, training=training), (-1, *lead, energy.shape[-1]))


class AttendCore(keras.layers.Layer):
    """masked_linformer.attend: q (b,h,nq,d), k/v (b,h,n,d), attn_mask bool (b,nq,n) -> (b,h,nq,d)."""

    def __init__(self, leaves: Leaves, heads, head_dim, nq, n, seq_len, k, renorm=True, name="attend"):
        super().__init__(name=name)
        self.h, self.d, self.nq, self.n, self.k, self.renorm = heads, head_dim, nq, n, k, renorm
        self.qk = leaves.einsum("bhqd,bhnd->bhqn", name=f"{name}_qk")
        self.E = SeqProjection(leaves, "bhqn,nk->bhqk", (heads, nq, k), n, seq_len, name=f"{name}_E")
        self.F = SeqProjection(leaves, "bhqk,nk->bhqn", (heads, nq, n), n, seq_len, name=f"{name}_F")
        self.softmax = leaves.f.softmax(axis=-1, name=f"{name}_softmax")
        self.pv = leaves.einsum("bhqn,bhnd->bhqd", name=f"{name}_pv")
        if renorm:
            self.kept_k = leaves.mask_einsum("bqn,znk->bqk", name=f"{name}_kept_k")
            self.kept_v = leaves.mask_einsum("bqn,znk->bqk", name=f"{name}_kept_v")
            self.lut_k = leaves.lut(rsqrt_floor, name=f"{name}_rsqrt_k")
            self.lut_v = leaves.lut(rsqrt_floor, name=f"{name}_rsqrt_v")
            self.mul_k = leaves.einsum("bhqk,bqk->bhqk", name=f"{name}_mul_k")
            self.mul_v = leaves.einsum("bhqk,bqk->bhqk", name=f"{name}_mul_v")

    def build(self, *_):
        h, d, nq, n, k = self.h, self.d, self.nq, self.n, self.k
        self.qk.build([(None, h, nq, d), (None, h, n, d)])
        self.E.build_all((None, h, nq, n), k)
        self.F.build_all((None, h, nq, k), k)
        self.softmax.build((None, h, nq, k))
        self.pv.build([(None, h, nq, n), (None, h, n, d)])
        if self.renorm:
            for kept in (self.kept_k, self.kept_v):
                kept.build([(None, nq, n), (1, n, k)])
            for lut in (self.lut_k, self.lut_v):
                lut.build((None, nq * k))
            for mul in (self.mul_k, self.mul_v):
                mul.build([(None, h, nq, k), (None, nq, k)])

    def call(self, q, k, v, attn_mask, training=None):
        m = attn_mask[:, None]
        seen = ops.expand_dims(ops.any(m, axis=-2), -1)
        k, v = ops.where(seen, k, 0.0), ops.where(seen, v, 0.0)  # NaN guard only (padded rows); a no-op in fixed point
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
    """org/main project path: project(k, E, kv_mask), project(v, F, kv_mask), then unmasked SDPA over k slots."""

    def __init__(self, leaves: Leaves, heads, head_dim, nq, n, seq_len, k, renorm=True, name="project"):
        super().__init__(name=name)
        self.h, self.d, self.nq, self.n, self.k, self.renorm = heads, head_dim, nq, n, k, renorm
        self.E = SeqProjection(leaves, "bhnd,nk->bhkd", (heads, k, head_dim), n, seq_len, name=f"{name}_E")
        self.F = SeqProjection(leaves, "bhnd,nk->bhkd", (heads, k, head_dim), n, seq_len, name=f"{name}_F")
        self.qk = leaves.einsum("bhqd,bhkd->bhqk", name=f"{name}_qk")
        self.softmax = leaves.f.softmax(axis=-1, name=f"{name}_softmax")
        self.av = leaves.einsum("bhqk,bhkd->bhqd", name=f"{name}_av")
        if renorm:
            self.kept_k = leaves.mask_einsum("bn,znk->bk", name=f"{name}_kept_k")
            self.kept_v = leaves.mask_einsum("bn,znk->bk", name=f"{name}_kept_v")
            self.lut_k = leaves.lut(rsqrt_floor, name=f"{name}_rsqrt_k")
            self.lut_v = leaves.lut(rsqrt_floor, name=f"{name}_rsqrt_v")
            self.mul_k = leaves.einsum("bhkd,bk->bhkd", name=f"{name}_mul_k")
            self.mul_v = leaves.einsum("bhkd,bk->bhkd", name=f"{name}_mul_v")

    def build(self, *_):
        h, d, nq, n, k = self.h, self.d, self.nq, self.n, self.k
        self.E.build_all((None, h, n, d), k)
        self.F.build_all((None, h, n, d), k)
        self.qk.build([(None, h, nq, d), (None, h, k, d)])
        self.softmax.build((None, h, nq, k))
        self.av.build([(None, h, nq, k), (None, h, k, d)])
        if self.renorm:
            for kept in (self.kept_k, self.kept_v):
                kept.build([(None, n), (1, n, k)])
            for lut in (self.lut_k, self.lut_v):
                lut.build((None, k))
            for mul in (self.mul_k, self.mul_v):
                mul.build([(None, h, k, d), (None, k)])

    def call(self, q, k, v, kv_mask=None, training=None):
        if kv_mask is not None:
            km = kv_mask[:, None, :, None]
            k, v = ops.where(km, k, 0.0), ops.where(km, v, 0.0)
        kp, vp = self.E(k, training=training), self.F(v, training=training)
        if kv_mask is not None and self.renorm:
            kp = self.mul_k([kp, rescale(self.kept_k, self.lut_k, kv_mask, self.E.energy(training), training)], training=training)
            vp = self.mul_v([vp, rescale(self.kept_v, self.lut_v, kv_mask, self.F.energy(training), training)], training=training)
        s = self.qk([q * self.d**-0.5, kp], training=training)
        a = apply_softmax(self.softmax, s, training=training)
        return self.av([a, vp], training=training)


class LinAttention(keras.layers.Layer):
    """org/main Attention with attn_type='linformer': q/k/v/out projections (bias) around a core."""

    def __init__(self, factory: LayerFactory, dim, heads, nq, n, seq_len, k, core="attend", name="lin"):
        super().__init__(name=name)
        leaves = Leaves(factory)
        self.dim, self.heads, self.nq, self.n = dim, heads, nq, n
        self.q_proj = factory.dense(dim, name=f"{name}_q_proj")
        self.k_proj = factory.dense(dim, name=f"{name}_k_proj")
        self.v_proj = factory.dense(dim, name=f"{name}_v_proj")
        self.out_proj = factory.dense(dim, name=f"{name}_out_proj")
        cls = AttendCore if core == "attend" else ProjectCore
        self.core = cls(leaves, heads, dim // heads, nq, n, seq_len, k, name=f"{name}_{core}")
        self.core_kind = core

    def build(self, *_):
        self.q_proj.build((None, self.nq, self.dim))
        self.k_proj.build((None, self.n, self.dim))
        self.v_proj.build((None, self.n, self.dim))
        self.out_proj.build((None, self.nq, self.dim))
        self.core.build()

    def call(self, q, kv, kv_mask=None, attn_mask=None, training=None):
        qp = separate_heads(self.q_proj(q, training=training), self.heads)
        kp = separate_heads(self.k_proj(kv, training=training), self.heads)
        vp = separate_heads(self.v_proj(kv, training=training), self.heads)
        if self.core_kind == "attend":
            mask = attn_mask if kv_mask is None else ops.logical_and(attn_mask, kv_mask[:, None])
            out = self.core(qp, kp, vp, mask, training=training)
        else:
            out = self.core(qp, kp, vp, kv_mask, training=training)
        return self.out_proj(recombine_heads(out), training=training)


class ExplicitCore(keras.layers.Layer):
    """attend in its definitional form, renorm off: K'_i = E^T diag(M_i) K, V'_i = F^T diag(M_i) V per query.

    Only for the cost comparison: same function as AttendCore(renorm=False), (b, h, nq, n, d) intermediates.
    """

    def __init__(self, leaves: Leaves, heads, head_dim, nq, n, seq_len, k, name="explicit"):
        super().__init__(name=name)
        self.h, self.d, self.nq, self.n, self.k = heads, head_dim, nq, n, k
        self.E = SeqProjection(leaves, "bhqnd,nk->bhqkd", (heads, nq, k, head_dim), n, seq_len, name=f"{name}_E")
        self.F = SeqProjection(leaves, "bhqnd,nk->bhqkd", (heads, nq, k, head_dim), n, seq_len, name=f"{name}_F")
        self.qk = leaves.einsum("bhqd,bhqkd->bhqk", name=f"{name}_qk")
        self.softmax = leaves.f.softmax(axis=-1, name=f"{name}_softmax")
        self.av = leaves.einsum("bhqk,bhqkd->bhqd", name=f"{name}_av")

    def build(self, *_):
        h, d, nq, n, k = self.h, self.d, self.nq, self.n, self.k
        self.E.build_all((None, h, nq, n, d), k)
        self.F.build_all((None, h, nq, n, d), k)
        self.qk.build([(None, h, nq, d), (None, h, nq, k, d)])
        self.softmax.build((None, h, nq, k))
        self.av.build([(None, h, nq, k), (None, h, nq, k, d)])

    def call(self, q, k, v, attn_mask, training=None):
        m = attn_mask[:, None, :, :, None]  # (b, 1, nq, n, 1)
        kq = ops.where(m, k[:, :, None], 0.0)  # diag(M_i) K for every query: (b, h, nq, n, d)
        vq = ops.where(m, v[:, :, None], 0.0)
        kp, vp = self.E(kq, training=training), self.F(vq, training=training)
        s = self.qk([q * self.d**-0.5, kp], training=training)
        a = apply_softmax(self.softmax, s, training=training)
        out = self.av([a, vp], training=training)
        return ops.where(ops.any(attn_mask, -1)[:, None, :, None], out, 0.0)
