"""Layer factory switching between plain-Keras (float) and HGQ2 (quantized) leaves.

Every parameterized compute layer in hepattn.keras and hepattn.keras_native is created
through a LayerFactory so that the float parity-reference model and the HGQ2
quantization-aware model share one graph definition and one weight layout. Containers
(attention, dense stacks, encoder/decoder layers) are torch modules in the hybrid and
keras Layers in keras_native; only the leaves built here differ.

This module is backend-neutral (no torch import). ``hepattn.keras.factory`` re-exports it,
so it must not select or check the keras backend itself.

Quantized layers pick up their configuration from the HGQ2 config scopes that are
active at CONSTRUCTION time, so any code building quantized layers must run inside
``factory.scopes()``. Keeping the scopes inside the factory (rather than around the
YAML/jsonargparse instantiation site) is what makes the config mechanism robust to
instantiation order.
"""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any

import keras
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.layers import QDense, QEinsum, QEinsumDense, QSoftmax, QUnaryFunctionLUT
from hgq.quantizer import QuantizerConfig


@dataclass
class QuantSpec:
    """YAML-friendly description of an HGQ2 quantization configuration.

    Attributes:
        weight: kwargs for ``QuantizerConfigScope(place="weight", ...)``, e.g.
            ``{"default_q_type": "kbi", "overflow_mode": "SAT_SYM"}``. Note that in
            HGQ2 scopes ``q_type``/``place`` are *selectors*; the quantizer type is
            chosen with ``default_q_type``.
        datalane: kwargs for ``QuantizerConfigScope(place="datalane", ...)``.
        table: kwargs for ``QuantizerConfigScope(place="table", ...)`` — governs the
            QSoftmax exp/inv lookup-table output precision, which bounds softmax
            accuracy INDEPENDENTLY of weight/datalane bitwidths.
        ebops: kwargs for ``LayerConfigScope``; ``enable_ebops=True, beta0=1e-5``
            unless overridden.
        seq_proj_frac_cap: upper bound on the fractional bits of the Linformer sequence
            projections E and F (``hepattn.keras_native.masked_linformer``). The renorm
            lookup table is sized from it (``2 * cap + 4`` input bits); hls4ml caps a table
            at 1e6 entries, so 7 is the largest deployable value. Raise it only for
            precision studies.
    """

    weight: dict[str, Any] = field(default_factory=dict)
    datalane: dict[str, Any] = field(default_factory=dict)
    table: dict[str, Any] = field(default_factory=dict)
    ebops: dict[str, Any] = field(default_factory=dict)
    seq_proj_frac_cap: int = 7


class EinsumOp(keras.layers.Layer):
    """Float twin of hgq.layers.QEinsum: a fixed-equation einsum over a list of inputs."""

    def __init__(self, equation: str, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation

    def call(self, inputs):
        return keras.ops.einsum(self.equation, *inputs)

    def get_config(self):
        return {**super().get_config(), "equation": self.equation}


class SoftmaxOp(keras.layers.Layer):
    """Float twin of hgq.layers.QSoftmax with optional boolean masking.

    The mask uses the hepattn convention (True = participates). Masked-out scores are
    set to -inf before the softmax. Fully-masked rows produce ZERO attention weights,
    matching the torch fused SDPA kernels the reference model runs on (an unguarded
    softmax would yield NaN there, which poisons valid outputs downstream via 0*NaN
    in the values contraction — padded decoder keys hit exactly this case).
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    # the argument is named attn_mask (not mask) to keep it out of keras's built-in
    # mask-propagation machinery, which special-cases a call argument named `mask`
    def call(self, x, attn_mask=None):
        if attn_mask is None:
            return keras.ops.softmax(x, axis=self.axis)
        x = keras.ops.where(attn_mask, x, float("-inf"))
        out = keras.ops.softmax(x, axis=self.axis)
        any_valid = keras.ops.any(attn_mask, axis=self.axis, keepdims=True)
        return keras.ops.where(any_valid, out, keras.ops.zeros_like(out))

    def get_config(self):
        return {**super().get_config(), "axis": self.axis}


class LayerFactory:
    """Builds float or HGQ2-quantized leaf layers with a single code path.

    Args:
        quant: None for the float parity-reference model, or a QuantSpec (or its dict
            form, as it appears in YAML) for the HGQ2 quantized model.
    """

    def __init__(self, quant: QuantSpec | dict[str, Any] | None = None):
        if isinstance(quant, dict):
            quant = QuantSpec(**quant)
        self.quant = quant

    @property
    def quantize(self) -> bool:
        return self.quant is not None

    @contextmanager
    def scopes(self) -> Iterator[None]:
        """HGQ2 config scopes that must be active while quantized layers are constructed."""
        if self.quant is None:
            yield
            return
        with ExitStack() as stack:
            if self.quant.weight:
                stack.enter_context(QuantizerConfigScope(place="weight", **self.quant.weight))
            if self.quant.datalane:
                stack.enter_context(QuantizerConfigScope(place="datalane", **self.quant.datalane))
            if self.quant.table:
                stack.enter_context(QuantizerConfigScope(place="table", **self.quant.table))
            stack.enter_context(LayerConfigScope(**{"enable_ebops": True, "beta0": 1e-5, **self.quant.ebops}))
            yield

    def dense(
        self,
        units: int,
        activation: str | None = None,
        use_bias: bool = True,
        name: str | None = None,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
    ) -> keras.layers.Layer:
        kwargs = {
            "activation": activation,
            "use_bias": use_bias,
            "name": name,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
        }
        if self.quantize:
            return QDense(units, **kwargs)
        return keras.layers.Dense(units, **kwargs)

    def einsum(self, equation: str, name: str | None = None) -> keras.layers.Layer:
        if self.quantize:
            return QEinsum(equation, name=name)
        return EinsumOp(equation, name=name)

    def softmax(self, axis: int = -1, name: str | None = None) -> keras.layers.Layer:
        if self.quantize:
            return QSoftmax(axis=axis, name=name)
        return SoftmaxOp(axis=axis, name=name)

    def einsum_dense(
        self,
        equation: str,
        output_shape,
        name: str | None = None,
        kernel_initializer: Any = "glorot_uniform",
        max_frac_bits: int | None = None,
    ) -> keras.layers.Layer:
        """Data x parameter einsum without bias (``QEinsumDense`` / ``EinsumDense``).

        ``max_frac_bits`` caps the kernel quantizer's fractional bits (see ``capped_weight_config``).
        """
        kwargs = {"bias_axes": None, "kernel_initializer": kernel_initializer, "name": name}
        if not self.quantize:
            return keras.layers.EinsumDense(equation, output_shape, **kwargs)
        kq_conf = self.capped_weight_config(max_frac_bits) if max_frac_bits is not None else None
        return QEinsumDense(equation, output_shape, kq_conf=kq_conf, **kwargs)

    def lut(self, fn: Callable, name: str | None = None, *, in_bits: tuple[int, int], out_bits: tuple[int, int]) -> keras.layers.Layer:
        """Unary lookup table of ``fn`` on a non-negative input (``QUnaryFunctionLUT`` / ``Activation``).

        ``in_bits`` / ``out_bits`` are fixed ``(integer, fractional)`` bits: one homogeneous, unsigned, saturating,
        non-trainable quantizer each, so the table has ``2**sum(in_bits)`` entries. hls4ml's unary-LUT handler needs
        a homogeneous power-of-two input range and a homogeneous table, and fails on inputs of rank > 2.
        """
        if not self.quantize:
            return keras.layers.Activation(fn, name=name)
        iq = _fixed_config("datalane", *in_bits)
        oq = _fixed_config("table", *out_bits)
        return QUnaryFunctionLUT(fn, iq_conf=iq, oq_conf=oq, allow_heterogeneous_input=False, allow_heterogeneous_table=False, name=name)

    def mask_einsum(self, equation: str, name: str | None = None, *, table_bits: tuple[int, int]) -> keras.layers.Layer:
        """Einsum of a boolean mask (cast to float) with a non-negative table.

        Quantized: the mask lane is a fixed 1-bit unsigned quantizer; the table lane has the fixed ``table_bits``
        (``(integer, fractional)``) per element.
        """
        if not self.quantize:
            return EinsumOp(equation, name=name)
        mask_conf = _fixed_config("datalane", 1, 0)
        table_conf = _fixed_config("weight", *table_bits, homogeneous=False)
        return QEinsum(equation, iq_confs=[mask_conf, table_conf], name=name)

    @staticmethod
    def capped_weight_config(max_frac_bits: int) -> QuantizerConfig:
        """The active weight quantizer config as KIF, with the fractional bits capped at ``max_frac_bits``.

        Reads ``i0`` / ``f0`` from the active ``QuantizerConfigScope`` (a ``b0``/``i0`` spec sets ``f0 = b0 - i0``
        for every quantizer type), starts at ``min(f0, max_frac_bits)``, and constrains ``f <= max_frac_bits``
        during training. KIF because a KBI quantizer trains ``b`` and ``i`` separately, so ``f = b - i`` has no
        constraint. Call it inside ``scopes()``.
        """
        base = QuantizerConfig("kif", "weight")
        return QuantizerConfig("kif", "weight", i0=base["i0"], f0=min(base["f0"], max_frac_bits), fc=MinMax(-24, max_frac_bits))


def _fixed_config(place: str, int_bits: int, frac_bits: int, homogeneous: bool = True) -> QuantizerConfig:
    """Non-trainable unsigned saturating KIF quantizer with ``int_bits`` / ``frac_bits``.

    ``homogeneous`` gives one bitwidth for the whole tensor; otherwise one per element.
    """
    axes = {"homogeneous_axis": None, "heterogeneous_axis": ()} if homogeneous else {"homogeneous_axis": ()}
    return QuantizerConfig("kif", place, k0=False, i0=int_bits, f0=frac_bits, overflow_mode="SAT", trainable=False, **axes)


def apply_softmax(layer: keras.layers.Layer, x, attn_mask=None, training: bool = False):
    """Call a factory-built softmax with a hepattn-convention (True=keep) boolean mask.

    QSoftmax takes the mask via its keras `mask` argument (multiplicative, same
    polarity, and — like SoftmaxOp and the torch fused SDPA kernels — produces zero
    rows when fully masked); the float twin uses `attn_mask` to stay clear of keras's
    mask-propagation machinery. QSoftmax is called directly rather than through a
    subclass so its class path stays `hgq.layers.softmax.QSoftmax` — the hls4ml
    converter dispatches on exactly that path.
    """
    if isinstance(layer, QSoftmax):
        return layer(x, training=training, mask=attn_mask)
    return layer(x, attn_mask=attn_mask, training=training)
