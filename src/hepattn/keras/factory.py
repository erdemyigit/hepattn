"""Layer factory switching between plain-Keras (float) and HGQ2 (quantized) leaves.

Every parameterized compute layer in hepattn.keras is created through a LayerFactory
so that the float parity-reference model and the HGQ2 quantization-aware model share
one graph definition and one weight layout. Containers (attention, dense stacks,
encoder/decoder layers) are torch modules; only the leaves built here differ.

Quantized layers pick up their configuration from the HGQ2 config scopes that are
active at CONSTRUCTION time, so any code building quantized layers must run inside
``factory.scopes()``. Keeping the scopes inside the factory (rather than around the
YAML/jsonargparse instantiation site) is what makes the config mechanism robust to
instantiation order.
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any

from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QDense, QEinsum, QSoftmax

from hepattn.keras import keras


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
    """

    weight: dict[str, Any] = field(default_factory=dict)
    datalane: dict[str, Any] = field(default_factory=dict)
    table: dict[str, Any] = field(default_factory=dict)
    ebops: dict[str, Any] = field(default_factory=dict)


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

    def dense(self, units: int, activation: str | None = None, use_bias: bool = True, name: str | None = None) -> keras.layers.Layer:
        if self.quantize:
            return QDense(units, activation=activation, use_bias=use_bias, name=name)
        return keras.layers.Dense(units, activation=activation, use_bias=use_bias, name=name)

    def einsum(self, equation: str, name: str | None = None) -> keras.layers.Layer:
        if self.quantize:
            return QEinsum(equation, name=name)
        return EinsumOp(equation, name=name)

    def softmax(self, axis: int = -1, name: str | None = None) -> keras.layers.Layer:
        if self.quantize:
            return QSoftmax(axis=axis, name=name)
        return SoftmaxOp(axis=axis, name=name)


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
