"""Torch-free norm helpers: a copy of ``hepattn.models.norm.get_hybrid_norm_config`` (that module imports torch)
and the keras twins of the torch LayerNorm, FastLayerNorm and DyT.
"""

import keras

from hepattn.keras_native.factory import LayerFactory

DENSE_NORM_PLACEMENTS = ("hybridnorm", "legacy")


def get_hybrid_norm_config(
    norm: str | None, depth: int, hybrid_norm: bool, qkv_norm: bool, dense_norm_placement: str = "hybridnorm"
) -> tuple[str | None, bool, bool]:
    """Get the normalization configuration for HybridNorm (copy of the torch function).

    Args:
        norm: The normalization type.
        depth: The layer depth.
        hybrid_norm: Whether to use HybridNorm.
        qkv_norm: Whether to use QKV normalization.
        dense_norm_placement: "hybridnorm" (2503.04598: dense pre-norm in the first layer, post-norm after it) or
            "legacy" (the clic-paper tag: post-norm in the first layer and in every layer without hybrid_norm,
            pre-norm in the layers after it).

    Returns:
        attn_norm: The normalization to use before attention.
        dense_post_norm: Whether to use post-normalization for the dense layer.
        qkv_norm: Whether to use QKV normalization.

    Raises:
        ValueError: If dense_norm_placement is not a known placement.
    """
    if dense_norm_placement not in DENSE_NORM_PLACEMENTS:
        raise ValueError(f"Unsupported dense_norm_placement: {dense_norm_placement}. Must be 'hybridnorm' or 'legacy'.")

    qkv_norm = qkv_norm or hybrid_norm
    is_hybrid_subsequent = hybrid_norm and depth > 0
    attn_norm = None if is_hybrid_subsequent else norm
    dense_post_norm = is_hybrid_subsequent if dense_norm_placement == "hybridnorm" else not is_hybrid_subsequent

    return attn_norm, dense_post_norm, qkv_norm


# torch NORM_TYPES names (hepattn/models/norm.py) this port expresses. CustomLayerNorm and FastLayerNorm add a dtype
# cast back to the input dtype, which keras does by its dtype policy, so in fp32 that cast is a no-op. FastLayerNorm
# is also non-affine (NON_AFFINE_NORMS). DyT is the only one with quantized leaves (``DyT``); the LayerNorms stay float
# keras LayerNormalization in a quantized model.
SUPPORTED_NORMS = ("LayerNorm", "CustomLayerNorm", "FastLayerNorm", "DyT")
TORCH_NORMS = ("LayerNorm", "RMSNorm", "CustomLayerNorm", "FastLayerNorm", "CustomRMSNorm", "SimpleRMSNorm", "DyT")
# torch FastLayerNorm is nn.LayerNorm(elementwise_affine=False): no weight, no bias
NON_AFFINE_NORMS = ("FastLayerNorm",)


def check_norm(norm: str | None) -> None:
    """Raise unless ``norm`` is None or a norm this port implements.

    Raises:
        ValueError: If ``norm`` is not a hepattn norm name at all.
        NotImplementedError: If it is a hepattn norm that this port does not implement.
    """
    if norm is None:
        return
    if not isinstance(norm, str) or norm not in TORCH_NORMS:
        raise ValueError(f"Unsupported norm: {norm}. Must be one of {list(TORCH_NORMS)}")
    if norm not in SUPPORTED_NORMS:
        raise NotImplementedError(f"norm '{norm}' is not ported to keras_native (supported: {list(SUPPORTED_NORMS)})")


def make_norm(norm: str, name: str | None = None, *, dim: int | None = None, factory: LayerFactory | None = None):
    """A keras twin of the torch norm ``norm`` over the last axis of a (B, N, ``dim``) input.

    LayerNorm variants: LayerNormalization with the torch epsilon 1e-5 (keras default: 1e-3); FastLayerNorm has no
    affine parameters (``center=False, scale=False``: 0 variables). DyT: ``DyT(dim, factory)``.

    Raises:
        ValueError: For DyT without ``dim``.
    """
    check_norm(norm)
    if norm == "DyT":
        if dim is None:
            raise ValueError("DyT needs the feature size dim")
        return DyT(dim, factory=factory, name=name)
    affine = norm not in NON_AFFINE_NORMS
    return keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, center=affine, scale=affine, name=name)


# Quantized DyT tanh table, (integer, fractional) bits of its signed input u = alpha x and of its signed output.
# Input [-4, 4) in steps of 2**-10 (2**13 = 8192 entries): 1 - tanh(4) = 6.7e-4 is below one output step 2**-10 =
# 9.8e-4, so saturating u at 4 costs less than the output rounding (from u = 3.73 on the table already reads
# 1 - 2**-10, the largest output). Output (-1, 1) in steps of 2**-10; tanh' <= 1, so one input step moves the output
# by at most one output step and the table error is <= 2**-11 (rounding) + 2**-11 (input step x slope 1) = 2**-10.
# That is 4x below the 2**-8 step of the datalane init (f0 8) that quantizes the next layer's input, and the table has
# the 13 input bits of the LUT SiLU that Alkaid converted bit-exactly in the 2026-10-28 trial (ALKAID_LOG.md S2).
DYT_TANH_IN = (2, 10)
DYT_TANH_OUT = (0, 10)


class DyT(keras.layers.Layer):
    """Dynamic Tanh (arXiv:2503.10622), the twin of torch ``hepattn.models.norm.DyT``: ``tanh(alpha x) w + b``.

    ``alpha`` is one scalar (init ``alpha_init``, torch 0.5), ``w`` / ``b`` are per channel (ones / zeros). Three
    factory leaves on a (B, N, dim) input, the same variables in the float and the quantized model:

    | step | leaf | quantized |
    |---|---|---|
    | ``u = alpha x`` | ``{name}_alpha``: einsum ``bnc,z->bncz``, kernel (1,) | ``QEinsumDense``: datalane iq on x, weight kq on alpha |
    | ``t = tanh(u)`` | ``{name}_tanh`` on u as (B, N dim) | ``QUnaryFunctionLUT``: fixed signed homogeneous SAT ``DYT_TANH_IN`` -> ``DYT_TANH_OUT`` |
    | ``t w + b`` | ``{name}_affine``: einsum ``bnc,c->bnc``, bias ``c`` | ``QEinsumDense``: datalane iq on t, weight kq on w, bias |

    The product alpha x is quantized only by the table's input quantizer, whose range is the one of tanh, so a trained
    alpha does not move the table's saturation point (HGQ2's ``QAffinedUnaryFunctionLUT`` puts the fixed range on x).
    The LUT input is rank 2 because hls4ml's LUT handler fails on rank > 2 (``LayerFactory.lut``). Every multiply is by
    a quantized (dyadic) weight: Alkaid implements it as an exact constant multiply (calad0i/alkaid#8).
    """

    def __init__(self, dim: int, factory: LayerFactory | None = None, alpha_init: float = 0.5, name: str | None = None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        factory = factory or LayerFactory()
        self.dim = dim
        self.alpha_init = alpha_init

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        self.alpha = factory.einsum_dense(
            "bnc,z->bncz", (None, dim, 1), name=sub("alpha"), kernel_initializer=keras.initializers.Constant(alpha_init)
        )
        self.tanh = factory.lut(keras.ops.tanh, name=sub("tanh"), in_bits=DYT_TANH_IN, out_bits=DYT_TANH_OUT, signed=True)
        self.affine = factory.einsum_dense("bnc,c->bnc", (None, dim), name=sub("affine"), kernel_initializer="ones", bias_axes="c")

    def build(self, input_shape) -> None:
        """Build the three leaves.

        Raises:
            ValueError: Unless the input is (B, N, dim) with a static N.
        """
        if len(input_shape) != 3 or input_shape[-1] != self.dim or input_shape[1] is None:
            raise ValueError(f"{self.name}: DyT takes (B, N, {self.dim}) with a static N, got {tuple(input_shape)}")
        n = input_shape[1]
        self.alpha.build(tuple(input_shape))
        self.tanh.build((None, n * self.dim))
        self.affine.build(tuple(input_shape))
        if not hasattr(self.affine, "full_output_shape"):
            # HGQ2 0.1.9 QEinsumDense with a bias reads full_output_shape in its EBOPs, which keras 3.15 no longer sets
            # (AttributeError in a training call); HGQ2 0.2.0 sets it in build (from compute_output_shape, which here
            # leaves N unresolved: output_shape (None, dim))
            self.affine.full_output_shape = (None, n, self.dim)

    def call(self, x, training=None):
        n = x.shape[1]
        u = keras.ops.reshape(self.alpha(x, training=training), (-1, n * self.dim))
        t = keras.ops.reshape(self.tanh(u, training=training), (-1, n, self.dim))
        return self.affine(t, training=training)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "alpha_init": self.alpha_init}
