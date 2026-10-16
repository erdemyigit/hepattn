"""Torch-free norm helpers: a copy of ``hepattn.models.norm.get_hybrid_norm_config`` (that module imports torch)
and the keras twins of the torch LayerNorm and FastLayerNorm.
"""

import keras

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
# is also non-affine (NON_AFFINE_NORMS).
SUPPORTED_NORMS = ("LayerNorm", "CustomLayerNorm", "FastLayerNorm")
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


def make_norm(norm: str, name: str | None = None):
    """A keras twin of the torch norm ``norm``: LayerNormalization with the torch epsilon 1e-5 (keras default: 1e-3).

    FastLayerNorm has no affine parameters (``center=False, scale=False``: 0 variables).
    """
    check_norm(norm)
    affine = norm not in NON_AFFINE_NORMS
    return keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, center=affine, scale=affine, name=name)
