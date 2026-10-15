"""Torch-free norm helpers: a copy of ``hepattn.models.norm.get_hybrid_norm_config`` (that module imports torch)
and the keras twin of the torch LayerNorm.
"""

import keras


def get_hybrid_norm_config(norm: str | None, depth: int, hybrid_norm: bool, qkv_norm: bool) -> tuple[str | None, bool, bool]:
    """Get the normalization configuration for HybridNorm.

    Args:
        norm: The normalization type.
        depth: The layer depth.
        hybrid_norm: Whether to use HybridNorm.
        qkv_norm: Whether to use QKV normalization.

    Returns:
        attn_norm: The normalization to use before attention.
        dense_post_norm: Whether to use post-normalization for the dense layer.
        qkv_norm: Whether to use QKV normalization.
    """
    qkv_norm = qkv_norm or hybrid_norm
    is_hybrid_subsequent = hybrid_norm and depth > 0
    attn_norm = None if is_hybrid_subsequent else norm
    dense_post_norm = is_hybrid_subsequent

    return attn_norm, dense_post_norm, qkv_norm


# torch NORM_TYPES names (hepattn/models/norm.py) this port expresses. CustomLayerNorm only adds a dtype
# cast back to the input dtype, which keras does by its dtype policy, so it is the same op in fp32.
SUPPORTED_NORMS = ("LayerNorm", "CustomLayerNorm")
TORCH_NORMS = ("LayerNorm", "RMSNorm", "CustomLayerNorm", "FastLayerNorm", "CustomRMSNorm", "SimpleRMSNorm", "DyT")


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
    """A keras twin of the torch norm ``norm``: LayerNormalization with the torch epsilon 1e-5 (keras default: 1e-3)."""
    check_norm(norm)
    return keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)
