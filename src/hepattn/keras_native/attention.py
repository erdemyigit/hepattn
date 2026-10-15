"""Keras-native dense multi-head attention (torch-SDPA semantics) and ``merge_masks``.

Mirrors ``hepattn.keras.attention.KerasAttention`` op for op: three factory-built projections, an explicit
scores einsum -> masked softmax -> values einsum core, and the output projection. Leaf names are identical
(``{name}_q_proj``, ``{name}_scores``, ...). Differences to the hybrid:
- QKV norms are keras ``LayerNormalization(epsilon=1e-5)`` (``{name}_q_norm`` ...), not torch modules.
- The value residual is returned instead of written into a dict: with ``value_residual=True`` ``call``
  returns ``(out, v_cache)`` (see ``Attention.call``).
Masks follow the hepattn convention: True = takes part in attention.
"""

import keras

from hepattn.keras_native.factory import LayerFactory, apply_softmax
from hepattn.keras_native.norm import check_norm, make_norm


def merge_masks(q_mask, kv_mask, attn_mask, q_shape, k_shape):
    """Combine padding masks and an attention mask (True = attend), as ``hepattn.models.attention.merge_masks``.

    ``q_shape`` / ``k_shape`` are the (batch, seq, dim) shapes of the query and key tensors; they size the
    all-True default when only one of the two padding masks is given.
    """
    merged = None
    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = keras.ops.ones(tuple(q_shape[:-1]), dtype="bool")
        if kv_mask is None:
            kv_mask = keras.ops.ones(tuple(k_shape[:-1]), dtype="bool")
        merged = keras.ops.logical_and(keras.ops.expand_dims(q_mask, -1), keras.ops.expand_dims(kv_mask, -2))
    if attn_mask is not None:
        merged = keras.ops.logical_and(attn_mask, merged) if merged is not None else attn_mask
    return merged


def separate_heads(x, num_heads: int):
    """(B, S, D) -> (B, H, S, D/H)."""
    _, seq, dim = x.shape
    return keras.ops.transpose(keras.ops.reshape(x, (-1, seq, num_heads, dim // num_heads)), (0, 2, 1, 3))


def recombine_heads(x):
    """(B, H, S, Dh) -> (B, S, H * Dh)."""
    _, heads, seq, head_dim = x.shape
    return keras.ops.reshape(keras.ops.transpose(x, (0, 2, 1, 3)), (-1, seq, heads * head_dim))


class Attention(keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        torch_compile: bool = False,
        window_size: int | None = None,
        qkv_norm: bool = False,
        norm: str | None = None,
        value_residual: bool = False,
        is_first_layer: bool = False,
        factory: LayerFactory | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Multi-head attention from factory-built (quantizable) leaves; arguments as hepattn Attention.

        A hepattn norm that keras_native does not implement raises NotImplementedError (``check_norm``).

        Raises:
            ValueError: For a non-torch backend, torch_compile/window_size, or qkv_norm without norm.
        """
        super().__init__(name=name, **kwargs)
        factory = factory or LayerFactory()

        if dim % num_heads != 0:
            raise ValueError("num_heads must divide dim.")
        if attn_type != "torch":
            raise ValueError(f"Attention implements torch-SDPA semantics only, got attn_type='{attn_type}'")
        if torch_compile or window_size is not None:
            raise ValueError("torch_compile / window_size are not supported by keras_native Attention")
        if qkv_norm and not norm:
            raise ValueError("norm must be provided when qkv_norm is True")
        check_norm(norm)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_norm = qkv_norm
        self.value_residual = value_residual
        self.is_first_layer = is_first_layer

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        self.q_proj = factory.dense(dim, use_bias=bias, name=sub("q_proj"))
        self.k_proj = factory.dense(dim, use_bias=bias, name=sub("k_proj"))
        self.v_proj = factory.dense(dim, use_bias=bias, name=sub("v_proj"))
        self.out_proj = factory.dense(dim, use_bias=bias, name=sub("out_proj"))
        self.value_residual_mix = None
        if value_residual and not is_first_layer:
            self.value_residual_mix = factory.dense(num_heads, activation="sigmoid", use_bias=True, name=sub("value_residual_mix"))
        self.q_norm = self.k_norm = self.v_norm = None
        if qkv_norm:
            self.q_norm = make_norm(norm, name=sub("q_norm"))
            self.k_norm = make_norm(norm, name=sub("k_norm"))
            self.v_norm = make_norm(norm, name=sub("v_norm"))

        self.scores_einsum = factory.einsum("bhqd,bhkd->bhqk", name=sub("scores"))
        self.attn_softmax = factory.softmax(axis=-1, name=sub("softmax"))
        self.values_einsum = factory.einsum("bhqk,bhkd->bhqd", name=sub("values"))

    def build(self, q_shape, k_shape=None, v_shape=None) -> None:
        k_shape = k_shape if k_shape is not None else q_shape
        v_shape = v_shape if v_shape is not None else k_shape
        for shape in (q_shape, k_shape, v_shape):
            if shape[-1] != self.dim:
                raise ValueError(f"{self.name}: feature size {shape[-1]} != dim {self.dim}")
        q_len, kv_len = q_shape[1], k_shape[1]
        heads, head_dim = self.num_heads, self.head_dim

        self.q_proj.build((None, q_len, self.dim))
        self.k_proj.build((None, kv_len, self.dim))
        self.v_proj.build((None, kv_len, self.dim))
        if self.value_residual_mix is not None:
            self.value_residual_mix.build((None, q_len, self.dim))
        if self.qkv_norm:
            self.q_norm.build((None, q_len, self.dim))
            self.k_norm.build((None, kv_len, self.dim))
            self.v_norm.build((None, kv_len, self.dim))
        self.scores_einsum.build([(None, heads, q_len, head_dim), (None, heads, kv_len, head_dim)])
        self.attn_softmax.build((None, heads, q_len, kv_len))
        self.values_einsum.build([(None, heads, q_len, kv_len), (None, heads, kv_len, head_dim)])
        self.out_proj.build((None, q_len, self.dim))

    def call(self, q, k=None, v=None, q_mask=None, kv_mask=None, attn_mask=None, initial_v=None, training=None):
        """Attention of ``q`` over ``k``/``v`` (self-attention when both are None).

        Returns:
            ``out`` (B, Sq, D). With ``value_residual=True`` it returns ``(out, v_cache)`` instead: the first
            layer returns its projected values (B, H, S, Dh) as the cache; a later layer mixes in
            ``initial_v`` and passes it on, or, when ``initial_v`` is None, does no mixing (as torch with
            ``initial_values=None``) and returns None.

        Raises:
            ValueError: If ``v`` is given without ``k``.
        """
        if k is None and v is None:
            k = v = q
        elif k is None:
            raise ValueError("k must be provided for cross-attention")
        elif v is None:
            v = k

        mix = None
        if self.value_residual_mix is not None:
            mix = self.value_residual_mix(q, training=training)  # (B, S, H), from the raw queries as in torch
            mix = keras.ops.expand_dims(keras.ops.transpose(mix, (0, 2, 1)), -1)  # (B, H, S, 1)

        qp = self.q_proj(q, training=training)
        kp = self.k_proj(k, training=training)
        vp = self.v_proj(v, training=training)
        if self.qkv_norm:
            qp = self.q_norm(qp)
            kp = self.k_norm(kp)
            vp = self.v_norm(vp)
        qp = separate_heads(qp, self.num_heads)
        kp = separate_heads(kp, self.num_heads)
        vp = separate_heads(vp, self.num_heads)

        v_cache = None
        if self.value_residual:
            if self.is_first_layer:
                v_cache = vp
            elif initial_v is not None:
                vp = vp * mix + initial_v * (1.0 - mix)
                v_cache = initial_v

        mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape)
        if mask is not None and len(mask.shape) == 3:
            mask = keras.ops.expand_dims(mask, -3)  # broadcast over heads

        qp = qp * (self.head_dim**-0.5)  # scale before the matmul, as torch SDPA does
        scores = self.scores_einsum([qp, kp], training=training)
        attn = apply_softmax(self.attn_softmax, scores, attn_mask=mask, training=training)
        out = self.values_einsum([attn, vp], training=training)
        out = self.out_proj(recombine_heads(out), training=training)
        if self.value_residual:
            return out, v_cache
        return out


def build_attention(dim: int, attn_type: str = "torch", *, factory: LayerFactory | None = None, name: str | None = None, **attn_kwargs):
    """Dispatch as ``hepattn.keras.attention.build_keras_attention``: linformer -> LinformerAttention, torch -> Attention.

    Any other attn_type raises ValueError in ``Attention``.
    """
    if attn_type == "linformer":
        from hepattn.keras_native.linformer import LinformerAttention  # noqa: PLC0415  (import cycle)

        seq_len = attn_kwargs.pop("linformer_seq_len", 256)
        k = attn_kwargs.pop("linformer_proj_dim", 256)
        return LinformerAttention(dim, seq_len=seq_len, k=k, factory=factory, name=name, **attn_kwargs)
    attn_kwargs.pop("linformer_seq_len", None)
    attn_kwargs.pop("linformer_proj_dim", None)
    return Attention(dim, attn_type=attn_type, factory=factory, name=name, **attn_kwargs)
