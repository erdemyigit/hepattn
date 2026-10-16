"""Keras-native twins of ``hepattn.models.encoder.{LayerScale, Residual, EncoderLayer, Encoder}``.

Only the dense-mask path exists (as in the hybrid ``hepattn.keras.encoder``): sliding windows, score mods,
flash-varlen unpadding and DropPath raise. ``flash``/``flash-varlen`` are coerced to ``torch`` with a warning
(identical math on the dense-mask path); ``linformer`` goes to ``attention.build_attention``.

Input sorting (org ``encoder.py`` ``sort_index`` and ``Encoder.forward``): with ``x_sort_value`` the encoder sorts the
tokens and the padding mask (padding last), prepends the registers, runs the layers, strips the registers and
unsorts. ``keras.ops.argsort`` is stable and torch's is not, so the two order tied values (the ``+inf`` padding)
differently; valid outputs do not depend on that order.

Names: attention and FFN leaves as in the hybrid (``encoder_l{i}_attn_*``, ``encoder_l{i}_ffn_*``); the parts that
are torch modules in the hybrid are named ``encoder_l{i}_attn_norm`` / ``_ffn_norm`` (Residual norms),
``encoder_l{i}_attn_ls`` / ``_ffn_ls`` (LayerScale), and the weight ``register_tokens`` of the encoder layer.
"""

import warnings

import keras

from hepattn.keras_native.attention import build_attention
from hepattn.keras_native.dense import Dense
from hepattn.keras_native.factory import LayerFactory
from hepattn.keras_native.norm import check_norm, get_hybrid_norm_config, make_norm

ALLOWED_ATTN_TYPES = {"torch", "flash", "flash-varlen", "linformer"}
COERCE_TO_TORCH = {"flash", "flash-varlen"}


def sort_index(value, valid=None):
    """Argsort of ``value`` (B, N) along the last axis with the padding (``valid`` False) last (org ``sort_index``)."""
    if valid is not None:
        value = keras.ops.where(valid, value, float("inf"))
    return keras.ops.argsort(value, axis=-1)


def gather_tokens(x, index):
    """``x[b, index[b, n]]`` for (B, N) or (B, N, D) ``x`` and a (B, N) ``index``."""
    if len(x.shape) == 2:
        return keras.ops.take_along_axis(x, index, axis=-1)
    index = keras.ops.broadcast_to(keras.ops.expand_dims(index, -1), keras.ops.shape(x))
    return keras.ops.take_along_axis(x, index, axis=-2)


class LayerScale(keras.layers.Layer):
    """x * gamma, gamma initialised to ``init_value`` (https://arxiv.org/abs/2103.17239)."""

    def __init__(self, dim: int, init_value: float = 1e-5, name: str | None = None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.init_value = init_value

    def build(self, input_shape) -> None:
        self.gamma = self.add_weight(name="gamma", shape=(self.dim,), initializer=keras.initializers.Constant(self.init_value))

    def call(self, x):
        return x * self.gamma


class Residual(keras.layers.Layer):
    """x + ls(fn(norm(x))), or with ``post_norm``: x = norm(x); x + ls(fn(x)) (HybridNorm, 2503.04598).

    If ``fn`` returns a tuple ``(out, aux)`` (attention with a value residual), this returns ``(x + ls(out), aux)``.
    """

    def __init__(
        self,
        fn: keras.layers.Layer,
        dim: int,
        norm: str | None,
        post_norm: bool = False,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        name: str | None = None,
        prefix: str | None = None,
        **kwargs,
    ) -> None:
        """Args as hepattn Residual; ``prefix`` (default: ``name``) names the norm ``{prefix}_norm`` and LayerScale ``{prefix}_ls``.

        Raises:
            ValueError: For post_norm without norm or an unknown norm.
            NotImplementedError: For drop_path > 0 (it would need keras.random.SeedGenerator state).
        """
        super().__init__(name=name, **kwargs)
        if post_norm and not norm:
            raise ValueError("post_norm is True but no norm is provided.")
        check_norm(norm)
        if drop_path:
            raise NotImplementedError("drop_path is not supported by keras_native (CLIC uses drop_path=0)")

        prefix = prefix or name

        def sub(part: str) -> str | None:
            return f"{prefix}_{part}" if prefix else None

        self.fn = fn
        self.dim = dim
        self.post_norm = post_norm
        self.norm = make_norm(norm, name=sub("norm")) if norm else None
        self.ls = LayerScale(dim, layer_scale, name=sub("ls")) if layer_scale is not None else None

    def build(self, x_shape) -> None:
        # fn is built by its owner (it may need more than x's shape, e.g. cross-attention keys) or at its first call
        if self.norm is not None:
            self.norm.build(x_shape)
        if self.ls is not None:
            self.ls.build(x_shape)

    def call(self, x, training=None, **kwargs):
        if self.norm is not None and self.post_norm:
            x = self.norm(x)
            h = self.fn(x, training=training, **kwargs)
        else:
            h = self.fn(self.norm(x) if self.norm is not None else x, training=training, **kwargs)
        aux = None
        is_tuple = isinstance(h, tuple)
        if is_tuple:
            h, aux = h
        if self.ls is not None:
            h = self.ls(h)
        out = x + h
        return (out, aux) if is_tuple else out


class EncoderLayer(keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        depth: int = 0,
        norm: str = "LayerNorm",
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        value_residual: bool = False,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        dense_norm_placement: str = "hybridnorm",
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        attn_type: str = "torch",
        factory: LayerFactory | None = None,
        name: str = "encoder_layer",
        **kwargs,
    ) -> None:
        """Residual self-attention + residual feed-forward, as hepattn EncoderLayer (``dense_norm_placement``: see
        ``norm.get_hybrid_norm_config``).
        """
        super().__init__(name=name, **kwargs)
        factory = factory or LayerFactory()
        attn_kwargs = dict(attn_kwargs or {})
        dense_kwargs = dict(dense_kwargs or {})

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm, dense_norm_placement)
        attn_kwargs["value_residual"] = value_residual
        attn_kwargs["is_first_layer"] = depth == 0

        self.dim = dim
        attn = build_attention(dim, attn_type=attn_type, qkv_norm=qkv_norm, norm=norm, factory=factory, name=f"{name}_attn", **attn_kwargs)
        ffn = Dense(dim, factory=factory, name=f"{name}_ffn", **dense_kwargs)
        common = {"dim": dim, "layer_scale": layer_scale, "drop_path": drop_path}
        self.attn = Residual(attn, norm=attn_norm, name=f"{name}_attn_res", prefix=f"{name}_attn", **common)
        self.dense = Residual(ffn, norm=norm, post_norm=dense_post_norm, name=f"{name}_ffn_res", prefix=f"{name}_ffn", **common)

    def build(self, x_shape) -> None:
        shape = (None, *x_shape[1:])
        for res in (self.attn, self.dense):
            res.build(shape)
            res.fn.build(shape)

    def call(self, x, kv_mask=None, attn_mask=None, initial_v=None, training=None):
        """Returns ``(x, initial_v)``; ``initial_v`` is the value-residual cache (None without value residual)."""
        h = self.attn(x, kv_mask=kv_mask, attn_mask=attn_mask, initial_v=initial_v, training=training)
        if isinstance(h, tuple):
            h, initial_v = h
        return self.dense(h, training=training), initial_v


class Encoder(keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        attn_type: str = "torch",
        window_size: int | None = None,
        window_wrap: bool = False,
        score_mod: str | None = None,
        value_residual: bool = False,
        num_register_tokens: int | None = None,
        factory: LayerFactory | None = None,
        name: str = "encoder",
        **layer_kwargs,
    ) -> None:
        """Hepattn Encoder for the dense-mask path; layers are named ``encoder_l{i}`` (as in the hybrid).

        Raises:
            ValueError: For windowed attention, score mods, or an attn_type with no keras_native counterpart.
        """
        super().__init__(name=name)
        factory = factory or LayerFactory()
        if attn_type not in ALLOWED_ATTN_TYPES:
            raise ValueError(f"keras_native Encoder cannot express attn_type='{attn_type}'")
        resolved = "torch" if attn_type in COERCE_TO_TORCH else attn_type
        if attn_type in COERCE_TO_TORCH:
            warnings.warn(f"keras_native Encoder coerces attn_type='{attn_type}' to 'torch' (identical math on the dense-mask path)", stacklevel=2)
        if window_size is not None or window_wrap:
            raise ValueError("windowed attention is not supported by the keras_native Encoder")
        if score_mod is not None:
            raise ValueError("score mods are not supported by the keras_native Encoder")

        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = resolved
        self.value_residual = value_residual
        self.num_register_tokens = num_register_tokens

        layer_kwargs = dict(layer_kwargs)
        attn_kwargs = dict(layer_kwargs.pop("attn_kwargs", None) or {})
        attn_kwargs.pop("attn_type", None)
        attn_kwargs.pop("window_size", None)
        layer_kwargs["value_residual"] = value_residual
        layer_kwargs["attn_kwargs"] = attn_kwargs
        layer_kwargs["attn_type"] = resolved
        self.layers = [EncoderLayer(dim=dim, depth=i, factory=factory, name=f"encoder_l{i}", **layer_kwargs) for i in range(num_layers)]

    def build(self, x_shape) -> None:
        if x_shape[-1] != self.dim:
            raise ValueError(f"{self.name}: feature size {x_shape[-1]} != dim {self.dim}")
        seq = x_shape[1] + (self.num_register_tokens or 0)
        if self.num_register_tokens is not None:
            # torch: nn.Parameter(torch.randn(1, R, dim))
            self.register_tokens = self.add_weight(
                name="register_tokens", shape=(1, self.num_register_tokens, self.dim), initializer=keras.initializers.RandomNormal(stddev=1.0)
            )
        for layer in self.layers:
            layer.build((None, seq, self.dim))

    def call(self, x, kv_mask=None, x_sort_value=None, training=None):
        """Encode ``x`` (B, N, D); with ``x_sort_value`` (B, N) the layers see the tokens sorted by it, padding last.

        The output is in the input order.
        """
        sort_idx = None
        if x_sort_value is not None:
            sort_idx = sort_index(x_sort_value, kv_mask)
            x = gather_tokens(x, sort_idx)
            if kv_mask is not None:
                kv_mask = gather_tokens(kv_mask, sort_idx)
        num_reg = self.num_register_tokens
        if num_reg is not None:
            batch = keras.ops.shape(x)[0]
            tokens = keras.ops.broadcast_to(keras.ops.cast(self.register_tokens, x.dtype), (batch, num_reg, self.dim))
            x = keras.ops.concatenate([tokens, x], axis=1)
            if kv_mask is not None:
                kv_mask = keras.ops.concatenate([keras.ops.ones((batch, num_reg), dtype=kv_mask.dtype), kv_mask], axis=1)

        initial_v = None
        for layer in self.layers:
            x, initial_v = layer(x, kv_mask=kv_mask, initial_v=initial_v, training=training)

        if num_reg is not None:
            x = x[:, num_reg:]
        if sort_idx is not None:
            x = gather_tokens(x, keras.ops.argsort(sort_idx, axis=-1))
        return x
