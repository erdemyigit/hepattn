"""Keras-native multi-head attention (org/main ``hepattn.models.attention.Attention``) and ``merge_masks``.

``attn_type="torch"`` mirrors ``hepattn.keras.attention.KerasAttention`` op for op: three factory-built projections,
an explicit scores einsum -> masked softmax -> values einsum core, and the output projection. Leaf names are identical
(``{name}_q_proj``, ``{name}_scores``, ...). ``attn_type="linformer"`` keeps the projections, ``qkv_norm`` and the
value residual and replaces the core by the masked-Linformer core of ``hepattn.keras_native.masked_linformer``
(org ``Attention._linformer_attention``). Differences to the hybrid:
- QKV norms are keras ``LayerNormalization(epsilon=1e-5)`` (``{name}_q_norm`` ...), not torch modules.
- The value residual is returned instead of written into a dict: with ``value_residual=True`` ``call``
  returns ``(out, v_cache)`` (see ``Attention.call``).
Masks follow the hepattn convention: True = takes part in attention.
"""

import inspect
import math

import keras

from hepattn.keras_native.factory import LayerFactory, apply_softmax
from hepattn.keras_native.masked_linformer import AttendCore, ProjectCore
from hepattn.keras_native.norm import check_norm, make_norm

LINFORMER_CORES = ("auto", "attend", "project", "project_unmasked")


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
        linformer_seq_len: int | None = None,
        linformer_k: int | None = None,
        linformer_core: str = "auto",
        factory: LayerFactory | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Multi-head attention from factory-built (quantizable) leaves; arguments as org/main hepattn Attention.

        ``linformer_seq_len`` / ``linformer_k`` (linformer only, both required) size the sequence projections.
        ``linformer_core`` (keras_native only) fixes which org path this layer runs, because the graph is static:
        ``"attend"`` (called with an ``attn_mask``), ``"project"`` (no ``attn_mask``; ``kv_mask`` optional),
        ``"project_unmasked"`` (no mask at all: no renorm leaves and no rows of E/F past the key length), or
        ``"auto"``: ``"project"`` for a self-attention build (``k_shape`` None or the same object as ``q_shape``,
        as org's ``q is k`` test), else ``"attend"``. A call that does not match the core raises ValueError.

        q/k/v kernels start as org's xavier-uniform on the packed (3 dim, dim) in-projection, U(+-sqrt(6 / (4 dim))),
        with zero bias. A hepattn norm that keras_native does not implement raises NotImplementedError (``check_norm``).

        Raises:
            ValueError: For an attn_type other than torch/linformer, torch_compile/window_size, qkv_norm without norm,
                or linformer without ``linformer_seq_len`` and ``linformer_k``.
        """
        super().__init__(name=name, **kwargs)
        factory = factory or LayerFactory()

        if dim % num_heads != 0:
            raise ValueError("num_heads must divide dim.")
        if attn_type not in {"torch", "linformer"}:
            raise ValueError(f"Attention implements attn_type 'torch' and 'linformer' only, got attn_type='{attn_type}'")
        if torch_compile or window_size is not None:
            raise ValueError("torch_compile / window_size are not supported by keras_native Attention")
        if qkv_norm and not norm:
            raise ValueError("norm must be provided when qkv_norm is True")
        if attn_type == "linformer" and (linformer_seq_len is None or linformer_k is None):
            raise ValueError("linformer attention needs linformer_seq_len and linformer_k")
        if linformer_core not in LINFORMER_CORES:
            raise ValueError(f"linformer_core must be one of {LINFORMER_CORES}, got '{linformer_core}'")
        check_norm(norm)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_type = attn_type
        self.qkv_norm = qkv_norm
        self.value_residual = value_residual
        self.is_first_layer = is_first_layer
        self.linformer_seq_len = linformer_seq_len
        self.linformer_k = linformer_k
        self.linformer_core = linformer_core
        self.core = None
        self.core_kind = None
        self._factory = factory

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        def in_proj(part: str):
            bound = math.sqrt(6.0 / (4 * dim))  # xavier on the packed (3 dim, dim) matrix: fan_in dim, fan_out 3 dim
            return factory.dense(dim, use_bias=bias, name=sub(part), kernel_initializer=keras.initializers.RandomUniform(-bound, bound))

        self.q_proj = in_proj("q_proj")
        self.k_proj = in_proj("k_proj")
        self.v_proj = in_proj("v_proj")
        self.out_proj = factory.dense(dim, use_bias=bias, name=sub("out_proj"))
        self.value_residual_mix = None
        if value_residual and not is_first_layer:
            self.value_residual_mix = factory.dense(num_heads, activation="sigmoid", use_bias=True, name=sub("value_residual_mix"))
        self.q_norm = self.k_norm = self.v_norm = None
        if qkv_norm:
            self.q_norm = make_norm(norm, name=sub("q_norm"))
            self.k_norm = make_norm(norm, name=sub("k_norm"))
            self.v_norm = make_norm(norm, name=sub("v_norm"))

        if attn_type == "torch":
            self.scores_einsum = factory.einsum("bhqd,bhkd->bhqk", name=sub("scores"))
            self.attn_softmax = factory.softmax(axis=-1, name=sub("softmax"))
            self.values_einsum = factory.einsum("bhqk,bhkd->bhqd", name=sub("values"))

    def build(self, q_shape, k_shape=None, v_shape=None) -> None:
        self_attention = k_shape is None or k_shape is q_shape
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
        if self.attn_type == "linformer":
            self._build_linformer_core(q_len, kv_len, self_attention)
        else:
            self.scores_einsum.build([(None, heads, q_len, head_dim), (None, heads, kv_len, head_dim)])
            self.attn_softmax.build((None, heads, q_len, kv_len))
            self.values_einsum.build([(None, heads, q_len, kv_len), (None, heads, kv_len, head_dim)])
        self.out_proj.build((None, q_len, self.dim))

    def _build_linformer_core(self, q_len: int, kv_len: int, self_attention: bool) -> None:
        """The core has static shapes (q_len, kv_len), so it is created here; its leaves under the factory scopes.

        Raises:
            ValueError: If the key length exceeds ``linformer_seq_len``.
        """
        if kv_len > self.linformer_seq_len:
            raise ValueError(f"{self.name}: key/value length {kv_len} exceeds linformer_seq_len {self.linformer_seq_len}")
        kind = self.linformer_core
        if kind == "auto":
            kind = "project" if self_attention else "attend"
        dims = (self.num_heads, self.head_dim, q_len, kv_len, self.linformer_seq_len, self.linformer_k)
        name = f"{self.name}_{kind}"
        with self._factory.scopes():
            if kind == "attend":
                core = AttendCore(self._factory, *dims, name=name)
            else:
                core = ProjectCore(self._factory, *dims, masked=kind == "project", name=name)
        core.build()
        self.core = core
        self.core_kind = kind
        self.kv_len = kv_len

    def call(self, q, k=None, v=None, q_mask=None, kv_mask=None, attn_mask=None, initial_v=None, kv_sort_idx=None, training=None):
        """Attention of ``q`` over ``k``/``v`` (self-attention when both are None).

        ``kv_sort_idx`` (B, N): with ``attn_type="linformer"`` the keys, values, ``kv_mask`` and the key axis of
        ``attn_mask`` enter the sequence projection in this order (the queries keep theirs), as org. Dense attention
        is permutation-equivariant and ignores it, as org. Linformer ignores ``q_mask``, as org.

        Returns:
            ``out`` (B, Sq, D). With ``value_residual=True`` it returns ``(out, v_cache)`` instead: the first
            layer returns its projected values (B, H, S, Dh) as the cache; a later layer mixes in
            ``initial_v`` and passes it on, or, when ``initial_v`` is None, does no mixing (as torch with
            ``initial_values=None``) and returns None.

        Raises:
            ValueError: If ``v`` is given without ``k``, or a linformer call does not match the built core.
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

        if self.attn_type == "linformer":
            out = self._linformer(qp, kp, vp, kv_mask, attn_mask, kv_sort_idx, training)
        else:
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

    def _linformer(self, q, k, v, kv_mask, attn_mask, kv_sort_idx, training):
        """Org ``Attention._linformer_attention`` on (B, H, S, Dh) heads.

        Raises:
            ValueError: If the key length differs from the built one, or the masks do not match the built core.
        """
        if k.shape[2] != self.kv_len:
            raise ValueError(f"{self.name}: key length {k.shape[2]} != the built length {self.kv_len}; build with (q_shape, k_shape) first")
        if kv_sort_idx is not None:
            idx = keras.ops.cast(kv_sort_idx, "int32")
            gather = keras.ops.broadcast_to(idx[:, None, :, None], keras.ops.shape(k))
            k = keras.ops.take_along_axis(k, gather, axis=2)
            v = keras.ops.take_along_axis(v, gather, axis=2)
            if kv_mask is not None:
                kv_mask = keras.ops.take_along_axis(kv_mask, idx, axis=1)
            if attn_mask is not None:
                attn_mask = keras.ops.take_along_axis(attn_mask, keras.ops.broadcast_to(idx[:, None, :], keras.ops.shape(attn_mask)), axis=2)
        if self.core_kind == "attend":
            if attn_mask is None:
                raise ValueError(f"{self.name}: the attend core needs an attn_mask; build with linformer_core='project' for calls without one")
            if kv_mask is not None:
                attn_mask = keras.ops.logical_and(attn_mask, kv_mask[:, None])
            return self.core(q, k, v, attn_mask, training=training)
        if attn_mask is not None:
            raise ValueError(f"{self.name}: the {self.core_kind} core takes no attn_mask; build with linformer_core='attend'")
        return self.core(q, k, v, kv_mask, training=training)


ATTENTION_KEYS = frozenset(inspect.signature(Attention.__init__).parameters) - {"self", "dim", "attn_type", "factory", "name", "kwargs"}


def build_attention(dim: int, attn_type: str = "torch", *, factory: LayerFactory | None = None, name: str | None = None, **attn_kwargs):
    """Native ``Attention`` for any supported attn_type (``torch``, ``linformer``); other types raise in ``Attention``.

    Only the org/main ``Attention`` options (plus ``linformer_core``) are accepted: an unknown key raises instead of
    being dropped, so a stale config cannot silently build another model (``linformer_proj_dim`` used to give k=256).

    Raises:
        TypeError: For an attention option that org/main ``Attention`` does not have (as org's
            ``Attention(**kwargs)`` does).
    """
    if "linformer_proj_dim" in attn_kwargs:
        raise TypeError("linformer_proj_dim is not an org/main Attention option; the projected length is linformer_k (org PR #9)")
    unknown = sorted(set(attn_kwargs) - ATTENTION_KEYS)
    if unknown:
        raise TypeError(f"unknown attention option(s) {unknown}; accepted: {sorted(ATTENTION_KEYS)}")
    return Attention(dim, attn_type=attn_type, factory=factory, name=name, **attn_kwargs)
