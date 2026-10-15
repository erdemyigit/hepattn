"""Keras-native twin of ``hepattn.keras.linformer.KerasLinformerAttention`` (Helen's LinformerAttention).

Same leaves and names as the hybrid (``{name}_to_q``, ``{name}_seqproj_k``, ``{name}_scores``, ...). The two
sequence projections ``proj_k`` / ``proj_v`` (seq_len, k) are float weights of this layer, initialised
U(-1/sqrt(k), 1/sqrt(k)) as in torch, and enter the quantized ``seqproj`` einsums as their second input.

Helen's semantics are kept: ``q_mask``, ``kv_mask``, ``qkv_norm``, ``value_residual`` and the other dense-attention
options are accepted and ignored. ``attn_mask`` is applied in the projected (rank-k) space:
``keep[..., :kv_len] = attn_mask``, ``keep[..., kv_len:] = False``, so masking needs ``k >= kv_len``. A fully
masked query row gets zero attention weights (output = ``to_out`` bias).
"""

import math

import keras

from hepattn.keras_native.factory import LayerFactory, apply_softmax


class LinformerAttention(keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        seq_len: int = 256,
        k: int = 256,
        bias: bool = True,
        attn_type: str = "linformer",
        factory: LayerFactory | None = None,
        name: str | None = None,
        **_ignored,
    ) -> None:
        """Args as the hybrid KerasLinformerAttention; other Attention kwargs are ignored (Helen's semantics).

        Raises:
            ValueError: If attn_type is not 'linformer' or num_heads does not divide dim.
        """
        super().__init__(name=name)
        factory = factory or LayerFactory()
        if attn_type != "linformer":
            raise ValueError(f"LinformerAttention only supports attn_type='linformer', got '{attn_type}'")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.k = k

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        # q/k/v projections are bias-free and the out projection always has a bias, whatever `bias` says
        # (as torch LinformerAttention and the hybrid)
        self.to_q = factory.dense(dim, use_bias=False, name=sub("to_q"))
        self.to_k = factory.dense(dim, use_bias=False, name=sub("to_k"))
        self.to_v = factory.dense(dim, use_bias=False, name=sub("to_v"))
        self.to_out = factory.dense(dim, use_bias=True, name=sub("to_out"))
        self.seq_proj_k = factory.einsum("bnd,nk->bkd", name=sub("seqproj_k"))
        self.seq_proj_v = factory.einsum("bnd,nk->bkd", name=sub("seqproj_v"))
        self.scores_einsum = factory.einsum("bhnd,bhkd->bhnk", name=sub("scores"))
        self.attn_softmax = factory.softmax(axis=-1, name=sub("softmax"))
        self.values_einsum = factory.einsum("bhnk,bhkd->bhnd", name=sub("values"))

    def build(self, q_shape, k_shape=None, v_shape=None) -> None:
        k_shape = k_shape if k_shape is not None else q_shape
        v_shape = v_shape if v_shape is not None else k_shape
        if tuple(k_shape[1:]) != tuple(v_shape[1:]):
            raise ValueError(f"{self.name}: k and v shapes differ: {k_shape} vs {v_shape}")
        n, kv_len = q_shape[1], k_shape[1]
        if kv_len > self.seq_len:
            raise ValueError(f"{self.name}: kv_len {kv_len} exceeds seq_len {self.seq_len}")
        heads, head_dim, rank = self.num_heads, self.head_dim, self.k

        self.to_q.build((None, n, self.dim))
        self.to_k.build((None, kv_len, self.dim))
        self.to_v.build((None, kv_len, self.dim))
        bound = 1.0 / math.sqrt(rank)
        self.proj_k = self.add_weight(name="proj_k", shape=(self.seq_len, rank), initializer=keras.initializers.RandomUniform(-bound, bound))
        self.proj_v = self.add_weight(name="proj_v", shape=(self.seq_len, rank), initializer=keras.initializers.RandomUniform(-bound, bound))
        self.seq_proj_k.build([(None, kv_len, self.dim), (kv_len, rank)])
        self.seq_proj_v.build([(None, kv_len, self.dim), (kv_len, rank)])
        self.scores_einsum.build([(None, heads, n, head_dim), (None, heads, rank, head_dim)])
        self.attn_softmax.build((None, heads, n, rank))
        self.values_einsum.build([(None, heads, n, rank), (None, heads, rank, head_dim)])
        self.to_out.build((None, n, self.dim))

    def call(self, q, k=None, v=None, q_mask=None, kv_mask=None, attn_mask=None, initial_v=None, training=None):
        """Low-rank attention; returns (B, N, D). ``q_mask``, ``kv_mask`` and ``initial_v`` are ignored.

        Raises:
            ValueError: If only one of ``k``/``v`` is given, or for masking with rank k < kv_len.
        """
        del q_mask, kv_mask, initial_v
        if k is None and v is None:
            k = v = q
        if k is None or v is None:
            raise ValueError("k and v must both be given for cross-attention")
        _, n, _ = q.shape
        kv_len = k.shape[1]

        queries = self.to_q(q, training=training)
        keys = self.to_k(k, training=training)
        values = self.to_v(v, training=training)

        # project keys/values along the sequence axis to rank k (proj sliced to kv_len, static)
        keys = self.seq_proj_k([keys, self.proj_k[:kv_len]], training=training)
        values = self.seq_proj_v([values, self.proj_v[:kv_len]], training=training)

        queries = keras.ops.transpose(keras.ops.reshape(queries, (-1, n, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        keys = keras.ops.transpose(keras.ops.reshape(keys, (-1, self.k, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        values = keras.ops.transpose(keras.ops.reshape(values, (-1, self.k, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        scores = self.scores_einsum([queries, keys], training=training) * (self.head_dim**-0.5)

        keep = None
        if attn_mask is not None:
            if self.k < kv_len:
                raise ValueError(f"linformer projection rank k={self.k} must be >= kv_len={kv_len} for masking")
            keep = keras.ops.expand_dims(keras.ops.cast(attn_mask, "bool"), 1)  # (B, 1, N, kv_len)
            keep = keras.ops.pad(keep, ((0, 0), (0, 0), (0, 0), (0, self.k - kv_len)), constant_values=False)

        attn = apply_softmax(self.attn_softmax, scores, attn_mask=keep, training=training)
        out = self.values_einsum([attn, values], training=training)
        out = keras.ops.reshape(keras.ops.transpose(out, (0, 2, 1, 3)), (-1, n, self.dim))
        return self.to_out(out, training=training)
