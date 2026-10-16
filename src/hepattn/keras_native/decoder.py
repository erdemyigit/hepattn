"""Keras-native twins of ``hepattn.models.decoder.{MaskFormerDecoderLayer, MaskFormerDecoder}`` (mask attention).

Static shapes and one constituent type: the merged keys are the single input's constituents, so the torch
``key_is_<input>`` scatter of the task attention masks and the ``unmerge_inputs`` gather are identities.

Names as in the hybrid: layers ``decoder_l{i}``, attention leaves ``decoder_l{i}_{q_ca,q_sa,kv_ca}_*``, FFN leaves
``decoder_l{i}_{q,kv}_ffn_*``. Parts that are torch modules in the hybrid: Residual containers
``decoder_l{i}_{q_ca,q_sa,q_ffn,kv_ca,kv_ffn}_res`` with norms ``..._norm``, and the weight ``initial_queries`` of the
decoder layer ``decoder``.

Unsupported decoder options raise: dynamic queries, kmeans cross-attention, local strided / flex attention,
decoder positional encodings, unified decoding, ``use_query_masks``, more than one input type.
"""

import warnings

import keras

from hepattn.keras_native.attention import build_attention
from hepattn.keras_native.dense import Dense
from hepattn.keras_native.encoder import COERCE_TO_TORCH, Residual
from hepattn.keras_native.factory import LayerFactory
from hepattn.keras_native.norm import get_hybrid_norm_config

DECODER_ATTN_TYPES = {"torch", "flash", "flash-varlen", "linformer"}
QUERY_UPDATE_ORDERS = ("ca_dense_sa", "ca_sa_dense")


class MaskFormerDecoderLayer(keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        bidirectional_ca: bool = True,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        dense_norm_placement: str = "hybridnorm",
        query_update_order: str = "ca_dense_sa",
        self_attn_norm_kv: bool = False,
        cross_attn_mode: str = "softmax",
        kmeans_kwargs: dict | None = None,
        factory: LayerFactory | None = None,
        name: str = "decoder_layer",
        **kwargs,
    ) -> None:
        """Query cross-attention, then FFN and query self-attention in ``query_update_order``, then (bidirectional)
        key cross-attention and FFN. Options as org ``MaskFormerDecoderLayer``: ``dense_norm_placement`` (see
        ``norm.get_hybrid_norm_config``), ``query_update_order`` ("ca_dense_sa" or "ca_sa_dense"), and
        ``self_attn_norm_kv`` (the query self-attention takes its keys and values from its normalised input too).

        Raises:
            NotImplementedError: For ``cross_attn_mode`` other than "softmax".
            ValueError: For an attention type without a keras_native counterpart (e.g. flex), a window size, or an
                unknown ``query_update_order`` / ``dense_norm_placement``.
        """
        super().__init__(name=name, **kwargs)
        if query_update_order not in QUERY_UPDATE_ORDERS:
            raise ValueError(f"Unsupported query_update_order: {query_update_order}. Must be 'ca_dense_sa' or 'ca_sa_dense'.")
        if cross_attn_mode != "softmax":
            raise NotImplementedError(f"cross_attn_mode='{cross_attn_mode}' is not ported to keras_native (softmax only)")
        if kmeans_kwargs:
            raise NotImplementedError("kmeans_kwargs need cross_attn_mode='kmeans', which is not ported")
        factory = factory or LayerFactory()

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm, dense_norm_placement)
        attn_kwargs = dict(attn_kwargs or {})
        attn_type = attn_kwargs.pop("attn_type", "torch")
        if attn_type not in DECODER_ATTN_TYPES:
            raise ValueError(f"keras_native decoder cannot express attn_type='{attn_type}'")
        if attn_type in COERCE_TO_TORCH:
            warnings.warn(f"keras_native decoder coerces attn_type='{attn_type}' to 'torch' (identical math on the dense-mask path)", stacklevel=2)
            attn_type = "torch"
        if attn_kwargs.pop("window_size", None) is not None:
            raise ValueError("windowed attention is not supported by the keras_native decoder")
        dense_kwargs = dict(dense_kwargs or {})

        self.dim = dim
        self.bidirectional_ca = bidirectional_ca
        self.attn_type = attn_type
        self.query_update_order = query_update_order
        self.self_attn_norm_kv = self_attn_norm_kv

        def attn(part: str):
            return build_attention(dim, attn_type=attn_type, qkv_norm=qkv_norm, norm=norm, factory=factory, name=f"{name}_{part}", **attn_kwargs)

        def residual(fn, part: str, res_norm, post_norm: bool = False):
            return Residual(fn, dim=dim, norm=res_norm, post_norm=post_norm, name=f"{name}_{part}_res", prefix=f"{name}_{part}")

        def ffn(part: str):
            return Dense(dim, factory=factory, name=f"{name}_{part}", **dense_kwargs)

        self.q_ca = residual(attn("q_ca"), "q_ca", attn_norm)
        self.q_sa = residual(attn("q_sa"), "q_sa", attn_norm)
        self.q_dense = residual(ffn("q_ffn"), "q_ffn", norm, dense_post_norm)
        self.kv_ca = self.kv_dense = None
        if bidirectional_ca:
            self.kv_ca = residual(attn("kv_ca"), "kv_ca", attn_norm)
            self.kv_dense = residual(ffn("kv_ffn"), "kv_ffn", norm, dense_post_norm)

    def build(self, q_shape, kv_shape) -> None:
        q = (None, *q_shape[1:])
        kv = (None, *kv_shape[1:])
        # Residual.build builds only the norm and LayerScale; the attention needs the key shape too
        for res, x_shape, fn_shapes in (
            (self.q_ca, q, (q, kv, kv)),
            (self.q_dense, q, (q,)),
            (self.q_sa, q, (q, q, q)),
            (self.kv_ca, kv, (kv, q, q)),
            (self.kv_dense, kv, (kv,)),
        ):
            if res is not None:
                res.build(x_shape)
                res.fn.build(*fn_shapes)

    def call(self, q, kv, attn_mask=None, q_mask=None, kv_mask=None, kv_sort_idx=None, training=None):
        """Returns the updated ``(q, kv)``; ``attn_mask`` is (B, Q, N), True = attend.

        ``kv_sort_idx`` (B, N), the key order of the encoder sort, goes to the query cross-attention only (org
        ``decoder.py``: the Linformer branch of ``q_ca`` reorders its keys with it).
        """
        q = self.q_ca(q, k=kv, v=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask, kv_sort_idx=kv_sort_idx, training=training)
        if self.query_update_order == "ca_sa_dense":
            q = self._query_self_attn(q, q_mask, training)
            q = self.q_dense(q, training=training)
        else:
            q = self.q_dense(q, training=training)
            q = self._query_self_attn(q, q_mask, training)
        if self.bidirectional_ca:
            mask_t = keras.ops.transpose(attn_mask, (0, 2, 1)) if attn_mask is not None else None
            kv = self.kv_ca(kv, k=q, v=q, attn_mask=mask_t, q_mask=kv_mask, kv_mask=q_mask, training=training)
            kv = self.kv_dense(kv, training=training)
        return q, kv

    def _query_self_attn(self, q, q_mask, training):
        # without k and v the attention takes all three from its (normalised) input
        if self.self_attn_norm_kv:
            return self.q_sa(q, q_mask=q_mask, training=training)
        return self.q_sa(q, k=q, v=q, q_mask=q_mask, training=training)


def restore_all_false_rows(attn_mask):
    """A query whose mask row is all False attends everywhere (torch ``unmask_all_false``)."""
    all_false = keras.ops.all(keras.ops.logical_not(attn_mask), axis=-1, keepdims=True)
    return keras.ops.where(all_false, True, attn_mask)


class TaskRef:
    """Holds the tasks outside keras tracking (keras tracks layers, variables and containers, not this object)."""

    def __init__(self, tasks=()) -> None:
        self.tasks = tuple(tasks)


class MaskFormerDecoder(keras.layers.Layer):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        posenc: dict | None = None,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
        fast_local_ca: bool = False,
        block_size: int = 128,
        unified_decoding: bool = False,
        phi_shift: float = 0.0,
        unmask_all_false: bool = True,
        dynamic_queries: bool = False,
        dynamic_query_source: str | None = None,
        factory: LayerFactory | None = None,
        name: str = "decoder",
        **kwargs,
    ) -> None:
        """Args as ``hepattn.models.decoder.MaskFormerDecoder``. ``window_size``, ``window_wrap``, ``block_size``,
        ``phi_shift`` and ``dynamic_query_source`` only act with options that raise here, so they are ignored.

        The tasks are set by the owning model with ``set_tasks`` (as torch ``decoder.tasks = tasks``); they are not
        tracked by this layer, so their variables belong to the model only.

        Raises:
            NotImplementedError: For dynamic queries, local strided attention (incl. fast_local_ca), decoder posenc,
                unified decoding, or use_query_masks.
        """
        super().__init__(name=name, **kwargs)
        for flag, what in (
            (dynamic_queries, "dynamic_queries"),
            (local_strided_attn, "local_strided_attn"),
            (fast_local_ca, "fast_local_ca"),
            (posenc is not None, "decoder posenc"),
            (unified_decoding, "unified_decoding"),
            (use_query_masks, "use_query_masks"),
        ):
            if flag:
                raise NotImplementedError(f"{what} is not ported to keras_native")
        factory = factory or LayerFactory()
        self.num_queries = num_queries
        self.dim = decoder_layer_config["dim"]
        self.mask_attention = mask_attention
        self.unmask_all_false = unmask_all_false
        cfg = dict(decoder_layer_config)
        self.layers = [MaskFormerDecoderLayer(depth=i, factory=factory, name=f"decoder_l{i}", **cfg) for i in range(num_decoder_layers)]
        self._task_ref = TaskRef()

    def set_tasks(self, tasks) -> None:
        self._task_ref = TaskRef(tasks)

    @property
    def tasks(self) -> tuple:
        return self._task_ref.tasks

    def build(self, key_shape) -> None:
        # torch: nn.Parameter(torch.randn(num_queries, dim))
        self.initial_queries = self.add_weight(
            name="initial_queries", shape=(self.num_queries, self.dim), initializer=keras.initializers.RandomNormal(stddev=1.0)
        )
        for layer in self.layers:
            layer.build((None, self.num_queries, self.dim), (None, *key_shape[1:]))

    def call(self, x: dict, input_name: str, attn_masks: dict | None = None, training=None):
        """Run the decoder layers with the tasks in between. Returns ``(x, outputs)``.

        ``x`` needs ``key_embed``, ``key_valid`` and ``{input_name}_valid`` (and ``key_sort_idx`` when the inputs are
        sorted); ``outputs`` is
        ``{"layer_{i}": {task_name: {...}, "attn_mask": (B, Q, N) bool}}``. ``attn_masks`` (``{"layer_{i}": mask}``)
        replaces the computed mask of those layers (teacher forcing: a parity check that is not amplified by
        threshold flips in the mask feedback); None in normal use.

        Raises:
            ValueError: If a task returns an attention mask for an input other than ``input_name``.
        """
        x = dict(x)
        key = x["key_embed"]
        batch = keras.ops.shape(key)[0]
        x["query_embed"] = keras.ops.broadcast_to(keras.ops.cast(self.initial_queries, key.dtype)[None], (batch, self.num_queries, self.dim))

        outputs = {}
        attn_mask = None
        for index, layer in enumerate(self.layers):
            layer_out = {}
            masks = {}
            for task in self.tasks:
                if not task.should_run_at_layer(index):
                    continue
                task_out = task(x, outputs=layer_out, training=training)
                layer_out[task.task_name] = task_out
                for name, mask in task.attn_mask(task_out).items():
                    masks[name] = keras.ops.logical_or(masks[name], mask) if name in masks else mask

            if masks and self.mask_attention:
                if set(masks) != {input_name}:
                    raise ValueError(f"attention masks for {sorted(masks)}, but the only input is '{input_name}'")
                # one input type: the torch key_is_<input> scatter is the identity; the mask is already detached
                attn_mask = masks[input_name]
                if self.unmask_all_false:
                    attn_mask = restore_all_false_rows(attn_mask)
            if attn_masks is not None and f"layer_{index}" in attn_masks:
                attn_mask = attn_masks[f"layer_{index}"]
            if attn_mask is not None:
                layer_out["attn_mask"] = attn_mask
            outputs[f"layer_{index}"] = layer_out

            x["query_embed"], x["key_embed"] = layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=attn_mask,
                q_mask=None,
                kv_mask=x["key_valid"],
                kv_sort_idx=x.get("key_sort_idx"),
                training=training,
            )
            x[f"{input_name}_embed"] = x["key_embed"]  # unmerge_inputs, identity for one input
        return x, outputs
