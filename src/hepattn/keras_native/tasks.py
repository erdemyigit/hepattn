"""Keras-native forwards of the four CLIC tasks (``hepattn.models.task``): classification, hit mask, incidence,
incidence-based regression.

Only the forward (and ``attn_mask`` / ``should_run_at_layer``) is ported here; costs and losses are S4. Each task is a
``keras.layers.Layer`` named ``task{i}`` (``layer_name``); the torch task name (``classification``, ``mask``, ...) is
``task_name`` and keys the output dicts. Leaves as in the hybrid: ``task{i}_net_*``, ``task{i}_object_net_*``,
``task{i}_node_net_*``, ``task{i}_mask_einsum``, ``task{i}_incidence_einsum``, ``task{i}_incidence_softmax``.

Differences to the hybrid:
- The mask and incidence einsum / softmax go through ``LayerFactory`` in both modes (the hybrid's float model uses
  torch ``einsum`` / ``softmax``; its quantized model swaps the task class). ``EinsumOp`` / ``SoftmaxOp`` compute the
  same thing.
- Task-head ``Dense`` layers default to ``init="torch"``: the hybrid ports them from torch ``Dense`` modules
  (``kerasify_module``), so they start from ``nn.Linear``'s init.
- ``.detach()`` is ``keras.ops.stop_gradient``; in-place column writes are rebuilt with ``concatenate``;
  ``topk_attn`` is ``top_k`` + a ``one_hot`` sum.
"""

import keras
import ml_dtypes

from hepattn.keras_native.dense import Dense
from hepattn.keras_native.factory import LayerFactory, apply_softmax
from hepattn.keras_native.scaling import FeatureScaler


def dense_kwargs(spec: dict | None) -> dict | None:
    """``Dense`` kwargs from a YAML spec ``{"class_path": "...Dense", "init_args": {...}}`` or a plain kwargs dict.

    Raises:
        NotImplementedError: For a class other than hepattn ``Dense``.
        ValueError: For keys next to ``class_path`` other than ``init_args``.
    """
    if spec is None:
        return None
    spec = dict(spec)
    if "class_path" not in spec:
        return spec
    class_path = spec.pop("class_path")
    if class_path.rsplit(".", 1)[-1] != "Dense":
        raise NotImplementedError(f"only hepattn Dense nets are ported, got '{class_path}'")
    args = dict(spec.pop("init_args", None) or {})
    if spec:
        raise ValueError(f"unexpected keys next to class_path: {sorted(spec)}")
    return args


def make_dense(spec: dict, factory: LayerFactory, name: str) -> Dense:
    """A task-head ``Dense`` (``init="torch"`` unless the spec says otherwise)."""
    return Dense(factory=factory, name=name, **{"init": "torch", **dense_kwargs(spec)})


def finfo_min(dtype) -> float:
    """``torch.finfo(dtype).min`` for float32 / bfloat16 / float16 (``ml_dtypes.finfo`` knows bfloat16)."""
    return float(ml_dtypes.finfo(keras.backend.standardize_dtype(dtype)).min)


def topk_mask(scores, k: int, axis: int):
    """Boolean mask, True at the ``k`` largest entries of ``scores`` along ``axis`` (``hepattn.utils.masks.topk_attn``).

    Ties: ``top_k`` keeps the lower index (JAX); torch's topk order on ties is unspecified.
    """
    moved = keras.ops.moveaxis(scores, axis, -1)
    _, idx = keras.ops.top_k(moved, k)
    hot = keras.ops.sum(keras.ops.one_hot(idx, moved.shape[-1], dtype="int32"), axis=-2) > 0
    return keras.ops.moveaxis(hot, -1, axis)


class Task(keras.layers.Layer):
    """Base: per-layer scheduling as ``hepattn.models.task.Task``. ``call(x, outputs=None, training=None)``."""

    def __init__(self, task_name: str, has_intermediate_loss: bool, has_first_layer_loss: bool | None = None, permute_loss: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.has_intermediate_loss = has_intermediate_loss
        self.has_first_layer_loss = has_first_layer_loss if has_first_layer_loss is not None else has_intermediate_loss
        self.permute_loss = permute_loss

    def should_run_at_layer(self, layer_index: int) -> bool:
        if not self.has_intermediate_loss:
            return False
        return not (layer_index == 0 and not self.has_first_layer_loss)

    def attn_mask(self, outputs: dict) -> dict:
        return {}


class ObjectClassificationTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict,
        costs: dict,
        net: dict | None = None,
        dim: int | None = None,
        num_classes: int = 1,
        class_weights: list[float] | None = None,
        null_weight: float = 1.0,
        mask_queries: bool = False,
        has_intermediate_loss: bool = True,
        has_first_layer_loss: bool = False,
        factory: LayerFactory | None = None,
        layer_name: str = "task",
    ) -> None:
        """Args as ``hepattn.models.task.ObjectClassificationTask``; ``net`` is a Dense spec.

        Raises:
            ValueError: As torch (net/dim both or neither, output size, class_weights length, first-layer loss).
            NotImplementedError: For ``mask_queries=True`` (``query_mask`` is not ported).
        """
        if has_first_layer_loss and not has_intermediate_loss:
            raise ValueError("has_first_layer_loss=True requires has_intermediate_loss=True")
        super().__init__(name, has_intermediate_loss, has_first_layer_loss, name=layer_name)
        if net is not None and dim is not None:
            raise ValueError("Cannot specify both 'net' and 'dim'. Choose one.")
        if net is None and dim is None:
            raise ValueError("Must specify either 'net' or 'dim'.")
        if mask_queries:
            raise NotImplementedError("mask_queries=True is not ported (the torch forward never reads it either)")
        factory = factory or LayerFactory()

        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses_config = losses
        self.costs_config = costs
        self.num_classes = num_classes
        self.output_size = 1 if num_classes == 1 else num_classes + 1
        net = net if net is not None else {"input_size": dim, "output_size": self.output_size}
        self.net = make_dense(net, factory, f"{layer_name}_net")
        if self.net.output_size != self.output_size:
            raise ValueError(
                f"Network output size ({self.net.output_size}) doesn't match expected size for num_classes={num_classes} ({self.output_size})"
            )

        weights = [1.0] * (num_classes + 1)
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(f"Length of class_weights ({len(class_weights)}) does not match number of classes ({num_classes})")
            weights[:num_classes] = [float(w) for w in class_weights]
        weights[-1] = float(null_weight)
        self.loss_weights = tuple(weights)  # a constant (torch: a buffer); used by the S4 losses

        self.logits_key = output_object + "_logit"
        self.probs_key = output_object + "_class_prob"

    def build(self, x_shape) -> None:
        self.net.build(x_shape[self.input_object + "_embed"])

    def call(self, x, outputs=None, training=None):
        logits = self.net(x[self.input_object + "_embed"], training=training)
        if self.num_classes == 1:
            logits = keras.ops.squeeze(logits, -1)
            sig = keras.ops.sigmoid(logits)
            probs = keras.ops.stack([sig, 1 - sig], axis=-1)
        else:
            probs = keras.ops.softmax(logits, axis=-1)
        return {self.logits_key: logits, self.probs_key: probs}


class ObjectHitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict,
        costs: dict,
        dim: int,
        object_net: dict | None = None,
        constituent_net: dict | None = None,
        null_weight: float = 1.0,
        mask_attn: bool = True,
        target_field: str = "valid",
        logit_scale: float = 1.0,
        pred_threshold: float = 0.5,
        mask_attention_threshold: float | None = None,
        has_intermediate_loss: bool = True,
        factory: LayerFactory | None = None,
        layer_name: str = "task",
    ) -> None:
        """Args as ``hepattn.models.task.ObjectHitMaskTask``; ``object_net`` / ``constituent_net`` are Dense specs.

        The logit einsum is a factory leaf ``{layer_name}_mask_einsum`` (QEinsum when quantized, as the hybrid).
        """
        super().__init__(name, has_intermediate_loss, name=layer_name)
        factory = factory or LayerFactory()
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.target_field = target_field
        self.losses_config = losses
        self.costs_config = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_attn = mask_attn
        self.logit_scale = logit_scale
        self.pred_threshold = pred_threshold
        self.mask_attention_threshold = mask_attention_threshold if mask_attention_threshold is not None else pred_threshold

        self.object_net = make_dense(
            object_net if object_net is not None else {"input_size": dim, "output_size": dim}, factory, f"{layer_name}_object_net"
        )
        self.constituent_net = make_dense(constituent_net, factory, f"{layer_name}_constituent_net") if constituent_net is not None else None
        self.mask_einsum = factory.einsum("bnc,bmc->bnm", name=f"{layer_name}_mask_einsum")

        self.output_object_hit = output_object + "_" + input_constituent
        self.logit_key = self.output_object_hit + "_logit"

    def build(self, x_shape) -> None:
        q_shape = x_shape[self.input_object + "_embed"]
        c_shape = x_shape[self.input_constituent + "_embed"]
        self.object_net.build(q_shape)
        c_dim = c_shape[-1]
        if self.constituent_net is not None:
            self.constituent_net.build(c_shape)
            c_dim = self.constituent_net.output_size
        self.mask_einsum.build([(None, q_shape[1], self.object_net.output_size), (None, c_shape[1], c_dim)])

    def call(self, x, outputs=None, training=None):
        mask_tokens = self.object_net(x[self.input_object + "_embed"], training=training)
        xs = x[self.input_constituent + "_embed"]
        if self.constituent_net is not None:
            xs = self.constituent_net(xs, training=training)
        logit = self.logit_scale * self.mask_einsum([mask_tokens, xs], training=training)
        valid = x.get(f"{self.input_constituent}_valid")
        if valid is not None:
            # torch: logit[~valid] = finfo.min. Kept on purpose (mask_bce is inf at random init in both frameworks).
            logit = keras.ops.where(keras.ops.expand_dims(valid, -2), logit, finfo_min(logit.dtype))
        return {self.logit_key: logit}

    def attn_mask(self, outputs: dict, threshold: float | None = None) -> dict:
        if not self.mask_attn:
            return {}
        thresh = threshold if threshold is not None else self.mask_attention_threshold
        return {self.input_constituent: keras.ops.sigmoid(keras.ops.stop_gradient(outputs[self.logit_key])) >= thresh}


class IncidenceRegressionTask(Task):
    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict,
        costs: dict,
        net: dict,
        node_net: dict | None = None,
        has_intermediate_loss: bool = True,
        factory: LayerFactory | None = None,
        layer_name: str = "task",
    ) -> None:
        """Args as ``hepattn.models.task.IncidenceRegressionTask``; nets are Dense specs.

        The einsum and the query-axis softmax are factory leaves (``{layer_name}_incidence_einsum`` /
        ``_incidence_softmax``; QEinsum / QSoftmax(axis=1) when quantized, as the hybrid).
        """
        super().__init__(name, has_intermediate_loss, name=layer_name)
        factory = factory or LayerFactory()
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses_config = losses
        self.costs_config = costs
        self.net = make_dense(net, factory, f"{layer_name}_net")
        self.node_net = make_dense(node_net, factory, f"{layer_name}_node_net") if node_net is not None else None
        self.incidence_einsum = factory.einsum("bqe,ble->bql", name=f"{layer_name}_incidence_einsum")
        self.incidence_softmax = factory.softmax(axis=1, name=f"{layer_name}_incidence_softmax")
        self.incidence_key = output_object + "_incidence"

    def build(self, x_shape) -> None:
        q_shape = x_shape[self.input_object + "_embed"]
        c_shape = x_shape[self.input_constituent + "_embed"]
        self.net.build(q_shape)
        c_dim = c_shape[-1]
        if self.node_net is not None:
            self.node_net.build(c_shape)
            c_dim = self.node_net.output_size
        if self.net.output_size != c_dim:
            raise ValueError(f"{self.name}: object features {self.net.output_size} != node features {c_dim}")
        self.incidence_einsum.build([(None, q_shape[1], self.net.output_size), (None, c_shape[1], c_dim)])
        self.incidence_softmax.build((None, q_shape[1], c_shape[1]))

    def call(self, x, outputs=None, training=None):
        x_object = self.net(x[self.input_object + "_embed"], training=training)
        x_hit = x[self.input_constituent + "_embed"]
        if self.node_net is not None:
            x_hit = self.node_net(x_hit, training=training)
        inc = self.incidence_einsum([x_object, x_hit], training=training)
        inc = apply_softmax(self.incidence_softmax, inc, training=training)
        valid = keras.ops.cast(keras.ops.expand_dims(x[self.input_constituent + "_valid"], 1), inc.dtype)
        return {self.incidence_key: inc * valid}


class IncidenceBasedRegressionTask(Task):
    REGRESSION_LOSSES = ("l1", "l2", "smooth_l1")

    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        scale_dict_path: str,
        net: dict,
        loss: str = "smooth_l1",
        use_incidence: bool = True,
        use_nodes: bool = False,
        has_intermediate_loss: bool = True,
        mode: str = "offset",
        cost: str = "old",
        incidence_task_name: str = "incidence",
        class_prob_task_name: str = "classification",
        factory: LayerFactory | None = None,
        layer_name: str = "task",
    ) -> None:
        """Args as ``hepattn.models.task.IncidenceBasedRegressionTask``; ``net`` is a Dense spec.

        Raises:
            ValueError: For an unknown ``mode``, ``cost`` or ``loss`` (as torch).
        """
        super().__init__(name, has_intermediate_loss, name=layer_name)
        if mode not in {"offset", "scale"}:
            raise ValueError(f"Invalid mode {mode}, must be 'offset' or 'scale'")
        if cost not in {"old", "new"}:
            raise ValueError(f"Invalid cost mode {cost}")
        if loss not in self.REGRESSION_LOSSES:
            raise ValueError(f"Invalid loss {loss}, must be one of {self.REGRESSION_LOSSES}")
        factory = factory or LayerFactory()
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.fields = list(fields)
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.loss_fn_name = loss
        self.cost_mode = cost
        self.scaler = FeatureScaler(scale_dict_path)
        self.use_incidence = use_incidence
        self.use_nodes = use_nodes
        self.mode = mode
        self.incidence_task_name = incidence_task_name
        self.incidence_output_key = f"{output_object}_incidence"
        self.class_prob_task_name = class_prob_task_name
        self.class_prob_output_key = f"{output_object}_class_prob"
        self.net = make_dense(net, factory, f"{layer_name}_net")
        self.regression_key = output_object + "_regr"
        self.proxy_key = output_object + "_proxy_regr"

    def build(self, x_shape) -> None:
        q_shape = x_shape[self.input_object + "_embed"]
        width = q_shape[-1]
        if self.use_incidence:
            width += len(self.fields) + 1
            if self.use_nodes:
                width += x_shape[self.input_constituent + "_embed"][-1]
        if width != self.net.input_size:
            raise ValueError(f"{self.name}: regression net input_size {self.net.input_size} != assembled width {width}")
        self.net.build((None, q_shape[1], width))

    def scale_proxy_feats(self, proxy_feats):
        return keras.ops.stack([self.scaler[field].transform(proxy_feats[..., i]) for i, field in enumerate(self.fields)], axis=-1)

    def get_proxy_feats(self, incidence, inputs: dict, class_probs):
        """Proxy particles from the (detached) incidence; ``hepattn.models.task.IncidenceBasedRegressionTask.get_proxy_feats``."""
        ops = keras.ops
        c = self.input_constituent
        dtype = incidence.dtype
        proxy_feats = ops.stack([inputs[f"{c}_{field}"] for field in self.fields], axis=-1)  # (B, N, F)
        is_track = ops.expand_dims(ops.cast(inputs[f"{c}_is_track"], dtype), 1)  # (B, 1, N)

        charged_inc = incidence * is_track  # (B, P, N)
        positive = charged_inc > 0
        # the most weighted track as proxy for charged particles
        top2 = ops.logical_and(topk_mask(charged_inc, 2, axis=-2), positive)
        is_max = ops.logical_and(charged_inc == ops.max(charged_inc, axis=-2, keepdims=True), positive)
        zero_track = ops.sum(ops.cast(is_max, "int32"), axis=-1, keepdims=True) == 0
        charged_inc = ops.cast(ops.where(zero_track, top2, is_max), dtype)

        is_charged = ops.argmax(class_probs, axis=-1) < 3  # (B, P)
        charged_w = ops.expand_dims(ops.cast(is_charged, dtype), -1)

        pfc = ops.matmul(charged_inc, proxy_feats)  # (B, P, F)
        pfc = ops.concatenate([ops.expand_dims(pfc[..., 1] * ops.cosh(pfc[..., 2]), -1), pfc[..., 1:]], axis=-1)
        pfc = self.scale_proxy_feats(pfc) * charged_w

        inc_e_weighted = incidence * ops.expand_dims(proxy_feats[..., 0], 1)
        inc_e_weighted = inc_e_weighted * (1 - is_track)
        e_sum = ops.sum(inc_e_weighted, axis=-1)  # (B, P)
        inc = inc_e_weighted / (ops.expand_dims(e_sum, -1) + 1e-6)
        pfn = ops.einsum("bnf,bpn->bpf", proxy_feats, inc)
        pt = e_sum / ops.cosh(pfn[..., 2])
        pfn = ops.concatenate([ops.expand_dims(e_sum, -1), ops.expand_dims(pt, -1), pfn[..., 2:]], axis=-1)
        pfn = self.scale_proxy_feats(pfn) * (1 - charged_w)

        return pfc + pfn, is_charged

    def call(self, x, outputs=None, training=None):
        ops = keras.ops
        q_embed = x[self.input_object + "_embed"]
        if self.use_incidence:
            inc = class_probs = None
            if outputs is not None:
                if self.incidence_task_name in outputs:
                    inc = outputs[self.incidence_task_name].get(self.incidence_output_key)
                if self.class_prob_task_name in outputs:
                    class_probs = outputs[self.class_prob_task_name].get(self.class_prob_output_key)
            if inc is None or class_probs is None:
                raise RuntimeError(
                    f"When use_incidence=True, both incidence and class_probs must be provided. Got inc={inc is not None}, "
                    f"class_probs={class_probs is not None} (tasks '{self.incidence_task_name}', '{self.class_prob_task_name}')"
                )
            inc = ops.stop_gradient(inc)
            class_probs = ops.stop_gradient(class_probs)
            proxy_feats, is_charged = self.get_proxy_feats(inc, x["inputs"], class_probs)
            parts = [q_embed, proxy_feats, ops.expand_dims(ops.cast(is_charged, q_embed.dtype), -1)]
            if self.use_nodes:
                valid = ops.expand_dims(ops.cast(x[self.input_constituent + "_valid"], q_embed.dtype), -1)
                parts.append(ops.matmul(inc, valid * x[self.input_constituent + "_embed"]))
            input_data = ops.concatenate(parts, axis=-1)
        else:
            input_data = q_embed
            proxy_feats = ops.zeros_like(input_data[..., : len(self.fields)])
        out = self.net(input_data, training=training)
        preds = out + proxy_feats if self.mode == "offset" else out * proxy_feats
        return {self.regression_key: preds, self.proxy_key: proxy_feats}


TASK_CLASSES = {
    "ObjectClassificationTask": ObjectClassificationTask,
    "ObjectHitMaskTask": ObjectHitMaskTask,
    "IncidenceRegressionTask": IncidenceRegressionTask,
    "IncidenceBasedRegressionTask": IncidenceBasedRegressionTask,
}


def build_task(spec: dict, factory: LayerFactory, layer_name: str) -> Task:
    """A task from its YAML spec ``{"class_path": "hepattn.models.task.<Class>", "init_args": {...}}``.

    Raises:
        NotImplementedError: For a task class that is not ported.
        ValueError: For keys next to ``class_path`` other than ``init_args``.
    """
    class_path = spec["class_path"]
    cls = TASK_CLASSES.get(class_path.rsplit(".", 1)[-1])
    if cls is None:
        raise NotImplementedError(f"task '{class_path}' is not ported to keras_native (ported: {sorted(TASK_CLASSES)})")
    extra = set(spec) - {"class_path", "init_args"}
    if extra:
        raise ValueError(f"unexpected keys in a task spec: {sorted(extra)}")
    return cls(**dict(spec.get("init_args") or {}), factory=factory, layer_name=layer_name)
