"""Matching costs and losses of the CLIC MaskFormer tasks, as pure functions of output/target dicts.

This replaces ``MaskFormer.loss`` (``src/hepattn/models/maskformer.py:385``) and the ``cost`` / ``loss``
methods of the four CLIC tasks (``src/hepattn/models/task.py``) inside a jitted JAX step:

    costs = model_costs(outputs, targets, specs)                    # {layer: [B, P, T]} float32
    idx = match_layers(costs, targets["particle_valid"])            # {layer: [B, P]} int32, one host call
    losses = model_losses(permute_outputs(outputs, idx, specs), targets, specs)
    total = total_loss(losses)

``matched_loss`` runs these four lines. Only the task configurations of ``pflow_hgq.yaml`` and org/main
``base.yaml`` (the same task block) are supported; anything else raises (``specs_from_config``).

Interface (B batch, P = 150 queries, T = 150 target particles, N = 160 nodes, registers already removed)
-----------------------------------------------------------------------------------------------------------
``outputs``: ``{layer_name: {task_name: {key: array}}}``. ``layer_name`` is ``"layer_0"`` ...
``"layer_{L-1}"`` and ``"final"``. An ``"encoder"`` entry must be absent or empty: in torch it holds only
encoder-task outputs and the dynamic-query ``query_mask`` (``maskformer.py:150-156``, ``decoder.py:223-232``),
neither of which is ported, so a non-empty one raises. A task contributes
costs and losses at a layer exactly when its name is a key of that layer (as in torch; the decoder decides
this with ``has_intermediate_loss`` / ``has_first_layer_loss``). Other keys of a layer dict are passed
through unchanged. Keys per task (CLIC names, ``output_object = "pflow"``):

==================  =====================================  ==================
task                key                                    shape, dtype
==================  =====================================  ==================
classification      ``pflow_logit``                        [B, P, 6] float
classification      ``pflow_class_prob``                   [B, P, 6] float (softmax of the logits)
mask                ``pflow_node_logit``                   [B, P, N] float, ``finfo.min`` at invalid nodes
incidence           ``pflow_incidence``                    [B, P, N] float (softmax over P, times node_valid)
regression          ``pflow_regr``, ``pflow_proxy_regr``   [B, P, 5] float
==================  =====================================  ==================

``targets`` (``CLICDataset`` names, ``target_object = "particle"``, ``input_constituent = "node"``):

==========================  ============================================================
key                         shape, dtype
==========================  ============================================================
``particle_valid``          [B, T] bool; the valid particles come first in each event
``particle_class``          [B, T] int in [0, 5]; 5 is the null class
``node_valid``              [B, N] bool
``particle_node_valid``     [B, T, N] bool (mask-task target, ``target_field = "valid"``)
``particle_incidence``      [B, T, N] float
``particle_<field>``        [B, T] float for e, pt, eta, sinphi, cosphi; NaN for invalid particles
==========================  ============================================================

``query_mask`` (dynamic queries) is not supported and raises. Outputs may be any float dtype; every
cost and loss is computed in float32 (this replaces the ``torch.autocast(enabled=False)`` blocks).
NaN regression targets are replaced by 0 BEFORE the loss. For the l1 loss in JAX this is not strictly
needed (the ``abs`` JVP is a select, so ``where(mask, |out - nan|, 0)`` has a finite gradient), but any
residual-weighted loss (l2, smooth_l1, a mask-multiply form) gives NaN gradients without it
(``docs/hgq/port_log/S4.md`` finding 2). The matcher settings are read from the YAML matcher block with
``matcher_n_jobs_from_config`` (org ``base.yaml`` uses 16 threads; the default here is 1). The costs
keep the torch NaN columns (they are under ``stop_gradient``, and the matcher reads only the valid target
rows); ``matching.host_match`` sanitises non-finite entries as the torch ``Matcher`` does.
"""

import logging
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from keras import ops

from hepattn.keras_native import losses as L
from hepattn.keras_native.matching import match, permute

logger = logging.getLogger(__name__)

ENCODER_KEY = "encoder"


def _nan_to_zero(x):
    return ops.nan_to_num(x, nan=0.0)


def _check_targets(targets: dict) -> None:
    if "query_mask" in targets:
        raise NotImplementedError("query_mask (dynamic queries) is not supported by keras_native")


def _check_encoder_outputs(outputs: dict) -> None:
    encoder = outputs.get(ENCODER_KEY) or {}
    if encoder:
        raise NotImplementedError(
            f"outputs[{ENCODER_KEY!r}] has {sorted(encoder)}: encoder tasks and query_mask (dynamic queries) are not supported by keras_native"
        )


def _require(names, supported: set[str], what: str) -> None:
    unsupported = set(names) - supported
    if unsupported:
        raise NotImplementedError(f"unsupported {what}: {sorted(unsupported)}; supported: {sorted(supported)}")


@dataclass
class ObjectClassificationSpec:
    """``task.ObjectClassificationTask`` with ``num_classes > 1`` (multi-class + null)."""

    name: str
    output_object: str
    target_object: str
    losses: dict[str, float]
    costs: dict[str, float]
    num_classes: int = 1
    class_weights: tuple[float, ...] | None = None
    null_weight: float = 1.0
    permute_loss: bool = True

    def __post_init__(self):
        if self.class_weights is not None:
            self.class_weights = tuple(self.class_weights)
        if self.num_classes < 2:
            raise NotImplementedError("binary ObjectClassificationTask (num_classes=1) is not supported")
        _require(self.losses, {"object_ce"}, f"{self.name} losses")
        _require(self.costs, {"object_ce"}, f"{self.name} costs")
        if self.class_weights is not None and len(self.class_weights) != self.num_classes:
            raise ValueError(f"len(class_weights)={len(self.class_weights)} != num_classes={self.num_classes}")

    @property
    def outputs(self) -> list[str]:
        return [self.output_object + "_logit", self.output_object + "_class_prob"]

    @property
    def loss_weights(self) -> np.ndarray:
        """The torch ``loss_weights`` buffer: [class weights..., null_weight]."""
        weights = np.ones(self.num_classes + 1, dtype=np.float32)
        if self.class_weights is not None:
            weights[: self.num_classes] = self.class_weights
        weights[-1] = self.null_weight
        return weights

    def cost(self, outputs: dict, targets: dict) -> dict:
        # PR #3 (a2bc9c7): object_ce_cost softmaxes, so it takes the logits, not *_class_prob
        logits = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_class"]
        return {name: w * L.object_ce_cost(logits, target) for name, w in self.costs.items()}

    def loss(self, outputs: dict, targets: dict) -> dict:
        logits = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_class"]
        return {name: w * L.object_ce_loss(logits, target, weight=self.loss_weights) for name, w in self.losses.items()}


@dataclass
class ObjectHitMaskSpec:
    """``task.ObjectHitMaskTask`` (losses ``mask_bce`` / ``mask_dice``, cost ``mask_dice``)."""

    name: str
    input_constituent: str
    output_object: str
    target_object: str
    losses: dict[str, float]
    costs: dict[str, float]
    null_weight: float = 1.0
    target_field: str = "valid"
    permute_loss: bool = True

    def __post_init__(self):
        _require(self.losses, {"mask_bce", "mask_dice"}, f"{self.name} losses")
        _require(self.costs, {"mask_dice"}, f"{self.name} costs")

    @property
    def logit_key(self) -> str:
        return f"{self.output_object}_{self.input_constituent}_logit"

    @property
    def target_key(self) -> str:
        return f"{self.target_object}_{self.input_constituent}_{self.target_field}"

    @property
    def outputs(self) -> list[str]:
        return [self.logit_key]

    def cost(self, outputs: dict, targets: dict) -> dict:
        logits = outputs[self.logit_key]
        target = targets[self.target_key]
        pad = targets[self.input_constituent + "_valid"]
        return {name: w * L.mask_dice_cost(logits, target, pad) for name, w in self.costs.items()}

    def loss(self, outputs: dict, targets: dict) -> dict:
        logits = outputs[self.logit_key]
        target = L.f32(targets[self.target_key])
        pad = targets[self.input_constituent + "_valid"]
        object_valid = targets[self.target_object + "_valid"]
        sample_weight = target + self.null_weight * (1.0 - target)
        fns = {
            "mask_bce": lambda: L.mask_bce_loss(logits, target, object_valid, pad, sample_weight=sample_weight),
            "mask_dice": lambda: L.mask_dice_loss(logits, target, object_valid, pad),
        }
        return {name: w * fns[name]() for name, w in self.losses.items()}


@dataclass
class IncidenceRegressionSpec:
    """``task.IncidenceRegressionTask`` (loss and cost ``kl_div``)."""

    name: str
    input_constituent: str
    output_object: str
    target_object: str
    losses: dict[str, float]
    costs: dict[str, float]
    permute_loss: bool = True

    def __post_init__(self):
        _require(self.losses, {"kl_div"}, f"{self.name} losses")
        _require(self.costs, {"kl_div"}, f"{self.name} costs")

    @property
    def incidence_key(self) -> str:
        return self.output_object + "_incidence"

    @property
    def outputs(self) -> list[str]:
        return [self.incidence_key]

    def cost(self, outputs: dict, targets: dict) -> dict:
        pred = outputs[self.incidence_key]
        true = targets[self.target_object + "_incidence"]
        return {name: w * L.kl_div_cost(pred, true) for name, w in self.costs.items()}

    def loss(self, outputs: dict, targets: dict) -> dict:
        pred = outputs[self.incidence_key]
        true = targets[self.target_object + "_incidence"]
        node_valid = ops.expand_dims(targets[self.input_constituent + "_valid"], 1)
        object_valid = ops.expand_dims(targets[self.target_object + "_valid"], -1)
        mask = ops.logical_and(node_valid, object_valid)
        return {name: w * L.kl_div_loss(pred, true, mask) for name, w in self.losses.items()}


@dataclass
class IncidenceBasedRegressionSpec:
    """``task.IncidenceBasedRegressionTask`` with ``loss="l1"`` and ``cost="new"`` (loss path only)."""

    name: str
    output_object: str
    target_object: str
    fields: tuple[str, ...]
    loss_weight: float
    cost_weight: float
    loss_fn_name: str = "smooth_l1"  # torch default (unsupported, raises)
    cost_mode: str = "old"  # torch default (unsupported, raises)
    permute_loss: bool = True

    def __post_init__(self):
        _require([self.loss_fn_name], {"l1"}, f"{self.name} loss")
        _require([self.cost_mode], {"new"}, f"{self.name} cost mode")
        self.fields = tuple(self.fields)

    @property
    def outputs(self) -> list[str]:
        return [self.output_object + "_regr", self.output_object + "_proxy_regr"]

    def _target(self, targets: dict):
        return ops.stack([L.f32(targets[f"{self.target_object}_{f}"]) for f in self.fields], axis=-1)

    def cost(self, outputs: dict, targets: dict) -> dict:
        pred = outputs[self.output_object + "_regr"]
        return {f"regr_{self.loss_fn_name}": self.cost_weight * L.regr_l1_cost(pred, self._target(targets))}

    def loss(self, outputs: dict, targets: dict) -> dict:
        pred = outputs[self.output_object + "_regr"]
        target = _nan_to_zero(self._target(targets))
        valid = targets[self.target_object + "_valid"]
        return {self.loss_fn_name: self.loss_weight * L.regr_l1_loss(pred, target, valid)}


TaskSpec = ObjectClassificationSpec | ObjectHitMaskSpec | IncidenceRegressionSpec | IncidenceBasedRegressionSpec


def clic_specs() -> list[TaskSpec]:
    """The four task specs of ``pflow_hgq.yaml``, in its task order (the order sets the cost summation order)."""
    common = {"output_object": "pflow", "target_object": "particle"}
    return [
        ObjectClassificationSpec(
            name="classification",
            num_classes=5,
            losses={"object_ce": 2.0},
            costs={"object_ce": 2.0},
            class_weights=(1.0, 3.0, 8.0, 1.5, 1.0),
            null_weight=0.5,
            **common,
        ),
        ObjectHitMaskSpec(
            name="mask", input_constituent="node", losses={"mask_bce": 5.0, "mask_dice": 1.0}, costs={"mask_dice": 1.0}, null_weight=1.0, **common
        ),
        IncidenceRegressionSpec(name="incidence", input_constituent="node", losses={"kl_div": 1.0}, costs={"kl_div": 1.0}, **common),
        IncidenceBasedRegressionSpec(
            name="regression",
            fields=("e", "pt", "eta", "sinphi", "cosphi"),
            loss_fn_name="l1",
            loss_weight=10.0,
            cost_weight=10.0,
            cost_mode="new",
            **common,
        ),
    ]


# Init args that only affect the forward pass, prediction, metrics or module construction.
_FORWARD_ONLY = {
    "input_object",
    "net",
    "node_net",
    "object_net",
    "constituent_net",
    "dim",
    "has_intermediate_loss",
    "has_first_layer_loss",
    "pred_threshold",
    "mask_attention_threshold",
    "logit_scale",
    "mask_attn",
    "use_incidence",
    "use_nodes",
    "mode",
    "scale_dict_path",
    "incidence_task_name",
    "class_prob_task_name",
}
_SPEC_CLASSES = {
    "ObjectClassificationTask": ObjectClassificationSpec,
    "ObjectHitMaskTask": ObjectHitMaskSpec,
    "IncidenceRegressionTask": IncidenceRegressionSpec,
    "IncidenceBasedRegressionTask": IncidenceBasedRegressionSpec,
}


def spec_from_config(class_path: str, init_args: dict[str, Any]) -> TaskSpec:
    """One task spec from a YAML task entry (``class_path`` + ``init_args``), with the torch defaults.

    Unsupported tasks or options raise ``NotImplementedError``; unknown init args raise ``TypeError``.
    """
    kind = class_path.rsplit(".", 1)[-1]
    if kind not in _SPEC_CLASSES:
        raise NotImplementedError(f"task {class_path} is not supported by keras_native")
    if init_args.get("mask_queries"):
        raise NotImplementedError("mask_queries=True is not supported")
    args = {k: v for k, v in init_args.items() if k not in _FORWARD_ONLY | {"mask_queries"}}
    if kind == "IncidenceBasedRegressionTask":
        args.pop("input_constituent", None)  # forward only for this task
        renamed = {"loss": "loss_fn_name", "cost": "cost_mode"}
        args = {renamed.get(k, k): v for k, v in args.items()}
    return _SPEC_CLASSES[kind](**args)


def specs_from_config(task_entries: list[dict[str, Any]]) -> list[TaskSpec]:
    """Specs from ``model.model.init_args.tasks.init_args.modules`` of a CLIC YAML config (e.g. ``pflow_hgq.yaml``)."""
    return [spec_from_config(e["class_path"], e.get("init_args", {})) for e in task_entries]


# Matcher init args that do not change the assignment (thread/process backend, timing, logging).
_MATCHER_ARGS = {
    "default_solver",
    "adaptive_solver",
    "adaptive_check_interval",
    "parallel_solver",
    "parallel_backend",
    "n_jobs",
    "device_solver",
    "verbose",
}
# Solvers whose optimum scipy reproduces. org/main base.yaml uses lap1015_late (matcher.py SOLVERS).
_HOST_SOLVERS = {"scipy", "lap1015_late"}


@cache
def _log_lap1015_fallback() -> None:
    logger.warning(
        "keras_native solves default_solver=lap1015_late with scipy: the optimal total cost is the same, "
        "but ties between equal-cost assignments may break differently"
    )


def matcher_n_jobs_from_config(matcher_entry: dict[str, Any]) -> int:
    """Host-matching threads from the YAML ``matcher`` block (``class_path`` + ``init_args``).

    Supported: ``Matcher`` with ``default_solver`` scipy or lap1015_late, ``adaptive_solver: false`` and no
    ``device_solver``. lap1015_late is solved with scipy (same optimum; logged once). Anything else raises
    ``NotImplementedError``. Returns ``n_jobs`` (torch default 8) when ``parallel_solver`` is set, else 1.
    """
    kind = matcher_entry.get("class_path", "").rsplit(".", 1)[-1]
    args = matcher_entry.get("init_args", {})
    if kind != "Matcher":
        raise NotImplementedError(f"matcher {matcher_entry.get('class_path')!r} is not supported by keras_native")
    unknown = set(args) - _MATCHER_ARGS
    if unknown:
        raise NotImplementedError(f"unsupported matcher options {sorted(unknown)}")
    solver = args.get("default_solver", "scipy")
    if solver not in _HOST_SOLVERS or args.get("adaptive_solver", True):
        raise NotImplementedError(f"only default_solver in {sorted(_HOST_SOLVERS)} with adaptive_solver=false is supported, got {args}")
    if args.get("device_solver") is not None:
        raise NotImplementedError(f"device_solver={args['device_solver']!r} is not supported by keras_native")
    if solver == "lap1015_late":
        _log_lap1015_fallback()
    return int(args.get("n_jobs", 8)) if args.get("parallel_solver", False) else 1


def _decoder_layers(outputs: dict) -> list[str]:
    _check_encoder_outputs(outputs)
    return [name for name in outputs if name != ENCODER_KEY]


def model_costs(outputs: dict, targets: dict, specs: list[TaskSpec]) -> dict:
    """``MaskFormer._compute_decoder_costs``: {layer: sum of the task costs, [B, P, T] float32}.

    Raises:
        ValueError: If no task has costs at a decoder layer (torch would fail to stack it).
    """
    _check_targets(targets)
    costs = {}
    for layer in _decoder_layers(outputs):
        total = None
        for spec in specs:
            if spec.name not in outputs[layer]:
                continue
            for cost in spec.cost(outputs[layer][spec.name], targets).values():
                total = cost if total is None else total + cost
        if total is None:
            raise ValueError(f"no task has costs at decoder layer {layer!r}")
        costs[layer] = total
    return costs


def match_layers(costs: dict, target_valid, *, n_jobs: int = 1, host_fn=None) -> dict:
    """Match every layer in ONE host call: stack to [L*B, P, T], match, split. Returns {layer: [B, P] int32}."""
    layers = list(costs)
    batch = target_valid.shape[0]
    stacked = ops.concatenate([costs[layer] for layer in layers], axis=0)
    valid = ops.tile(target_valid, (len(layers), 1))
    idx = match(stacked, valid, n_jobs=n_jobs, host_fn=host_fn)
    return {layer: idx[i * batch : (i + 1) * batch] for i, layer in enumerate(layers)}


def permute_outputs(outputs: dict, idx: dict, specs: list[TaskSpec]) -> dict:
    """``MaskFormer._match_and_permute_outputs`` without mutation: permute each task's outputs at the matched layers."""
    permuted = dict(outputs)
    for layer, layer_idx in idx.items():
        layer_out = dict(outputs[layer])
        for spec in specs:
            if not spec.permute_loss or spec.name not in layer_out:
                continue
            task_out = dict(layer_out[spec.name])
            for key in spec.outputs:
                task_out[key] = permute(task_out[key], layer_idx)
            layer_out[spec.name] = task_out
        permuted[layer] = layer_out
    return permuted


def model_losses(outputs: dict, targets: dict, specs: list[TaskSpec]) -> dict:
    """``MaskFormer._compute_decoder_losses`` on permuted outputs: {layer: {task: {loss_name: scalar}}}."""
    _check_targets(targets)
    return {
        layer: {spec.name: spec.loss(outputs[layer][spec.name], targets) for spec in specs if spec.name in outputs[layer]}
        for layer in _decoder_layers(outputs)
    }


def total_loss(losses: dict):
    """Sum of every leaf of a nested loss dict."""
    leaves = [v for task in losses.values() for parts in task.values() for v in parts.values()]
    return sum(leaves[1:], leaves[0])


def matched_loss(outputs: dict, targets: dict, specs: list[TaskSpec], *, n_jobs: int = 1, host_fn=None, target_object: str = "particle"):
    """Costs, one host matching call, permutation and losses. Returns (total, losses, idx)."""
    costs = model_costs(outputs, targets, specs)
    idx = match_layers(costs, targets[f"{target_object}_valid"], n_jobs=n_jobs, host_fn=host_fn)
    losses = model_losses(permute_outputs(outputs, idx, specs), targets, specs)
    return total_loss(losses), losses, idx
