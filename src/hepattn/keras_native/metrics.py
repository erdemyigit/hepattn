"""Host numpy metrics of the CLIC Lightning module, outside ``jit`` (plan section 3.10).

Replaces, with the same arithmetic:
- the torchmetrics objects of ``experiments/clic/lightning_module.py`` (``MulticlassAccuracy(num_classes=6)`` micro and
  macro, ``BinaryRecall``, ``BinaryPrecision``): streaming tp/fp/fn counts, reduced as torchmetrics 1.8
  (``_accuracy_reduce``, ``_precision_recall_reduce``, ``_adjust_weights_safe_divide``: macro averages over the
  classes with ``tp + fp + fn > 0``; a zero denominator gives 0);
- ``MaskInference.exact_match`` / ``eff`` / ``pur`` (``experiments/clic/metrics.py``) and ``MPflow.mask_metrics``;
- the task ``metrics`` the wrapper logs (``ObjectClassificationTask.metrics``, ``IncidenceBasedRegressionTask.metrics``).

``ClicMetrics`` accumulates one stage (val / test) over batches: the torchmetrics objects per layer, the per-batch
values as Lightning's epoch reduction does (mean weighted by the logged ``batch_size``: 1 for the mask metrics, the
batch's event count for the task metrics). Deviation from the Lightning module, on purpose: there one torchmetrics
object per metric is shared by every decoder layer (``self.eff`` etc. are updated once per layer), so its epoch value
pools all layers under each layer's name; here each layer has its own (``docs/hgq/port_log/S6a.md``).

Inputs are predictions after matching (``MaskFormer.predict`` of the permuted outputs), numpy, with padding events
removed (``event_valid``).
"""

from collections import defaultdict

import numpy as np


def _safe_divide(num, den, zero_division: float = 0.0):
    """Torchmetrics ``_safe_divide`` in float32 (its dtype for integer counts)."""
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, np.float32(zero_division), num / np.where(den == 0, np.float32(1), den)).astype(np.float32)


class MulticlassAccuracy:
    """torchmetrics ``MulticlassAccuracy(num_classes, average)`` with ``top_k=1``, global, no ``ignore_index``."""

    def __init__(self, num_classes: int, average: str = "macro") -> None:
        if average not in {"micro", "macro"}:
            raise ValueError(f"average must be micro or macro, got {average!r}")
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        self.tp = np.zeros(self.num_classes, np.int64)
        self.fp = np.zeros(self.num_classes, np.int64)
        self.fn = np.zeros(self.num_classes, np.int64)

    def update(self, preds, target) -> None:
        preds = np.asarray(preds).reshape(-1).astype(np.int64)
        target = np.asarray(target).reshape(-1).astype(np.int64)
        if preds.shape != target.shape:
            raise ValueError(f"preds {preds.shape} and target {target.shape} differ")
        if preds.size and (min(preds.min(), target.min()) < 0 or max(preds.max(), target.max()) >= self.num_classes):
            raise ValueError(f"class index outside [0, {self.num_classes})")
        n = self.num_classes
        hit = preds == target
        self.tp += np.bincount(target[hit], minlength=n)
        self.fp += np.bincount(preds[~hit], minlength=n)
        self.fn += np.bincount(target[~hit], minlength=n)

    def compute(self) -> float:
        if self.average == "micro":
            return float(_safe_divide(self.tp.sum(), self.tp.sum() + self.fn.sum()))
        score = _safe_divide(self.tp, self.tp + self.fn)
        weights = ((self.tp + self.fp + self.fn) != 0).astype(np.float32)
        return float(_safe_divide(weights * score, weights.sum()).sum(dtype=np.float32))


class _BinaryStat:
    stat = ""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = 0

    def update(self, preds, target) -> None:
        preds = np.asarray(preds).reshape(-1).astype(bool)
        target = np.asarray(target).reshape(-1).astype(bool)
        if preds.shape != target.shape:
            raise ValueError(f"preds {preds.shape} and target {target.shape} differ")
        self.tp += int((preds & target).sum())
        self.fp += int((preds & ~target).sum())
        self.fn += int((~preds & target).sum())

    def compute(self) -> float:
        other = self.fp if self.stat == "precision" else self.fn
        return float(_safe_divide(self.tp, self.tp + other))


class BinaryRecall(_BinaryStat):
    """torchmetrics ``BinaryRecall()`` on boolean inputs: tp / (tp + fn), 0 if empty."""

    stat = "recall"


class BinaryPrecision(_BinaryStat):
    """torchmetrics ``BinaryPrecision()`` on boolean inputs: tp / (tp + fp), 0 if empty."""

    stat = "precision"


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def mask_exact_match(pred: np.ndarray, tgt: np.ndarray) -> float:
    """``MaskInference.exact_match``: fraction of rows with every node right; NaN with no row."""
    if len(tgt) == 0:
        return float("nan")
    return _mean((pred == tgt).all(-1).astype(np.float32))


def mask_eff(pred: np.ndarray, tgt: np.ndarray) -> float:
    """``MaskInference.eff``: mean over rows of |pred & tgt| / |tgt| (float32, as torch)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _mean((pred & tgt).sum(-1).astype(np.float32) / tgt.sum(-1).astype(np.float32))


def mask_pur(pred: np.ndarray, tgt: np.ndarray) -> float:
    """``MaskInference.pur``: mean over rows of |pred & tgt| / |pred| (float32, as torch)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _mean((pred & tgt).sum(-1).astype(np.float32) / pred.sum(-1).astype(np.float32))


def mask_metrics(pred_masks: np.ndarray, truth_masks: np.ndarray, truth_valid: np.ndarray) -> dict[str, float]:
    """``MPflow.mask_metrics`` values (exact match, recall, purity) on [B, P, N] masks and [B, P] truth validity."""
    truth = truth_masks[truth_valid]
    pred = pred_masks[truth_valid]
    recall_idx = truth.sum(-1) > 0
    pur_idx = pred.sum(-1) > 0
    return {
        "mask_exact_match": mask_exact_match(pred, truth),
        "mask_recall": mask_eff(pred[recall_idx], truth[recall_idx]),
        "mask_purity": mask_pur(pred[pur_idx], truth[pur_idx]),
    }


def classification_task_metrics(pred_valid: np.ndarray, true_valid: np.ndarray) -> dict[str, float]:
    """``ObjectClassificationTask.metrics`` without ``query_mask``."""
    pred_valid = pred_valid.astype(bool)
    true_valid = true_valid.astype(bool)
    eps = np.float32(1e-12)
    tp = np.float32((pred_valid & true_valid).sum())
    fp = np.float32((pred_valid & ~true_valid).sum())
    return {
        "num_queries": float(pred_valid.shape[1]),
        "query_frac_pred_valid": float(pred_valid.astype(np.float32).mean()),
        "query_eff": float(tp / np.maximum(np.float32(true_valid.sum()), eps)),
        "query_fr": float(fp / np.maximum(np.float32(pred_valid.sum()), eps)),
    }


def regression_task_metrics(
    preds: dict, targets: dict, fields, output_object: str = "pflow", target_object: str = "particle", proxy: bool = True
) -> dict[str, float]:
    """``IncidenceBasedRegressionTask.metrics`` (``RegressionTask.metrics`` plus the proxy rows), float32 as torch."""
    valid = np.asarray(targets[f"{target_object}_valid"]).astype(bool)
    out = {}
    for prefix, suffix in (("", ""), ("proxy_", "_proxy")) if proxy else (("", ""),):
        for field in fields:
            pred = np.asarray(preds[f"{output_object}_{prefix}{field}"], dtype=np.float32)[valid]
            target = np.asarray(targets[f"{target_object}_{field}"], dtype=np.float32)[valid]
            abs_err = np.abs(pred - target)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"{field}{suffix}_abs_res"] = _mean(abs_err)
                out[f"{field}{suffix}_abs_norm_res"] = _mean(abs_err / np.abs(target) + np.float32(1e-8))
    return out


class WeightedMean:
    """Lightning's epoch reduction of a logged scalar: sum(value * batch_size) / sum(batch_size)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        self.total += float(value) * weight
        self.weight += weight

    def compute(self) -> float:
        return self.total / self.weight if self.weight else float("nan")


class ClicMetrics:
    """The validation / test metrics of ``MPflow.log_metrics`` for one stage, accumulated over batches.

    ``update(preds, targets)``: ``preds`` is ``{layer: {task: {key: array}}}`` (``predict`` of the matched outputs),
    ``targets`` the batch targets, both numpy with padding events removed. ``compute()`` returns
    ``{"<stage>/<layer>_<name>": value}`` with the Lightning module's names.
    """

    def __init__(self, stage: str = "val", num_classes: int = 6, regression_fields=("e", "pt", "eta", "sinphi", "cosphi")) -> None:
        self.stage = stage
        self.num_classes = num_classes
        self.fields = tuple(regression_fields)
        self.reset()

    def reset(self) -> None:
        self.counters: dict[str, object] = {}
        self.means: dict[str, WeightedMean] = defaultdict(WeightedMean)

    def _counter(self, name: str, factory):
        if name not in self.counters:
            self.counters[name] = factory()
        return self.counters[name]

    def update(self, preds: dict, targets: dict) -> None:
        batch_size = float(np.asarray(targets["particle_valid"]).shape[0])
        class_labels = np.asarray(targets["particle_class"])
        truth_valid = class_labels < self.num_classes - 1
        prefix = f"{self.stage}/"
        for layer, layer_preds in preds.items():
            for task_name, task_preds in layer_preds.items():
                if task_name == "classification":
                    classes = np.asarray(task_preds["pflow_class"])
                    for average in ("micro", "macro"):
                        acc = self._counter(
                            f"{prefix}{layer}_obj_class_accuracy_{average}", lambda a=average: MulticlassAccuracy(self.num_classes, a)
                        )
                        acc.update(classes, class_labels)
                    pred_valid = classes < self.num_classes - 1
                    self._counter(f"{prefix}{layer}_eff", BinaryRecall).update(pred_valid, truth_valid)
                    self._counter(f"{prefix}{layer}_pur", BinaryPrecision).update(pred_valid, truth_valid)
                    for key, value in classification_task_metrics(
                        np.asarray(task_preds["pflow_valid"]), np.asarray(targets["particle_valid"])
                    ).items():
                        self.means[f"{prefix}{layer}_{task_name}_{key}"].update(value, batch_size)
                elif task_name == "mask":
                    masks = mask_metrics(np.asarray(task_preds["pflow_node_valid"]), np.asarray(targets["particle_node_valid"]), truth_valid)
                    for key, value in masks.items():
                        self.means[f"{prefix}{layer}_{key}"].update(value, 1.0)
                elif task_name == "regression":
                    for key, value in regression_task_metrics(task_preds, targets, self.fields).items():
                        self.means[f"{prefix}{layer}_{task_name}_{key}"].update(value, batch_size)

    def compute(self) -> dict[str, float]:
        out = {name: counter.compute() for name, counter in self.counters.items()}
        out.update({name: mean.compute() for name, mean in self.means.items()})
        return dict(sorted(out.items()))
