"""Predictions and the CLIC prediction h5 file (plan section 3.10), without torch or Lightning.

``predict(outputs, tasks)`` is ``MaskFormer.predict`` for the four CLIC tasks in ``keras.ops`` (it runs under
``jax.jit``); ``tasks`` are the native task layers (``model.tasks``). Per task, as torch ``task.predict``:
classification ``<obj>_class`` (argmax of the class probabilities), ``<obj>_valid_prob`` (1 - null probability),
``<obj>_valid`` (class < num_classes); mask ``<obj>_<constituent>_valid_prob`` (sigmoid) and ``..._valid``
(>= ``pred_threshold``); incidence ``<obj>_incidence``; regression ``<obj>_<field>`` and ``<obj>_proxy_<field>``.

``PredictionWriter`` writes the h5 layout of ``hepattn.experiments.clic.PflowPredictionWriter.on_test_batch_end``
(same groups, structured dtypes and shapes, regression inverse-transformed with the scale dict, the same ftag
``H5Writer`` settings), so ``load_convert_h5`` reads it unchanged. Padding events (``event_valid`` False) are not
written. The ROOT file that the torch writer derives at ``on_test_end`` is not written here (S6b / eval).
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
from ftag.hdf5 import H5Writer
from keras import ops
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from hepattn.keras_native.scaling import FeatureScaler

REGRESSION_FIELDS = ("e", "pt", "eta", "sinphi", "cosphi")


def _predict_task(task, out: dict) -> dict:
    kind = type(task).__name__
    obj = getattr(task, "output_object", None)
    if kind == "ObjectClassificationTask":
        probs = out[f"{obj}_class_prob"]
        classes = ops.argmax(probs, axis=-1)
        return {f"{obj}_class": classes, f"{obj}_valid_prob": 1 - probs[..., -1], f"{obj}_valid": classes < task.num_classes}
    if kind == "ObjectHitMaskTask":
        hit = task.output_object_hit
        probs = ops.sigmoid(out[f"{hit}_logit"])
        return {f"{hit}_valid_prob": probs, f"{hit}_valid": probs >= task.pred_threshold}
    if kind == "IncidenceRegressionTask":
        return {f"{obj}_incidence": out[f"{obj}_incidence"]}
    if kind == "IncidenceBasedRegressionTask":
        regr, proxy = out[f"{obj}_regr"], out[f"{obj}_proxy_regr"]
        preds = {f"{obj}_{field}": regr[..., i] for i, field in enumerate(task.fields)}
        preds.update({f"{obj}_proxy_{field}": proxy[..., i] for i, field in enumerate(task.fields)})
        return preds
    raise NotImplementedError(f"predict is not ported for task {kind}")


def predict(outputs: dict, tasks) -> dict:
    """``{layer: {task_name: {key: array}}}`` for every layer of ``outputs`` except ``encoder`` and ``_*`` keys."""
    preds = {}
    for layer, layer_out in outputs.items():
        if layer == "encoder" or layer.startswith("_"):
            continue
        preds[layer] = {task.task_name: _predict_task(task, layer_out[task.task_name]) for task in tasks if task.task_name in layer_out}
    return preds


def join_structured_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """``hepattn.utils.array_utils.join_structured_arrays`` (that module is outside keras_native).

    Raises:
        ValueError: For no arrays or arrays of different shapes.
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("All arrays in the list must have the same shape.")
    joined = np.empty(arrays[0].shape, dtype=reduce(operator.add, (a.dtype.descr for a in arrays)))
    for a in arrays:
        for name in a.dtype.names:
            joined[name] = a[name]
    return joined


def _field(array: np.ndarray, name: str, dtype) -> np.ndarray:
    return u2s(np.asarray(array)[..., None].astype(dtype), dtype=np.dtype([(name, dtype)]))


def inverse_transform(scaler: FeatureScaler, name: str, x: np.ndarray) -> np.ndarray:
    """The scale-dict inverse of ``name`` in float32 numpy (``VarTransform.inverse_transform``)."""
    return np.asarray(scaler.transforms[name].inverse_transform(np.asarray(x, np.float32)), dtype=np.float32)


def batch_records(
    outputs: dict, preds: dict, targets: dict, event_number: np.ndarray, scaler: FeatureScaler, fields=REGRESSION_FIELDS
) -> dict[str, dict[str, np.ndarray]]:
    """The structured arrays ``PflowPredictionWriter.on_test_batch_end`` builds, from final-layer ``outputs`` / ``preds``."""
    rec: dict[str, dict[str, np.ndarray]] = {"events": {"event_number": _field(np.asarray(event_number).reshape(-1), "event_number", "i8")}}
    rec["object_class"] = {"targets": _field(targets["particle_class"], "object_class", "i8")}
    if "classification" in preds:
        rec["object_class"]["preds"] = _field(preds["classification"]["pflow_class"], "pflow_class", "i8")
    rec["object_masks"] = {
        "targets": _field(targets["particle_node_valid"], "truth_masks", "i8"),
        "preds": _field(outputs["mask"]["pflow_node_logit"], "mask_logits", np.float32),
    }
    if "particle_incidence" in targets and "incidence" in preds:
        rec["incidence"] = {
            "targets": _field(targets["particle_incidence"], "truth_incidence", np.float32),
            "preds": _field(preds["incidence"]["pflow_incidence"], "pred_incidence", np.float32),
        }
    rec["regression"] = {}
    regression = preds["regression"]
    for name in fields:
        truth = np.asarray(targets[f"particle_{name}"], np.float32)
        pred = np.asarray(regression[f"pflow_{name}"], np.float32)
        proxy = np.asarray(regression[f"pflow_proxy_{name}"], np.float32) if f"pflow_proxy_{name}" in regression else None
        if name in scaler.transforms:
            truth, pred = inverse_transform(scaler, name, truth), inverse_transform(scaler, name, pred)
            proxy = inverse_transform(scaler, name, proxy) if proxy is not None else None
        rec["regression"][f"truth_{name}"] = _field(truth, f"truth_{name}", np.float32)
        rec["regression"][f"pred_{name}"] = _field(pred, f"pred_{name}", np.float32)
        if proxy is not None:
            rec["regression"][f"proxy_{name}"] = _field(proxy, f"proxy_{name}", np.float32)
    return rec


def output_path(ckpt_path: str | Path, test_suff: str | None = None) -> Path:
    """``PflowPredictionWriter.output_path``: ``<ckpt dir>/<ckpt stem>__test[_<suffix>].h5``."""
    ckpt_path = Path(ckpt_path)
    suffix = f"_{test_suff}" if test_suff else ""
    return ckpt_path.parent / f"{ckpt_path.stem}__test{suffix}.h5"


class PredictionWriter:
    """Accumulates test batches into the ``PflowPredictionWriter`` h5 file.

    Args:
        path: Output ``.h5`` path (``output_path`` gives the torch naming).
        num_events: Number of events of the test set (the dataset shapes' first axis, as the torch writer).
        scale_dict_path: The scale dict of the regression targets (``data.scale_dict_path``).
    """

    def __init__(self, path: str | Path, num_events: int, scale_dict_path: str | Path, fields=REGRESSION_FIELDS) -> None:
        self.path = Path(path)
        self.num_events = num_events
        self.scaler = FeatureScaler(scale_dict_path)
        self.fields = tuple(fields)
        self.writer: H5Writer | None = None

    def write_batch(self, outputs: dict, preds: dict, targets: dict, event_number, event_valid=None) -> None:
        """One batch: final-layer ``outputs`` and ``preds`` (numpy or device arrays), its targets and event numbers."""
        keep = slice(None) if event_valid is None else np.asarray(event_valid, bool)

        def host(tree):
            if isinstance(tree, dict):
                return {k: host(v) for k, v in tree.items()}
            return np.asarray(tree)[keep]

        rec = batch_records(host(outputs), host(preds), host(targets), np.asarray(event_number)[keep], self.scaler, self.fields)
        to_write = {group: join_structured_arrays(list(arrays.values())) for group, arrays in rec.items()}
        if self.writer is None:
            self.writer = H5Writer(
                jets_name="events",
                dst=self.path,
                dtypes={k: v.dtype for k, v in to_write.items()},
                shapes={k: (self.num_events, *v.shape[1:]) for k, v in to_write.items()},
                shuffle=False,
                precision="full",
            )
        self.writer.write(to_write)

    def close(self) -> Path:
        if self.writer is not None:
            self.writer.close()
        return self.path
