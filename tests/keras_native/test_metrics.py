"""S6a gate: ``keras_native.metrics`` equals torchmetrics and the CLIC Lightning module on fixed predictions.

References in this process: torchmetrics 1.8 objects, ``MaskInference``, the torch task ``metrics`` of the org tasks,
and ``MPflow.log_metrics`` with ``self.log`` captured (one decoder layer, so the shared-metric pooling of the
Lightning module, documented in ``metrics.py``, does not enter). Controls: one perturbed prediction changes each value.
"""

import math

import numpy as np
import pytest
import torch
import torchmetrics as tm
from torch import nn

from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.experiments.clic.metrics import MaskInference
from hepattn.keras_native import metrics as M

from .golden.mf_cases import instantiate, org_init_args

FIELDS = ("e", "pt", "eta", "sinphi", "cosphi")
B, P, N = 3, 150, 160


def fixed_case(seed: int = 0) -> tuple[dict, dict]:
    """Predictions of one layer and targets, as ``MaskFormer.predict`` / ``CLICDataset`` give them (numpy)."""
    rng = np.random.default_rng(seed)
    n_valid = rng.integers(20, 60, size=B)
    particle_valid = np.arange(P)[None] < n_valid[:, None]
    particle_class = np.where(particle_valid, rng.integers(0, 5, size=(B, P)), 5)
    particle_class[0, 0] = 2  # class 2 present; class 4 kept out of the targets below (fp only)
    particle_class[particle_class == 4] = 3
    agree = rng.random((B, P)) < 0.6
    pflow_class = np.where(agree, particle_class, rng.integers(0, 6, size=(B, P)))
    truth_masks = (rng.random((B, P, N)) < 0.05) & particle_valid[..., None]
    truth_masks[0, 1] = False  # a valid particle with no node (filtered from the recall)
    pred_masks = np.where(rng.random((B, P, N)) < 0.9, truth_masks, ~truth_masks)
    pred_masks[0, 2] = False  # no predicted node (filtered from the purity)
    targets = {
        "particle_class": particle_class.astype(np.int64),
        "particle_valid": particle_valid,
        "particle_node_valid": truth_masks,
    }
    regression = {}
    for i, field in enumerate(FIELDS):
        target = rng.normal(size=(B, P)).astype(np.float32) + i
        targets[f"particle_{field}"] = np.where(particle_valid, target, np.nan).astype(np.float32)
        regression[f"pflow_{field}"] = (target + 0.1 * rng.normal(size=(B, P))).astype(np.float32)
        regression[f"pflow_proxy_{field}"] = (target + 0.3 * rng.normal(size=(B, P))).astype(np.float32)
    preds = {
        "classification": {
            "pflow_class": pflow_class.astype(np.int64),
            "pflow_valid": pflow_class < 5,
            "pflow_valid_prob": rng.random((B, P)).astype(np.float32),
        },
        "mask": {"pflow_node_valid": pred_masks, "pflow_node_valid_prob": rng.random((B, P, N)).astype(np.float32)},
        "regression": regression,
    }
    return preds, targets


def tt(tree):
    if isinstance(tree, dict):
        return {k: tt(v) for k, v in tree.items()}
    return torch.from_numpy(np.ascontiguousarray(tree))


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_multiclass_accuracy_streams_like_torchmetrics(average):
    ref = tm.classification.MulticlassAccuracy(num_classes=6, average=average)
    ours = M.MulticlassAccuracy(6, average)
    for seed in range(3):
        preds, targets = fixed_case(seed)
        ref.update(tt(preds["classification"]["pflow_class"]).view(-1), tt(targets["particle_class"]).view(-1))
        ours.update(preds["classification"]["pflow_class"], targets["particle_class"])
    expected = float(ref.compute())
    assert ulps(ours.compute(), expected) <= tolerance(f"accuracy_{average}")
    assert (ours.fp[4], ours.tp[4] + ours.fn[4]) != (0, 0)  # class 4: predicted, never a target


def test_macro_and_micro_differ_on_this_case():
    """The data separates the two averages, so the parametrised test checks each reduction."""
    preds, targets = fixed_case(0)
    micro, macro = M.MulticlassAccuracy(6, "micro"), M.MulticlassAccuracy(6, "macro")
    for metric in (micro, macro):
        metric.update(preds["classification"]["pflow_class"], targets["particle_class"])
    assert abs(micro.compute() - macro.compute()) > 1e-3


def test_macro_skips_classes_without_support_like_torchmetrics():
    preds, target = np.array([0, 0, 1]), np.array([0, 1, 1])  # classes 2..5 absent everywhere
    ref = tm.classification.MulticlassAccuracy(num_classes=6, average="macro")
    ours = M.MulticlassAccuracy(6, "macro")
    ours.update(preds, target)
    assert ours.compute() == float(ref(torch.from_numpy(preds), torch.from_numpy(target))) == 0.75


@pytest.mark.parametrize(
    ("ours_cls", "ref_cls"), [(M.BinaryRecall, tm.classification.BinaryRecall), (M.BinaryPrecision, tm.classification.BinaryPrecision)]
)
def test_binary_recall_precision_equal_torchmetrics(ours_cls, ref_cls):
    ours, ref = ours_cls(), ref_cls()
    for seed in range(3):
        preds, targets = fixed_case(seed)
        pred_valid = preds["classification"]["pflow_class"] < 5
        truth_valid = targets["particle_class"] < 5
        ours.update(pred_valid, truth_valid)
        ref.update(torch.from_numpy(pred_valid), torch.from_numpy(truth_valid))
    assert ours.compute() == float(ref.compute())
    empty, ref_empty = ours_cls(), ref_cls()
    empty.update(np.zeros(4, bool), np.zeros(4, bool))
    assert empty.compute() == float(ref_empty(torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))) == 0.0


def test_mask_inference_equals_torch():
    preds, targets = fixed_case(1)
    pred, tgt = preds["mask"]["pflow_node_valid"][0], targets["particle_node_valid"][0]
    tp, tt_ = torch.from_numpy(pred), torch.from_numpy(tgt)
    keep = tgt.sum(-1) > 0
    assert M.mask_exact_match(pred, tgt) == float(MaskInference.exact_match(tp, tt_))
    assert M.mask_eff(pred[keep], tgt[keep]) == pytest.approx(float(MaskInference.eff(tp[keep], tt_[keep])), rel=1e-6)
    keep = pred.sum(-1) > 0
    assert M.mask_pur(pred[keep], tgt[keep]) == pytest.approx(float(MaskInference.pur(tp[keep], tt_[keep])), rel=1e-6)
    assert math.isnan(M.mask_exact_match(pred[:0], tgt[:0])) and math.isnan(float(MaskInference.exact_match(tp[:0], tt_[:0])))
    assert math.isnan(M.mask_eff(pred[:0], tgt[:0])) and math.isnan(float(MaskInference.eff(tp[:0], tt_[:0])))


@pytest.fixture(scope="module")
def torch_tasks():
    return instantiate(org_init_args()["tasks"])


@pytest.fixture(scope="module")
def mpflow(torch_tasks):
    model = nn.Module()
    model.tasks = torch_tasks
    model.encoder_tasks = nn.ModuleList()
    return MPflow(name="t", model=model, lrs_config={})


def torch_logged(mpflow, preds: dict, targets: dict) -> dict[str, float]:
    """``MPflow.log_metrics(..., "val")`` with ``self.log`` captured; metric objects read with ``compute``."""
    logged = {}

    def log(name, value, **_):
        logged[name] = value

    mpflow.log = log
    mpflow.log_dict = lambda values, **_: logged.update(values)
    for metric in (mpflow.obj_accuracy_micro, mpflow.obj_accuracy_macro, mpflow.eff, mpflow.pur):
        metric.reset()
    mpflow.log_metrics(tt(preds), tt(targets), "val")
    return {k: float(v.compute() if isinstance(v, tm.Metric) else v) for k, v in logged.items()}


# Float32 means and the macro average are reduced in another order than torch: measured up to 3 ulps (means over
# ~120 values, 10 seeds) and 1 ulp (macro); tolerance 2x. Every count-based value is exactly equal (measured 0).
ULP_TOLERANCE = {"regression": 6, "mask_recall": 6, "mask_purity": 6, "accuracy_macro": 2}


def ulps(a: float, b: float) -> float:
    return abs(a - b) / float(np.spacing(np.float32(abs(b)))) if b else abs(a)


def tolerance(key: str) -> int:
    return next((n for part, n in ULP_TOLERANCE.items() if part in key), 0)


def assert_equal_metrics(ours: dict, ref: dict) -> None:
    assert ours.keys() == ref.keys(), ours.keys() ^ ref.keys()
    for key in ref:
        if math.isnan(ref[key]):
            assert math.isnan(ours[key]), key
        else:
            assert ulps(ours[key], ref[key]) <= tolerance(key), (key, ours[key], ref[key])


def test_clic_metrics_equal_the_lightning_module(mpflow):
    preds, targets = fixed_case(2)
    ref = torch_logged(mpflow, {"final": preds}, targets)
    ours = M.ClicMetrics("val")
    ours.update({"final": preds}, targets)
    values = ours.compute()
    assert len(values) == 4 + 3 + 4 + 20  # accuracy x2, eff, pur; 3 mask; 4 classification-task; 20 regression
    assert_equal_metrics(values, ref)


@pytest.mark.parametrize(
    ("perturb", "key"),
    [
        (
            lambda p, t: p["classification"]["pflow_class"].__setitem__((0, 0), (t["particle_class"][0, 0] + 1) % 5),
            "val/final_obj_class_accuracy_micro",
        ),
        (lambda p, _: p["mask"]["pflow_node_valid"].__setitem__((0, 0, 0), ~p["mask"]["pflow_node_valid"][0, 0, 0]), "val/final_mask_exact_match"),
        (lambda p, _: p["regression"]["pflow_e"].__setitem__((0, 0), p["regression"]["pflow_e"][0, 0] + 1.0), "val/final_regression_e_abs_res"),
        (
            lambda p, _: p["regression"]["pflow_proxy_pt"].__setitem__((0, 0), p["regression"]["pflow_proxy_pt"][0, 0] + 1.0),
            "val/final_regression_pt_proxy_abs_res",
        ),
    ],
)
def test_control_one_perturbed_prediction_changes_the_value(perturb, key, mpflow):
    preds, targets = fixed_case(2)
    preds["classification"]["pflow_class"][0, 0] = targets["particle_class"][0, 0]  # a correct prediction to break
    preds["mask"]["pflow_node_valid"][0, 0] = targets["particle_node_valid"][0, 0]  # an exact mask to break
    ref = torch_logged(mpflow, {"final": preds}, targets)
    perturb(preds, targets)
    preds["classification"]["pflow_valid"] = preds["classification"]["pflow_class"] < 5
    ours = M.ClicMetrics("val")
    ours.update({"final": preds}, targets)
    assert ulps(ours.compute()[key], ref[key]) > 100 * max(tolerance(key), 1)
    assert_equal_metrics(ours.compute(), torch_logged(mpflow, {"final": preds}, targets))


def test_lightning_module_pools_layers_and_native_does_not(mpflow):
    """Deviation evidence: MPflow updates one metric object per layer key, so its value pools the layers."""
    good, targets = fixed_case(3)
    bad = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in good.items()}
    bad["classification"]["pflow_class"] = np.full_like(bad["classification"]["pflow_class"], 5)
    bad["classification"]["pflow_valid"] = np.zeros_like(bad["classification"]["pflow_valid"])
    preds = {"layer_0": bad, "final": good}
    ref = torch_logged(mpflow, preds, targets)
    ours = M.ClicMetrics("val")
    ours.update(preds, targets)
    values = ours.compute()
    assert values["val/layer_0_eff"] == 0.0 < values["val/final_eff"]
    assert ref["val/layer_0_eff"] == ref["val/final_eff"]  # pooled in the torch module
    assert min(values["val/layer_0_eff"], values["val/final_eff"]) < ref["val/final_eff"] < max(values["val/layer_0_eff"], values["val/final_eff"])


def test_weighted_mean_is_lightnings_epoch_reduction():
    mean = M.WeightedMean()
    assert math.isnan(mean.compute())
    mean.update(1.0, 4)
    mean.update(4.0, 2)
    assert mean.compute() == 2.0
