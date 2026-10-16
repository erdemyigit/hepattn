"""R5 full-model gates (docs/hgq/RETARGET_ORG_MAIN.md section 5.8): native vs the org/main torch model.

The model is org/main ``ac0da8f`` ``base.yaml`` + ``linformer.yaml`` at the test width (``mf_cases`` variant ``org``:
dim 32, every other key as the YAML: 6 encoder / 4 decoder layers, masked Linformer k = 64, phi sorting, PR #8 layers,
SwiGLU, classification from layer 0). Native is built by ``from_config``, torch by instantiating the same config; the
torch weights (seed 30, biases perturbed) are ported. Fixture: the first two CLIC dummy events, each with a padded
node (plan 1.2: a batch of one with every node valid takes org's unmasked Linformer path).

Gates:
- every output at 2x its measured max abs error (valid slots; the ``finfo.min`` padding sentinel exactly equal);
- the 4 attention-mask booleans exactly equal, on a fixture where every valid mask logit is >= 10x the logit
  tolerance away from ``logit(0.1)`` (asserted, so the exact check cannot pass or fail by luck);
- the 17 matched losses and the total at the ``maskformer.loss`` rows (1e-4 / 1e-3, ``test_objective_e2e.py``);
- controls, each > 10x the tolerance: no input sorting and ``ca_dense_sa`` (final mask logits), and the losses before
  bugs 1 and 2 (cross-event mask losses, double-softmax class cost; the mask_bce row of every layer).
Measured 2026-10-16 (macOS arm64 CPU, jax 0.11.2, torch 2.10.0): docs/hgq/port_log/R5.md.
"""

import math

import keras
import numpy as np
import pytest
import torch

from hepattn.keras_native import losses as L
from hepattn.keras_native import objective as O
from hepattn.models.matcher import Matcher
from hepattn.models.wrapper import ModelWrapper

from .golden.mf_cases import clic_dummy_batch, torch_model
from .mf_golden_utils import build_mf, flatten
from .objective_utils import flatten as flatten_tree
from .parity_utils import record_measurement
from .porting import port_maskformer

SEED = 30
LOGIT_THRESHOLD = math.log(0.1 / 0.9)  # sigmoid(logit) >= pred_threshold 0.1 (base.yaml)
SENTINEL = 1e30
# measured max |native - torch| per output (valid slots); tolerance = 2x
MEASURED = {
    "final.classification.pflow_class_prob": 4.470e-08,
    "final.classification.pflow_logit": 5.215e-08,
    "final.incidence.pflow_incidence": 4.657e-09,
    "final.mask.pflow_node_logit": 2.384e-06,
    "final.regression.pflow_proxy_regr": 1.118e-07,
    "final.regression.pflow_regr": 1.583e-08,
    "layer_0.classification.pflow_class_prob": 2.980e-08,
    "layer_0.classification.pflow_logit": 3.725e-08,
    "layer_0.mask.pflow_node_logit": 1.550e-06,
    "layer_1.classification.pflow_class_prob": 4.470e-08,
    "layer_1.classification.pflow_logit": 6.706e-08,
    "layer_1.mask.pflow_node_logit": 1.669e-06,
    "layer_2.classification.pflow_class_prob": 4.470e-08,
    "layer_2.classification.pflow_logit": 5.960e-08,
    "layer_2.mask.pflow_node_logit": 1.907e-06,
    "layer_3.classification.pflow_class_prob": 4.470e-08,
    "layer_3.classification.pflow_logit": 5.215e-08,
    "layer_3.mask.pflow_node_logit": 2.861e-06,
}
TOL = {key: 2 * value for key, value in MEASURED.items()}
LOSS_TOL = {"atol": 1e-4, "rtol": 1e-3}  # the maskformer.loss rows (test_objective_e2e.py)
ATTN_MASKS = [f"layer_{i}.attn_mask" for i in range(4)]
CONTROL_KEY = "final.mask.pflow_node_logit"


def torch_tensors(arrays: dict) -> dict:
    return {key: torch.from_numpy(value.astype(np.int64) if value.dtype == np.int32 else value) for key, value in arrays.items()}


@pytest.fixture(scope="module")
def batch():
    inputs, targets = clic_dummy_batch()
    assert (~inputs["node_valid"]).any(-1).all(), "every event needs a padded node (plan 1.2, batch of one all valid)"
    return inputs, targets


@pytest.fixture(scope="module")
def tmodel():
    model = torch_model("org", seed=SEED)
    model.matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)
    return model


def native(tmodel, inputs, **org_kwargs):
    model = build_mf("org", None, inputs, **org_kwargs)
    port_maskformer(tmodel, model)
    return model


@pytest.fixture(scope="module")
def outputs(batch, tmodel):
    inputs, _ = batch
    nmodel = native(tmodel, inputs)
    with torch.no_grad():
        out_t = {key: value.numpy() for key, value in flatten_tree(tmodel(torch_tensors(inputs))).items()}
    return out_t, flatten(nmodel(inputs, training=False)), nmodel


def valid(ref: np.ndarray) -> np.ndarray:
    return np.abs(ref) < SENTINEL


def test_outputs_and_masks_equal_torch(outputs):
    out_t, out_n, _ = outputs
    assert set(out_t) == set(out_n) == set(MEASURED) | set(ATTN_MASKS)
    for key in ATTN_MASKS:
        assert out_t[key].dtype == bool and np.array_equal(out_t[key], out_n[key]), f"{key}: {int((out_t[key] != out_n[key]).sum())} flips"
        assert 0.05 < out_t[key].mean() < 0.95, f"{key}: mask fraction {out_t[key].mean():.3f}"
    for key, tol in TOL.items():
        ref, got = out_t[key].astype(np.float64), np.asarray(out_n[key], dtype=np.float64)
        keep = valid(ref)
        assert np.array_equal(ref[~keep], got[~keep]), f"{key}: padding sentinel differs"
        err = float(np.abs(ref[keep] - got[keep]).max())
        record_measurement({"tag": "maskformer.org_model", "config": key, "max_abs_err": err, "atol": tol, "rtol": 0.0})
        assert err <= tol, f"{key}: {err:.3e} > {tol:.3e}"


@pytest.mark.parametrize("layer", range(4))
def test_mask_logits_are_far_from_the_threshold(outputs, layer):
    """The exact boolean check is meaningful: every valid logit is >= 10x its tolerance from logit(0.1)."""
    out_t, _, _ = outputs
    ref = out_t[f"layer_{layer}.mask.pflow_node_logit"].astype(np.float64)
    margin = float(np.abs(ref[valid(ref)] - LOGIT_THRESHOLD).min())
    assert margin >= 10 * TOL[f"layer_{layer}.mask.pflow_node_logit"], f"layer {layer}: margin {margin:.3e}"


def matched_losses(tmodel, nmodel, inputs, targets) -> tuple[dict, dict, float, float]:
    """(torch, native) flattened matched losses and totals; torch with scipy (lap1015 is not built locally)."""
    with torch.no_grad():
        _, _, losses_t = tmodel.loss(tmodel(torch_tensors(inputs)), torch_tensors(targets))
    total_n, losses_n, _ = O.matched_loss(nmodel(inputs, training=False), targets, O.clic_specs())
    log = type("Stub", (), {"model": tmodel, "log": staticmethod(lambda *_a, **_k: None)})()
    total_t = float(ModelWrapper.aggregate_losses(log, losses_t, stage="train"))
    flat_t = {key: float(value) for key, value in flatten_tree(losses_t).items()}
    flat_n = {key: float(np.asarray(value)) for key, value in flatten_tree(losses_n).items()}
    return flat_t, flat_n, total_t, float(total_n)


def loss_tol(ref: float) -> float:
    return LOSS_TOL["atol"] + LOSS_TOL["rtol"] * abs(ref)


def test_matched_losses_equal_torch(batch, tmodel, outputs):
    inputs, targets = batch
    flat_t, flat_n, total_t, total_n = matched_losses(tmodel, outputs[2], inputs, targets)
    assert set(flat_t) == set(flat_n) and len(flat_t) == 17, sorted(set(flat_t) ^ set(flat_n))
    for key, ref in flat_t.items():
        assert np.isfinite(ref), key
        assert abs(flat_n[key] - ref) <= loss_tol(ref), f"{key}: native {flat_n[key]} vs torch {ref}"
    assert abs(total_n - total_t) <= loss_tol(total_t), (total_n, total_t)


@pytest.mark.parametrize(
    "control", [{"sort": False}, {"decoder_layer_overrides": {"query_update_order": "ca_dense_sa"}}], ids=["no_sort", "ca_dense_sa"]
)
def test_model_controls_exceed_the_tolerance(batch, tmodel, outputs, control):
    out_t, _, _ = outputs
    ref = out_t[CONTROL_KEY].astype(np.float64)
    keep = valid(ref)
    got = np.asarray(flatten(native(tmodel, batch[0], **control)(batch[0], training=False))[CONTROL_KEY], dtype=np.float64)
    diff = float(np.abs(ref[keep] - got[keep]).max())
    assert diff > 10 * TOL[CONTROL_KEY], f"{control}: {CONTROL_KEY} moves by only {diff:.3e}"


def _old_cross_event_mean(term, object_valid_mask):
    """Mean of term [b (pad event), q (object event), p] over b and the valid (q, p): the removed formula."""
    valid_qp = keras.ops.cast(object_valid_mask, "float32")[None]
    batch = keras.ops.cast(keras.ops.shape(term)[0], "float32")
    return keras.ops.sum(term * valid_qp) / keras.ops.maximum(batch * keras.ops.sum(valid_qp), 1.0)


def old_mask_bce(logits, targets, object_valid_mask, input_pad_mask, sample_weight=None):
    loss = L.bce_with_logits(L.f32(logits), L.f32(targets))
    if sample_weight is not None:
        loss *= L.f32(sample_weight)
    pad = L.f32(input_pad_mask)
    term = keras.ops.einsum("qpn,bn->bqp", loss, pad) / keras.ops.maximum(keras.ops.sum(pad, -1), 1.0)[:, None, None]
    return _old_cross_event_mean(term, object_valid_mask)


def old_mask_dice(logits, targets, object_valid_mask, input_pad_mask):
    probs, pad, tgt = keras.ops.sigmoid(L.f32(logits)), L.f32(input_pad_mask), L.f32(targets)
    num = 2 * keras.ops.einsum("qpn,bn->bqp", probs * tgt, pad)
    den = keras.ops.einsum("qpn,bn->bqp", probs, pad) + keras.ops.sum(tgt, -1)[None]
    return _old_cross_event_mean(1 - (num + 1) / (den + 1), object_valid_mask)


def test_losses_before_bugs_1_and_2_exceed_the_tolerance(batch, tmodel, outputs, monkeypatch):
    """Control: the pre-a4263c1 cross-event mask losses and the pre-PR #3 double-softmax class cost.

    Real-like targets (no target 1 at padded nodes): with the dummy targets the old formula is inf (S4.md), which
    the cross-event einsum turns into NaN, and a NaN total would pass a ``>`` check by failing it for another reason.
    """
    inputs, targets = batch
    targets = dict(targets)
    targets["particle_node_valid"] = targets["particle_node_valid"] & inputs["node_valid"][:, None, :]
    cost = L.object_ce_cost
    monkeypatch.setattr(L, "mask_bce_loss", old_mask_bce)
    monkeypatch.setattr(L, "mask_dice_loss", old_mask_dice)
    monkeypatch.setattr(L, "object_ce_cost", lambda logits, target_class: cost(keras.ops.softmax(logits, -1), target_class))
    flat_t, flat_n, total_t, total_n = matched_losses(tmodel, outputs[2], inputs, targets)
    gaps = {key: abs(flat_n[key] - ref) / loss_tol(ref) for key, ref in flat_t.items()}
    record_measurement({"tag": "maskformer.org_model.old_losses", "config": "gaps", "max_abs_err": max(gaps.values())})
    assert np.isfinite(total_n) and np.isfinite(total_t), (total_n, total_t)
    # measured: mask_bce 26-28x its row tolerance at every layer; object_ce 4.8-5.7x, mask_dice 1.3-1.7x; the total
    # 7.9x (its rtol 1e-3 of ~50 is loose). The mask_bce rows are the gate.
    bce = {key: gap for key, gap in gaps.items() if key.endswith("mask_bce")}
    assert len(bce) == 5 and min(bce.values()) > 10, gaps
    assert total_n != total_t
