"""R1: bugs 1 and 2 of org/main in keras_native (docs/hgq/RETARGET_ORG_MAIN.md section 3.3).

Bug 1 (``a4263c1``): ``mask_bce`` / ``mask_dice`` are normalised per event. Before, each valid object was
weighted by EVERY event's node mask (a cross-event broadcast). Bug 2 (PR #3, ``a2bc9c7``): the multi-class
matching cost takes the logits and applies one softmax; before, it soft-maxed ``*_class_prob`` again.

The reference is the org/main torch code in this process (``hepattn.models.loss`` / ``task``). The "before"
values are from native at ``5db27a8`` (``docs/hgq/port_log/R1.md``); ``_old_*`` are test-local copies of the
removed native formula (they reproduce the old module to 2e-7 on F1), used as controls.

Tolerances are 2x the values measured on this implementation (macOS, JAX CPU); see R1.md.
"""

import jax
import numpy as np
import pytest
import torch

from hepattn.keras_native import losses as L
from hepattn.keras_native import matching
from hepattn.keras_native import objective as O
from hepattn.models import loss as TL
from hepattn.models.dense import Dense
from hepattn.models.matcher import Matcher
from hepattn.models.task import ObjectClassificationTask

# measured |native - torch| on F1: mask_bce 2.38e-7, mask_dice 1.19e-7, class cost 2.38e-7
BCE_ATOL = 4.8e-7
DICE_ATOL = 2.4e-7
CLS_COST_ATOL = 4.8e-7
# measured |L(batch) - weighted mean of L(event)| for native: mask_bce 3.35e-7, mask_dice 2.35e-8
DECOMP_ATOL = {"mask_bce": 6.7e-7, "mask_dice": 4.7e-8}
# measured max |old(B=1) - new(B=1)| over the 3 events: mask_bce 1.40e-7, mask_dice 4.35e-8
SINGLE_EVENT_ATOL = {"mask_bce": 2.8e-7, "mask_dice": 8.7e-8}

FMIN = np.finfo(np.float32).min


def _t(x):
    return torch.from_numpy(np.asarray(x))


def f1() -> dict:
    """Fixture F1 (``specgap/probe_losses.py``, seed 0): 3 events with different padding and object counts."""
    rng = np.random.default_rng(0)
    b, p, n = 3, 150, 160
    pad = np.arange(n)[None] < np.array([120, 37, 159])[:, None]
    ov = np.arange(p)[None] < np.array([40, 9, 60])[:, None]
    logits = rng.normal(size=(b, p, n)).astype(np.float32) * 3
    logits = np.where(pad[:, None], logits, FMIN)  # as ObjectHitMaskTask.forward
    tgt = ((rng.random((b, p, n)) < 0.05) & pad[:, None] & ov[..., None]).astype(np.float32)
    class_logits = rng.normal(size=(b, p, 6)).astype(np.float32)
    target_class = np.where(ov, rng.integers(0, 5, (b, p)), 5)
    return {"logits": logits, "tgt": tgt, "ov": ov, "pad": pad, "class_logits": class_logits, "target_class": target_class}


@pytest.fixture(scope="module")
def data():
    return f1()


# ----------------------------------------------------------------------------------------- references


def _bce64(logits, tgt):
    x = logits.astype(np.float64)
    return (1.0 - tgt) * x + np.logaddexp(0.0, -x)


def _sigmoid64(logits):
    return 0.5 * (1.0 + np.tanh(0.5 * logits.astype(np.float64)))


def _per_event_bce(logits, tgt, ov, pad):
    """Independent float64 reference of the a4263c1 formula."""
    per = (_bce64(logits, tgt) * pad[:, None]).sum(-1) / np.maximum(pad.sum(-1), 1)[:, None]
    return float((per * ov).sum() / max(ov.sum(), 1))


def _old_cross_event(term, ov):
    """Mean of term [b (pad event), q (object event), p] over b and the valid (q, p)."""
    valid = np.broadcast_to(ov[None], term.shape)
    return float(term[valid].sum() / max(term.shape[0] * ov.sum(), 1))


def _old_bce(logits, tgt, ov, pad, dtype=np.float64):
    """The removed native/torch formula (cross-event broadcast)."""
    loss = (1 - tgt.astype(dtype)) * logits.astype(dtype) + np.logaddexp(dtype(0), -logits.astype(dtype))
    padf = pad.astype(dtype)
    term = np.einsum("qpn,bn->bqp", loss, padf) / np.maximum(padf.sum(-1), 1)[:, None, None]
    return _old_cross_event(term, ov)


def _old_dice(logits, tgt, ov, pad):
    probs, padf = _sigmoid64(logits), pad.astype(np.float64)
    num = 2 * np.einsum("qpn,bn->bqp", probs * tgt, padf)
    den = np.einsum("qpn,bn->bqp", probs, padf) + tgt.sum(-1)[None]
    return _old_cross_event(1 - (num + 1) / (den + 1), ov)


def _native(name, logits, tgt, ov, pad):
    return float(getattr(L, f"{name}_loss")(logits, tgt, ov, pad))


def _torch(name, logits, tgt, ov, pad, **kw):
    return getattr(TL, f"{name}_loss")(_t(logits), _t(tgt), object_valid_mask=_t(ov), input_pad_mask=_t(pad), **kw).item()


_OLD = {"mask_bce": _old_bce, "mask_dice": _old_dice}


# ------------------------------------------------------------------------------------------------ bug 1


def test_b1a_mask_losses_equal_org_torch(data):
    """Before (native 5db27a8): 1.3090 / 0.89237. After = torch: 1.4038 / 0.89392."""
    args = (data["logits"], data["tgt"], data["ov"], data["pad"])
    bce_t, bce_n = _torch("mask_bce", *args), _native("mask_bce", *args)
    dice_t, dice_n = _torch("mask_dice", *args), _native("mask_dice", *args)
    print(f"mask_bce torch {bce_t:.7f} native {bce_n:.7f}; mask_dice torch {dice_t:.7f} native {dice_n:.7f}")
    assert abs(bce_n - bce_t) <= BCE_ATOL
    assert abs(dice_n - dice_t) <= DICE_ATOL
    assert abs(_per_event_bce(*args) - bce_t) <= BCE_ATOL  # independent float64 reference
    # control: the old formula is >= 10x the tolerance away (measured gaps 0.0948 and 1.55e-3)
    assert abs(_old_bce(*args) - bce_t) >= 10 * BCE_ATOL
    assert abs(_old_dice(*args) - dice_t) >= 10 * DICE_ATOL
    assert abs(_old_bce(*args) - bce_t) > 0.09
    assert abs(_old_dice(*args) - dice_t) > 1.5e-3


@pytest.mark.parametrize("name", ["mask_bce", "mask_dice"])
def test_b1b_batch_loss_is_the_object_weighted_mean_of_event_losses(data, name):
    """Per-event normalisation makes events independent: L(batch) = sum_b n_b L(event b) / sum_b n_b."""
    args = (data["logits"], data["tgt"], data["ov"], data["pad"])
    n = data["ov"].sum(1)

    def combined(fn):
        parts = np.array([fn(*(a[b : b + 1] for a in args)) for b in range(len(n))])
        return float((n * parts).sum() / n.sum())

    full = _native(name, *args)
    gap = abs(full - combined(lambda *a: _native(name, *a)))
    old_gap = abs(_OLD[name](*args) - combined(_OLD[name]))
    print(f"{name}: native decomposition gap {gap:.2e}; old formula gap {old_gap:.2e}")
    assert gap <= DECOMP_ATOL[name]
    assert old_gap >= 10 * DECOMP_ATOL[name]  # control: the old formula couples events


@pytest.mark.parametrize("name", ["mask_bce", "mask_dice"])
def test_b1c_single_event_old_and_new_formulas_agree(data, name):
    """Control: with one event there is no cross-event term, so B1-a/B1-b react to batch coupling only."""
    args = (data["logits"], data["tgt"], data["ov"], data["pad"])
    worst = 0.0
    for b in range(3):
        one = tuple(a[b : b + 1] for a in args)
        worst = max(worst, abs(_native(name, *one) - _OLD[name](*one)))
    print(f"{name}: max |old - new| at B=1: {worst:.2e}")
    assert worst <= SINGLE_EVENT_ATOL[name]


def test_b1d_target_one_at_padded_nodes_is_finite_and_equals_torch(data):
    """A finfo.min logit with target 1 at a padded node (dummy data). Before: native and old torch gave inf."""
    tgt = data["tgt"].copy()
    pad, ov, logits = data["pad"], data["ov"], data["logits"]
    padded_target = np.broadcast_to(~pad[:, None, :], tgt.shape) & ov[..., None]
    tgt[padded_target] = 1.0
    sample_weight = tgt + 1.0 * (1.0 - tgt)  # ObjectHitMaskTask.loss with null_weight 1
    ref = _torch("mask_bce", logits, tgt, ov, pad, sample_weight=_t(sample_weight))
    got = float(L.mask_bce_loss(logits, tgt, ov, pad, sample_weight=sample_weight))
    assert np.isfinite(ref)
    assert abs(got - ref) <= BCE_ATOL
    dice_ref, dice = _torch("mask_dice", logits, tgt, ov, pad), _native("mask_dice", logits, tgt, ov, pad)
    assert np.isfinite(dice)
    assert abs(dice - dice_ref) <= DICE_ATOL  # measured 0 (0.9179240 both); DICE_ATOL from B1-a
    # control: the old formula in float32 (the native dtype) overflows
    assert np.isinf(_old_bce(logits, tgt, ov, pad, dtype=np.float32))
    # padded nodes get an exactly zero gradient
    grad = np.asarray(jax.grad(lambda x: L.mask_bce_loss(x, tgt, ov, pad, sample_weight=sample_weight))(logits))
    assert np.isfinite(grad).all()
    assert np.abs(grad[np.broadcast_to(~pad[:, None, :], grad.shape)]).max() == 0.0
    assert np.abs(grad).max() > 0.0


@pytest.mark.parametrize("name", ["mask_bce", "mask_dice"])
def test_b1e_no_valid_object_gives_zero_as_torch(data, name):
    """Regression pin: torch now divides by clamp_min(1) and gives 0 as well (before: torch NaN, native 0)."""
    args = (data["logits"], data["tgt"], np.zeros_like(data["ov"]), data["pad"])
    assert _torch(name, *args) == 0.0
    assert _native(name, *args) == 0.0


@pytest.mark.parametrize("name", ["mask_bce", "mask_dice"])
def test_b1f_finite_logits_at_padded_nodes_change_nothing(data, name):
    """Padded nodes are removed by the event's pad mask, not by the finfo.min logit: replacing those logits
    with finite values leaves native exactly unchanged and equal to torch. Control: without the pad mask
    the value moves (measured: mask_bce 1.948, mask_dice 0.9108, against 1.4038 and 0.8939).
    """
    rng = np.random.default_rng(1)
    pad, ov, tgt = data["pad"], data["ov"], data["tgt"]
    logits = data["logits"].copy()
    padded = np.broadcast_to(~pad[:, None, :], logits.shape)
    logits[padded] = (3 * rng.normal(size=int(padded.sum()))).astype(np.float32)
    got = _native(name, logits, tgt, ov, pad)
    assert got == _native(name, data["logits"], tgt, ov, pad)  # exact padding invariance
    assert abs(got - _torch(name, logits, tgt, ov, pad)) <= {"mask_bce": BCE_ATOL, "mask_dice": DICE_ATOL}[name]
    if name == "mask_bce":
        unmasked = float(((_bce64(logits, tgt).sum(-1) / pad.sum(-1)[:, None]) * ov).sum() / ov.sum())
    else:
        probs = _sigmoid64(logits)
        per = 1 - (2 * (probs * tgt).sum(-1) + 1) / (probs.sum(-1) + tgt.sum(-1) + 1)
        unmasked = float((per * ov).sum() / ov.sum())
    assert abs(unmasked - got) > 0.01


@pytest.mark.parametrize("objects", ["with_objects", "no_objects"])
def test_b1g_event_without_valid_nodes_is_finite_and_equals_torch(data, objects):
    """One event of the batch has no valid node (fixed-shape batch padded with an empty event).

    torch divides that event's node sum by ``valid_counts.clamp_min(1)``, so its objects contribute 0 to
    mask_bce. Measured: mask_bce 1.2725188 / 1.3870456 and mask_dice 0.8231073 / 0.8971870 (with / without
    valid objects in that event), equal to torch within the B1-a tolerances. Control: without the clamp the
    per-event bce is 0/0 = NaN, and NaN * 0 poisons the batch even when the event has no valid object.
    """
    pad = data["pad"].copy()
    pad[1] = False
    ov = data["ov"].copy()
    if objects == "no_objects":
        ov[1] = False
    logits = np.where(pad[:, None], data["logits"], FMIN).astype(np.float32)
    tgt = (data["tgt"] * pad[:, None]).astype(np.float32)
    for name, atol in (("mask_bce", BCE_ATOL), ("mask_dice", DICE_ATOL)):
        got, ref = _native(name, logits, tgt, ov, pad), _torch(name, logits, tgt, ov, pad)
        print(f"{objects} {name}: torch {ref:.7f} native {got:.7f}")
        assert np.isfinite(got)
        assert abs(got - ref) <= atol
    with np.errstate(invalid="ignore"):
        per = (_bce64(logits, tgt) * pad[:, None]).sum(-1) / pad.sum(-1)[:, None]  # no clamp
    assert np.isnan((per * ov).sum() / max(ov.sum(), 1))


@pytest.mark.parametrize("name", ["mask_bce", "mask_dice"])
def test_b1h_target_one_and_finite_logits_at_padded_nodes_equal_torch(data, name):
    """Dummy targets (1 at the padded nodes of valid objects) together with finite padded logits.

    This is the only case where torch's pad multiply on the dice numerator matters (B1-f has target 0 at
    padded nodes, B1-d has ``finfo.min`` logits there). Measured: mask_bce 1.4038032 (torch 1.4038030),
    mask_dice 0.9179240 (= torch). The value is exactly the one with ``finfo.min`` padded logits. Control: a
    dice numerator with unmasked probabilities gives 0.7080; bce without the pad mask moves by more than 0.01.
    """
    rng = np.random.default_rng(1)
    pad, ov = data["pad"], data["ov"]
    padded = np.broadcast_to(~pad[:, None, :], data["logits"].shape)
    logits = data["logits"].copy()
    logits[padded] = (3 * rng.normal(size=int(padded.sum()))).astype(np.float32)
    tgt = data["tgt"].copy()
    tgt[padded & ov[..., None]] = 1.0
    got = _native(name, logits, tgt, ov, pad)
    assert abs(got - _torch(name, logits, tgt, ov, pad)) <= {"mask_bce": BCE_ATOL, "mask_dice": DICE_ATOL}[name]
    assert got == _native(name, data["logits"], tgt, ov, pad)
    if name == "mask_bce":
        unmasked = float(((_bce64(logits, tgt).sum(-1) / pad.sum(-1)[:, None]) * ov).sum() / ov.sum())
    else:
        probs = _sigmoid64(logits)
        den = (probs * pad[:, None]).sum(-1) + tgt.sum(-1)
        per = 1 - (2 * (probs * tgt).sum(-1) + 1) / (den + 1)  # numerator not masked
        unmasked = float((per * ov).sum() / ov.sum())
    print(f"{name}: native {got:.7f}; control {unmasked:.4f}")
    assert abs(unmasked - got) > 0.01


# ------------------------------------------------------------------------------------------------ bug 2


def _torch_classification_task() -> ObjectClassificationTask:
    return ObjectClassificationTask(
        name="classification",
        input_object="query",
        output_object="pflow",
        target_object="particle",
        num_classes=5,
        losses={"object_ce": 2},
        costs={"object_ce": 2},
        net=Dense(input_size=8, output_size=6),
        null_weight=0.5,
        class_weights=[1.0, 3.0, 8.0, 1.5, 1.0],
    )


def _class_outputs(logits):
    return {"pflow_logit": logits, "pflow_class_prob": torch.softmax(_t(logits), -1).numpy()}


def _class_costs(logits, target_class):
    """(torch task cost, native spec cost, double-softmax mutant) for the CLIC classification task."""
    outputs = _class_outputs(logits)
    ref = _torch_classification_task().cost({k: _t(v) for k, v in outputs.items()}, {"particle_class": _t(target_class)})["object_ce"].numpy()
    spec = O.clic_specs()[0]
    got = np.asarray(spec.cost(outputs, {"particle_class": target_class})["object_ce"])
    mutant = spec.costs["object_ce"] * np.asarray(L.object_ce_cost(outputs["pflow_class_prob"], target_class))
    return ref, got, mutant


def test_b2a_class_cost_equals_org_torch(data):
    """Before (native 5db27a8): max diff 0.5626 per unit weight (1.125 at weight 2)."""
    ref, got, mutant = _class_costs(data["class_logits"], data["target_class"])
    err, mutant_err = np.abs(got - ref).max(), np.abs(mutant - ref).max()
    print(f"class cost: max |native - torch| {err:.2e}; double-softmax mutant {mutant_err:.3f}")
    assert err <= CLS_COST_ATOL
    assert mutant_err >= 10 * CLS_COST_ATOL
    assert mutant_err > 1.0


def test_b2b_class_cost_keeps_the_dynamic_range(data):
    """Ptp of the cost: torch 1.761 (0.881 per unit weight); the double softmax compresses it 4.7x."""
    ref, got, mutant = _class_costs(data["class_logits"], data["target_class"])
    print(f"ptp: torch {np.ptp(ref):.4f}, native {np.ptp(got):.4f}, mutant {np.ptp(mutant):.4f}")
    assert abs(np.ptp(got) - np.ptp(ref)) <= 2 * CLS_COST_ATOL
    assert np.ptp(ref) >= 4 * np.ptp(mutant)


def test_b2c_two_by_two_assignment_follows_the_single_softmax():
    """2x2 problem where the double softmax lets the mask cost win and flips the assignment.

    Totals of ``2 ce + dice``: single softmax, diagonal -1.9696 (picked); double softmax, swap 0.0747.
    """
    logits = np.zeros((1, 2, 6), np.float32)
    logits[0, 0, 0] = logits[0, 1, 1] = 5.0
    target_class = np.array([[0, 1]])
    dice = np.array([[[0.95, 0.30], [0.30, 0.95]]], np.float32)
    valid = np.ones((1, 2), bool)

    ref, got, mutant = _class_costs(logits, target_class)
    idx_torch = Matcher(default_solver="scipy", adaptive_solver=False)(_t(ref + dice), _t(valid)).numpy()
    idx_native = matching.host_match(got + dice, valid)
    idx_mutant = matching.host_match(mutant + dice, valid)
    total = float((got + dice)[0, [0, 1], [0, 1]].sum())
    total_mutant = float((mutant + dice)[0, [1, 0], [0, 1]].sum())
    print(f"single softmax: idx {idx_native[0]}, total {total:.4f}; double softmax: idx {idx_mutant[0]}, total {total_mutant:.4f}")
    np.testing.assert_array_equal(idx_torch, [[0, 1]])
    np.testing.assert_array_equal(idx_native, [[0, 1]])
    np.testing.assert_array_equal(idx_mutant, [[1, 0]])  # control
    np.testing.assert_allclose([total, total_mutant], [-1.9696, 0.0747], atol=1e-4)
