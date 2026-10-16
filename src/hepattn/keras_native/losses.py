"""``keras.ops`` ports of the cost and loss functions in ``hepattn.models.loss`` that the CLIC config uses.

The semantics are those of org/main ``ac0da8f``, which includes ``a4263c1`` (per-event mask losses) and
PR #3 (``a2bc9c7``, the class cost reads logits).

Every function takes arrays with static shapes and returns float32. Boolean indexing in torch
(``x[mask]``, a data-dependent shape) becomes ``sum(where(mask, x, 0)) / maximum(count, 1)``. One
deliberate difference stays for ``kl_div_loss`` and ``regr_l1_loss``: when the mask selects nothing,
torch takes the mean of an empty tensor (NaN) and these functions return 0. ``mask_bce_loss`` and
``mask_dice_loss`` have no such difference: torch now divides by ``clamp_min(1.0)`` and also gives 0.

- ``mask_bce_loss`` and ``mask_dice_loss`` (``a4263c1``): each object is reduced over the valid
  constituents of its OWN event, then the mean runs over the valid objects
  (``loss._mean_over_valid_objects``). Padded nodes are multiplied by that event's pad mask (0), so a
  ``finfo.min`` logit with target 1 at a padded node gives a finite loss.
- ``object_ce_cost`` (PR #3) receives LOGITS and applies one softmax.

Shapes: ``B`` batch, ``P`` predicted objects (queries), ``T`` target objects, ``N`` constituents (nodes),
``C`` classes (including the null class), ``F`` regression fields.
"""

from keras import ops

DEFAULT_EPS = 1e-8


def f32(x):
    return ops.cast(x, "float32")


def bce_with_logits(logits, targets):
    """Elementwise ``F.binary_cross_entropy_with_logits(..., reduction="none")``: ``(1 - t) x + softplus(-x)``.

    This is the torch formula. It stays finite for ``x = finfo.min`` and ``t = 0`` (value 0), and gives
    ``|x|`` (about 3.4e38) for ``t = 1``.
    """
    return (1.0 - targets) * logits + ops.softplus(-logits)


def _count(mask):
    return ops.sum(f32(mask))


# --------------------------------------------------------------------------------------------- costs
# Costs have shape [B, P, T]. Callers apply stop_gradient; invalid targets are dropped by the matcher.


def object_ce_cost(logits, target_class):
    """``loss.object_ce_cost``: ``-softmax(logits)[b, p, target_class[b, t]]``.

    Args:
        logits: [B, P, C] class logits (``ObjectClassificationTask.cost`` passes ``*_logit``).
        target_class: [B, T] integer class labels in [0, C).
    """
    probs = ops.softmax(f32(logits), axis=-1)
    b, p, _ = probs.shape
    index = ops.broadcast_to(ops.expand_dims(ops.cast(target_class, "int32"), 1), (b, p, target_class.shape[1]))
    return -ops.take_along_axis(probs, index, axis=2)


def mask_dice_cost(logits, targets, input_pad_mask):
    """``loss.mask_dice_cost`` with ``input_pad_mask``. logits [B, P, N], targets [B, T, N], pad [B, N]."""
    probs = ops.sigmoid(f32(logits)) * ops.expand_dims(f32(input_pad_mask), 1)
    targets = f32(targets)
    numerator = 2.0 * ops.einsum("bnc,bmc->bnm", probs, targets)
    denominator = ops.expand_dims(ops.sum(probs, axis=-1), 2) + ops.expand_dims(ops.sum(targets, axis=-1), 1)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def kl_div_cost(pred, true, eps=DEFAULT_EPS):
    """``loss.kl_div_cost``: ``mean_n(-true[b, t, n] * log(pred[b, p, n] + eps))``. pred [B, P, N], true [B, T, N].

    Torch builds the [B, P, T, N] product; the einsum gives the same sum without it.
    """
    log_pred = ops.log(f32(pred) + eps)
    return -ops.einsum("bpn,btn->bpt", log_pred, f32(true)) / pred.shape[-1]


def regr_l1_cost(pred, target):
    """``IncidenceBasedRegressionTask.new_cost`` with ``l1``, before the cost weight: ``mean_f |pred - target|``.

    pred [B, P, F], target [B, T, F]. NaN-padded target rows give NaN columns, as in torch; the matcher
    only reads the valid target rows.
    """
    diff = ops.expand_dims(f32(pred), 2) - ops.expand_dims(f32(target), 1)
    return ops.mean(ops.abs(diff), axis=-1)


# --------------------------------------------------------------------------------------------- losses


def object_ce_loss(logits, target_class, weight=None):
    """``F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1), weight=weight)`` (``loss.object_ce_loss``).

    With weights this is ``sum_i w[y_i] l_i / sum_i w[y_i]``, not the plain mean. Torch ignores the mask
    argument here, so this function has none.

    Args:
        logits: [B, P, C].
        target_class: [B, P] integer labels in [0, C).
        weight: optional [C] class weights.
    """
    log_probs = ops.log_softmax(f32(logits), axis=-1)
    labels = ops.expand_dims(ops.cast(target_class, "int32"), -1)
    nll = -ops.squeeze(ops.take_along_axis(log_probs, labels, axis=-1), -1)
    if weight is None:
        return ops.mean(nll)
    w = ops.take(f32(ops.convert_to_tensor(weight)), ops.cast(target_class, "int32"))
    return ops.sum(w * nll) / ops.sum(w)


def _mean_over_valid_objects(per_object, object_valid_mask):
    """``loss._mean_over_valid_objects``: ``sum(per_object * valid) / max(sum(valid), 1)``. per_object [B, P]."""
    weight = f32(object_valid_mask)
    return ops.sum(per_object * weight) / ops.maximum(ops.sum(weight), 1.0)


def mask_bce_loss(logits, targets, object_valid_mask, input_pad_mask, sample_weight=None):
    """``loss.mask_bce_loss`` with both masks: per object, the mean over its own event's valid nodes.

    Args:
        logits: [B, P, N] (outputs already permuted to target order).
        targets: [B, P, N] binary masks.
        object_valid_mask: [B, P] bool.
        input_pad_mask: [B, N] bool.
        sample_weight: optional [B, P, N] elementwise BCE weight.
    """
    loss = bce_with_logits(f32(logits), f32(targets))
    if sample_weight is not None:
        loss *= f32(sample_weight)
    pad = f32(input_pad_mask)
    counts = ops.maximum(ops.sum(pad, axis=-1, keepdims=True), 1.0)  # [B, 1]
    per_object = ops.sum(loss * ops.expand_dims(pad, 1), axis=-1) / counts
    return _mean_over_valid_objects(per_object, object_valid_mask)


def mask_dice_loss(logits, targets, object_valid_mask, input_pad_mask):
    """``loss.mask_dice_loss`` with both masks: per-object DICE, mean over the valid objects.

    The probabilities are multiplied by the event's pad mask; the target sum is not masked (torch
    semantics). Shapes as in ``mask_bce_loss``.
    """
    probs = ops.sigmoid(f32(logits)) * ops.expand_dims(f32(input_pad_mask), 1)
    targets = f32(targets)
    numerator = 2.0 * ops.sum(probs * targets, axis=-1)
    denominator = ops.sum(probs, axis=-1) + ops.sum(targets, axis=-1)
    per_object = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return _mean_over_valid_objects(per_object, object_valid_mask)


def kl_div_loss(pred, true, mask, eps=DEFAULT_EPS):
    """``loss.kl_div_loss``: mean of ``-true * log(pred + eps)`` over ``mask``. All [B, P, N]."""
    loss = -f32(true) * ops.log(f32(pred) + eps)
    return ops.sum(ops.where(mask, loss, 0.0)) / ops.maximum(_count(mask), 1.0)


def regr_l1_loss(pred, target, object_valid_mask):
    """``F.l1_loss(pred[mask], target[mask])``: the mean over valid objects AND fields. pred, target [B, P, F].

    ``target`` should be finite (callers replace NaN padding first). For this l1 form JAX would still give
    finite gradients with NaN targets, because the ``abs`` JVP is a select (``jax/_src/lax/lax.py``
    ``_abs_jvp_rule``); the replacement protects residual-weighted losses, which give NaN gradients.
    """
    diff = ops.abs(f32(pred) - f32(target))
    mask = ops.broadcast_to(ops.expand_dims(object_valid_mask, -1), diff.shape)
    return ops.sum(ops.where(mask, diff, 0.0)) / ops.maximum(_count(mask), 1.0)


__all__ = [
    "bce_with_logits",
    "kl_div_cost",
    "kl_div_loss",
    "mask_bce_loss",
    "mask_dice_cost",
    "mask_dice_loss",
    "object_ce_cost",
    "object_ce_loss",
    "regr_l1_cost",
    "regr_l1_loss",
]
