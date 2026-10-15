"""``keras.ops`` ports of the cost and loss functions in ``hepattn.models.loss`` that the CLIC config uses.

Every function takes arrays with static shapes and returns float32. Boolean indexing in torch
(``x[mask]``, a data-dependent shape) becomes ``sum(where(mask, x, 0)) / maximum(count, 1)``. The
normalisation is the torch one, with one deliberate difference: when the mask selects nothing, torch
takes the mean of an empty tensor (NaN) and these functions return 0.

Two torch semantics are kept on purpose, because the hybrid trains with them:

- ``mask_bce_loss`` and ``mask_dice_loss`` index the objects to ``[Nv, N]`` and then multiply by
  ``input_pad_mask.unsqueeze(1)`` (``[B, 1, N]``). That broadcasts to ``[B, Nv, N]``: each valid object of
  each event is weighted by the node mask of EVERY event ``b`` and divided by ``b``'s node count, and the
  mean runs over ``B * Nv`` terms. Here this is an einsum over a second batch axis. With a finfo.min logit
  at a padded node and a target of 1 there, the cross-event term overflows to ``inf``
  (``docs/hgq/PARITY.md``, "mask_bce can be inf").
- ``object_ce_cost`` receives the class PROBABILITIES (``ObjectClassificationTask.cost`` passes
  ``*_class_prob``) and applies a softmax again.

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


def object_ce_cost(class_probs, target_class):
    """``loss.object_ce_cost``: ``-softmax(class_probs)[b, p, target_class[b, t]]``.

    Args:
        class_probs: [B, P, C]. The task passes probabilities, which are soft-maxed again (torch semantics).
        target_class: [B, T] integer class labels in [0, C).
    """
    probs = ops.softmax(f32(class_probs), axis=-1)
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


def _cross_event_object_mean(term, object_valid_mask):
    """Mean of ``term`` [B (pad event b), B (object event q), P] over b and the valid objects (q, p).

    This is the torch broadcast of ``x[object_valid_mask] * input_pad_mask.unsqueeze(1)`` (module docstring).
    """
    batch = object_valid_mask.shape[0]
    valid = ops.broadcast_to(ops.expand_dims(object_valid_mask, 0), term.shape)
    total = ops.sum(ops.where(valid, term, 0.0))
    return total / ops.maximum(batch * _count(object_valid_mask), 1.0)


def mask_bce_loss(logits, targets, object_valid_mask, input_pad_mask, sample_weight=None):
    """``loss.mask_bce_loss`` with both masks, including the cross-event broadcast of torch.

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
    counts = ops.maximum(ops.sum(pad, axis=-1), 1.0)  # [B]
    term = ops.einsum("qpn,bn->bqp", loss, pad) / counts[:, None, None]
    return _cross_event_object_mean(term, object_valid_mask)


def mask_dice_loss(logits, targets, object_valid_mask, input_pad_mask):
    """``loss.mask_dice_loss`` with both masks, including the cross-event broadcast of torch.

    The target sum is not masked (torch semantics). Shapes as in ``mask_bce_loss``.
    """
    probs = ops.sigmoid(f32(logits))
    targets = f32(targets)
    pad = f32(input_pad_mask)
    numerator = 2.0 * ops.einsum("qpn,bn->bqp", probs * targets, pad)
    denominator = ops.einsum("qpn,bn->bqp", probs, pad) + ops.sum(targets, axis=-1)[None]
    term = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return _cross_event_object_mean(term, object_valid_mask)


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
