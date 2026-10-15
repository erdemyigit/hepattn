"""The mask losses must normalise every object by its own event's padding.

The other loss tests pad every event by the same amount, which is exactly the case in which a
padding mask broadcast against the wrong axis still gives the right answer. These tests pad
the events unequally and compare against a plain per-event loop, so a regression to the
flattened-object broadcast is caught.
"""

import pytest
import torch

from hepattn.models.loss import mask_bce_loss, mask_dice_loss, mask_focal_loss, mask_kl_div_loss


def make_problem(seed: int, batch_size: int = 4, num_objects: int = 5, num_inputs: int = 12):
    """A batch whose events have different numbers of valid constituents and valid objects."""
    gen = torch.Generator().manual_seed(seed)
    pred_logits = torch.randn(batch_size, num_objects, num_inputs, generator=gen)
    targets = (torch.rand(batch_size, num_objects, num_inputs, generator=gen) > 0.6).float()

    # Unequal padding per event, at least two valid constituents each.
    num_valid_inputs = torch.tensor([num_inputs, num_inputs // 2, 3, 2])[:batch_size]
    input_pad_mask = torch.arange(num_inputs)[None, :] < num_valid_inputs[:, None]
    num_valid_objects = torch.tensor([num_objects, 1, 3, 2])[:batch_size]
    object_valid_mask = torch.arange(num_objects)[None, :] < num_valid_objects[:, None]

    # Padded constituents belong to no object, as in real data, and every valid object owns at
    # least one valid constituent so the KL targets are normalisable.
    targets = targets * input_pad_mask[:, None, :]
    targets[..., 0] = 1.0
    return pred_logits, targets, object_valid_mask, input_pad_mask


def per_event_reference(loss_fn, pred_logits, targets, object_valid_mask, input_pad_mask, **kwargs):
    """The docstring semantics: one valid object at a time, with its event's padding physically removed.

    Each call still passes an (all-true) input mask so that it takes the same code path as the
    batched call and only the padding handling is under test.
    """
    per_object = [
        loss_fn(
            pred_logits[b, n, input_pad_mask[b]][None, None],
            targets[b, n, input_pad_mask[b]][None, None],
            input_pad_mask=torch.ones(1, int(input_pad_mask[b].sum()), dtype=torch.bool),
            **kwargs,
        )
        for b in range(pred_logits.shape[0])
        for n in torch.nonzero(object_valid_mask[b]).flatten()
    ]
    return torch.stack(per_object).mean()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("loss_fn", "kwargs"),
    [(mask_bce_loss, {}), (mask_dice_loss, {}), (mask_focal_loss, {"gamma": 2.0}), (mask_kl_div_loss, {})],
    ids=["bce", "dice", "focal", "kl_div"],
)
def test_mask_loss_normalises_within_each_event(loss_fn, kwargs, seed):
    pred_logits, targets, object_valid_mask, input_pad_mask = make_problem(seed)

    batched = loss_fn(pred_logits, targets, object_valid_mask=object_valid_mask, input_pad_mask=input_pad_mask, **kwargs)
    reference = per_event_reference(loss_fn, pred_logits, targets, object_valid_mask, input_pad_mask, **kwargs)

    assert torch.isfinite(batched)
    assert torch.allclose(batched, reference, atol=1e-6)


def test_mask_kl_div_loss_ignores_invalid_object_slots():
    """An invalid object slot with an all-zero target must not leak a NaN into the mean."""
    pred_logits, targets, object_valid_mask, input_pad_mask = make_problem(3)
    targets[~object_valid_mask] = 0.0

    loss = mask_kl_div_loss(pred_logits, targets, object_valid_mask=object_valid_mask, input_pad_mask=input_pad_mask)

    assert torch.isfinite(loss)
