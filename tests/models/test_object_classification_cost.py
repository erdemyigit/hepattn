"""The multi-class matching cost must be computed from logits, not from probabilities.

`object_ce_cost` applies its own softmax, while `ObjectClassificationTask.forward` stores
`softmax(logits)` under the `_class_prob` key. Reading that key in `cost()` applies a second
softmax: the per-row ordering survives (softmax is monotone), but the dynamic range collapses,
so the classification term loses weight against the mask costs it is summed with in the
Hungarian assignment. These tests pin the contract and the consequence.
"""

import pytest
import torch

from hepattn.models.dense import Dense
from hepattn.models.loss import object_ce_cost
from hepattn.models.task import ObjectClassificationTask

NUM_CLASSES, BATCH, QUERIES, DIM = 5, 2, 12, 16


def _task():
    torch.manual_seed(0)
    return ObjectClassificationTask(
        name="classification",
        input_object="query",
        output_object="pflow",
        target_object="particle",
        losses={"object_ce": 2.0},
        costs={"object_ce": 2.0},
        net=Dense(input_size=DIM, output_size=NUM_CLASSES + 1),
        num_classes=NUM_CLASSES,
    )


def _batch(scale=3.0):
    torch.manual_seed(1)
    x = {"query_embed": torch.randn(BATCH, QUERIES, DIM) * scale}
    targets = {"particle_class": torch.randint(0, NUM_CLASSES + 1, (BATCH, QUERIES))}
    return x, targets


def test_cost_equals_single_softmax_reference():
    """cost() must equal object_ce_cost applied to the raw logits, weight included."""
    task, (x, targets) = _task(), _batch()
    outputs = task(x)
    got = task.cost(outputs, targets)["object_ce"]
    reference = 2.0 * object_ce_cost(outputs[task.logits_key].detach().to(torch.float32), targets["particle_class"].long())
    torch.testing.assert_close(got, reference, atol=1e-6, rtol=1e-6)

    # Discriminating: the double-softmax value is far from the reference, so this is not vacuous.
    double = 2.0 * object_ce_cost(outputs[task.probs_key].detach().to(torch.float32), targets["particle_class"].long())
    assert not torch.allclose(got, double, atol=1e-3), "double-softmax path is indistinguishable; test cannot discriminate"


def test_cost_keeps_its_dynamic_range():
    """The spread of the cost must be that of one softmax, which is several times wider."""
    task, (x, targets) = _task(), _batch()
    outputs = task(x)
    got = task.cost(outputs, targets)["object_ce"]
    double = 2.0 * object_ce_cost(outputs[task.probs_key].detach().to(torch.float32), targets["particle_class"].long())

    spread = (got.max() - got.min()).item()
    double_spread = (double.max() - double.min()).item()
    assert spread > 3 * double_spread, f"expected a much wider range from single softmax, got {spread:.4f} vs {double_spread:.4f}"


def test_binary_path_unchanged():
    """num_classes=1 already fed logits to a sigmoid-based cost; it must be untouched."""
    torch.manual_seed(0)
    task = ObjectClassificationTask(
        name="classification",
        input_object="query",
        output_object="pflow",
        target_object="particle",
        losses={"object_bce": 1.0},
        costs={"object_bce": 1.0},
        net=Dense(input_size=DIM, output_size=1),
        num_classes=1,
    )
    x = {"query_embed": torch.randn(BATCH, QUERIES, DIM)}
    targets = {"particle_valid": torch.randint(0, 2, (BATCH, QUERIES)).bool()}
    outputs = task(x)
    cost = task.cost(outputs, targets)["object_bce"]
    assert cost.shape == (BATCH, QUERIES, QUERIES)
    assert torch.isfinite(cost).all()


@pytest.mark.parametrize("scale", [1.0, 3.0, 6.0])
def test_assignment_weight_relative_to_mask_cost(scale):
    """With a mask cost of fixed scale, the classification term must retain enough range to
    influence the assignment. Under the double softmax its contribution is ~4-5x smaller.
    """
    task, (x, targets) = _task(), _batch(scale)
    outputs = task(x)
    got = task.cost(outputs, targets)["object_ce"]
    double = 2.0 * object_ce_cost(outputs[task.probs_key].detach().to(torch.float32), targets["particle_class"].long())
    ratio = (got.max() - got.min()).item() / (double.max() - double.min()).item()
    assert ratio > 3.0, f"classification term only {ratio:.2f}x wider than the double-softmax path at scale {scale}"
