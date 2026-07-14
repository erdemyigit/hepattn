"""Training-dynamics check for the quantized model outside Lightning.

Discriminating pair: the loss must decrease (training works at all) AND at least one
quantizer parameter must move (catches "trains but quantizers frozen", e.g. lazy-built
quantizer variables created after the optimizer collected parameters).
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from test_maskformer_parity import clic_dummy_batch, make_keras_model
from test_quantized import HIGH_QUANT


def test_quantized_training_step_decreases_loss_and_moves_quantizers():
    torch.manual_seed(60)
    # beta=0: EBOPs pressure off so the loss trend reflects the task losses (the raw
    # EBOPs magnitude ~1e12 would swallow task-loss progress below fp32 resolution);
    # quantizer params still receive straight-through gradients from the task loss.
    # EBOPs differentiability itself is covered by test_quantized.py.
    quant = {**HIGH_QUANT, "ebops": {"beta0": 0.0}}
    model = make_keras_model(60, quant=quant).train()
    inputs, targets = clic_dummy_batch()

    # materialize lazily-built quantized layers BEFORE the optimizer collects params
    # (the same contract MPflowHGQ.setup fulfils in the Lightning driver)
    model.eval()
    with torch.no_grad():
        model(inputs)
    model.train()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.endswith("/beta")]
    quantizer_before = {n: p.detach().clone() for n, p in named if "quantizer" in n}
    assert quantizer_before, "no quantizer parameters found after materialization"

    opt = torch.optim.AdamW([p for _, p in named], lr=1e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        outputs = model(inputs)
        _, _, loss_dict = model.loss(outputs, dict(targets))
        total = sum(v for layer in loss_dict.values() for task in layer.values() for v in task.values() if torch.isfinite(v))
        total = total + model.quant_losses()
        total.backward()
        opt.step()
        losses.append(float(total.detach()))

    assert losses[-1] < losses[0], f"training loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"

    moved = [n for n, p in named if "quantizer" in n and not torch.equal(p.detach(), quantizer_before[n])]
    assert moved, "no quantizer parameter changed during training — quantizers are frozen"
