"""EBOPs accounting for a trained KerasMaskFormer (resource side of the eval harness).

EBOPs (effective bit-operations) are HGQ2's differentiable proxy for FPGA cost
(≈ LUTs + 55·DSPs). They are materialized on the keras layers during a training-mode
forward, so a representative batch must be passed through the model in training mode
before reading them.
"""

from collections.abc import Iterable

import torch
from torch import nn

from hepattn.keras import keras


@torch.no_grad()
def _populate_ebops(model: nn.Module, batch: dict) -> None:
    was_training = model.training
    model.train()
    model(batch)  # training-mode forward populates each quantized layer's ebops tracker
    model.train(was_training)


def total_ebops(model: nn.Module, batch: dict) -> float:
    """Total EBOPs of the model, summed over its keras layers, for a representative batch."""
    _populate_ebops(model, batch)
    total = 0.0
    for layer in model.keras_layers():
        ebops = getattr(layer, "ebops", None)
        if ebops is not None:
            total += float(torch.as_tensor(ebops))
    return total


def ebops_by_region(model: nn.Module, batch: dict, regions: Iterable[str] = ("input_nets", "encoder", "decoder", "tasks")) -> dict[str, float]:
    """EBOPs grouped by top-level model region, to show where the FPGA cost concentrates.

    Mirrors total_ebops: a top-level keras Layer's own ``ebops`` already includes its
    sub-quantizers, so recursion STOPS at each keras Layer (descending further would
    double-count). Each layer is counted once globally (by object id) so the decoder's
    task modules — which alias model.tasks — are not counted twice. Region totals
    therefore sum to the grand total.
    """
    _populate_ebops(model, batch)
    totals: dict[str, float] = dict.fromkeys(regions, 0.0)
    seen: set[int] = set()

    def walk(module: nn.Module, region: str | None) -> None:
        for name, child in module.named_children():
            child_region = region if region is not None else (name if name in totals else None)
            if isinstance(child, keras.layers.Layer):
                ebops = getattr(child, "ebops", None)
                if ebops is not None and child_region is not None and id(child) not in seen:
                    totals[child_region] += float(torch.as_tensor(ebops))
                    seen.add(id(child))
                # do NOT recurse: child.ebops already accounts for its sub-layers
            else:
                walk(child, child_region)

    walk(model, None)
    return totals
