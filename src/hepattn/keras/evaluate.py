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
    """Total EBOPs of the model, summed over its keras layers, for a representative batch.

    NOTE: this is the *reported* resource estimate, and it is NOT the quantity the
    training objective minimizes — see `effective_ebops`. Measured on the dim-32 test
    model the two differ by ~5 orders of magnitude, because ~94% of the reported total
    comes from `QSoftmax` layers whose cost does not enter the regularization loss.
    Use this for resource accounting; use `effective_ebops` to calibrate beta.
    """
    _populate_ebops(model, batch)
    total = 0.0
    for layer in model.keras_layers():
        ebops = getattr(layer, "ebops", None)
        if ebops is not None:
            total += float(torch.as_tensor(ebops))
    return total


def effective_ebops(model: nn.Module, batch: dict, probes: tuple[float, float] = (1e-9, 1e-6)) -> tuple[float, float]:
    """Measure the EBOPs coefficient the training loss actually sees, plus its constant floor.

    `quant_losses()` is affine in beta: ``quant_loss(beta) = floor + beta * E_eff``. The
    slope ``E_eff`` is what the optimizer trades against the task loss, so it -- not
    `total_ebops` -- is what beta must be calibrated against. `floor` is HGQ2's separate
    always-on regularization term, which beta cannot influence.

    Calibrating from `total_ebops` instead is what made the Polaris beta sweep inert:
    four decades of beta moved the mean learned bitwidth by 0.01%.

    Returns:
        (E_eff, floor). To make the resource term a fraction `f` of a task loss `L`,
        set ``beta = f * L / E_eff``.

    Raises:
        ValueError: If the two probe values are equal (the slope would be undefined).
        RuntimeError: If no quantized layer exposes a `_beta` variable, i.e. this is not
            a quantized model and there is no resource penalty to measure.
    """
    b1, b2 = probes
    if b1 == b2:
        raise ValueError("probes must differ")

    layers = [layer for layer in model.keras_layers() if getattr(layer, "_beta", None) is not None]
    if not layers:
        raise RuntimeError("no quantized layers expose a _beta variable — is this a quantized model?")
    saved = [float(keras.ops.convert_to_numpy(layer._beta)) for layer in layers]  # noqa: SLF001

    def _quant_loss_at(beta: float) -> float:
        for layer in layers:
            layer._beta.assign(beta)  # noqa: SLF001
        was_training = model.training
        model.train()
        with torch.no_grad():
            model(batch)  # training-mode forward re-registers the add_loss terms
            value = float(model.quant_losses())
        model.train(was_training)
        return value

    try:
        q1, q2 = _quant_loss_at(b1), _quant_loss_at(b2)
    finally:
        for layer, beta in zip(layers, saved, strict=True):
            layer._beta.assign(beta)  # noqa: SLF001

    e_eff = (q2 - q1) / (b2 - b1)
    return e_eff, q1 - e_eff * b1


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
