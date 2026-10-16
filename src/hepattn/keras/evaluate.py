"""EBOPs accounting for a trained KerasMaskFormer (resource side of the eval harness).

EBOPs (effective bit-operations) are HGQ2's differentiable proxy for FPGA cost
(≈ LUTs + 55·DSPs). They are materialized on the keras layers during a training-mode
forward, so a representative batch must be passed through the model in training mode
before reading them.

Every function here acts on the same HGQ2 leaves as the training callbacks
(``callbacks.quantized_leaves``: nested ones included, such as the QSoftmax exp/inv tables and the
masked-Linformer core leaves), so the readout, the beta calibration and the beta the callbacks
write describe one objective.
"""

from collections.abc import Iterable

import torch
from torch import nn

from hepattn.keras import keras
from hepattn.keras.callbacks import leaf_ebops, quantized_leaves
from hepattn.keras.callbacks import total_ebops as leaves_total_ebops


@torch.no_grad()
def _populate_ebops(model: nn.Module, batch: dict) -> None:
    was_training = model.training
    model.train()
    model(batch)  # training-mode forward populates each quantized layer's ebops tracker
    model.train(was_training)


def total_ebops(model: nn.Module, batch: dict) -> float:
    """Total EBOPs of the model for a representative batch: the sum of every HGQ2 leaf's own EBOPs.

    This is the resource estimate and, since beta reaches every leaf (R4b), also the slope of
    the regularization loss up to the task heads' share, which ``quant_losses()`` counts twice
    (see ``effective_ebops``).
    """
    _populate_ebops(model, batch)
    return leaves_total_ebops(model)


def effective_ebops(model: nn.Module, batch: dict, probes: tuple[float, float] = (1e-9, 1e-6)) -> tuple[float, float]:
    """Measure the EBOPs coefficient the training loss actually sees, plus its constant floor.

    `quant_losses()` is affine in beta: ``quant_loss(beta) = floor + beta * E_eff``. The
    slope ``E_eff`` is what the optimizer trades against the task loss. Beta is written to
    every HGQ2 leaf, as the callbacks write it.

    History: the Polaris beta sweep was inert because beta was written only to the top-level
    keras layers, while the QSoftmax tables (99.99% of the dense model's EBOPs) are nested
    leaves with their own beta; ``E_eff`` was then ~5 orders of magnitude below
    ``total_ebops``. With every leaf written, ``E_eff`` equals ``total_ebops`` plus the
    task heads' EBOPs, which ``quant_losses()`` counts twice.

    The slope comes from the two probes; the floor from a third forward at beta 0, because
    at the probes ``beta * E_eff`` (up to ~1e9) hides the floor (~1) in float32.

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

    layers = [layer for layer in quantized_leaves(model) if getattr(layer, "_beta", None) is not None]
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
        q1, q2, floor = _quant_loss_at(b1), _quant_loss_at(b2), _quant_loss_at(0.0)
    finally:
        for layer, beta in zip(layers, saved, strict=True):
            layer._beta.assign(beta)  # noqa: SLF001

    return (q2 - q1) / (b2 - b1), floor


def ebops_by_region(model: nn.Module, batch: dict, regions: Iterable[str] = ("input_nets", "encoder", "decoder", "tasks")) -> dict[str, float]:
    """EBOPs grouped by top-level model region, to show where the FPGA cost concentrates.

    Each HGQ2 leaf (``callbacks.quantized_leaves``) counts once with its own EBOPs, in the
    region of its shortest module path: the decoder's task modules alias model.tasks, and
    count once, under ``tasks``. When every leaf sits under one of ``regions``, the totals sum
    to ``total_ebops``.
    """
    _populate_ebops(model, batch)
    totals: dict[str, float] = dict.fromkeys(regions, 0.0)
    paths: dict[int, tuple[str, nn.Module]] = {id(layer): ("", layer) for layer in quantized_leaves(model)}
    for path, module in model.named_modules(remove_duplicate=False):
        best = paths.get(id(module))
        if best is not None and (not best[0] or len(path) < len(best[0])):
            paths[id(module)] = (path, module)
    for path, module in paths.values():
        region = path.split(".", 1)[0]
        if region in totals:
            totals[region] += leaf_ebops(module)
    return totals
