"""Lightning callbacks for HGQ2 housekeeping (ports of the keras-native utilities)."""

import torch
from lightning import Callback, LightningModule, Trainer


class EBOPsMonitor(Callback):
    """Log the model's total EBOPs each validation epoch (Lightning port of hgq FreeEBOPs).

    EBOPs (effective bit operations) estimate FPGA resource usage (~LUT + 55*DSP);
    tracking them alongside the physics metrics is how the resource/accuracy tradeoff
    is steered during quantization-aware training.
    """

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total = 0.0
        for layer in pl_module.model.keras_layers():
            ebops = getattr(layer, "ebops", None)
            if ebops is not None:
                total += float(torch.as_tensor(ebops))
        pl_module.log("val/ebops", total, sync_dist=True)


class BetaScheduler(Callback):
    """Linearly ramp the HGQ2 EBOPs regularization strength (beta) over training steps.

    Starting QAT with a small (or zero) beta lets the task loss settle before the
    resource pressure kicks in; beta is written to every quantized layer's underlying
    variable (the attribute HGQ2's add_loss actually reads).
    """

    def __init__(self, beta_start: float = 0.0, beta_end: float = 1e-5, warmup_steps: int = 1000):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_steps = warmup_steps

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        fraction = min(1.0, trainer.global_step / max(1, self.warmup_steps))
        beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
        for layer in pl_module.model.keras_layers():
            beta_var = getattr(layer, "_beta", None)
            if beta_var is not None:
                beta_var.assign(beta)
        pl_module.log("train/quant_beta", beta, sync_dist=True)
