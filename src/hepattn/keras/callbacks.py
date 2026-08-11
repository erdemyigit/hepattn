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


class BitwidthMonitor(Callback):
    """Log statistics of the *learned bitwidths* — the actual output variable of QAT.

    Nothing else logs this. `EBOPsMonitor` reports the resource estimate and
    `BetaScheduler` reports the penalty strength, but neither answers the only question
    QAT exists to answer: are the bitwidths moving?

    This matters because the reported EBOPs total is NOT the quantity the loss minimizes
    (see `hepattn.keras.evaluate.effective_ebops`). A beta sweep can therefore look
    plausible on every logged metric while being completely inert. Measured on Polaris:
    four decades of beta moved the mean bitwidth from 8.0000 to 7.981 — a 0.01% spread
    across the whole sweep — and it took reading tensors out of checkpoints to notice,
    because no metric on disk carried the number.

    Bitwidths live in the quantizer variables whose parameter names end in `/b`.
    """

    def _bit_params(self, pl_module: LightningModule) -> list[torch.Tensor]:
        return [p.detach() for name, p in pl_module.model.named_parameters() if name.endswith("/b")]

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        bits = self._bit_params(pl_module)
        if not bits:
            return
        per_layer_means = torch.stack([b.float().mean() for b in bits])
        pl_module.log("val/bits_mean", per_layer_means.mean(), sync_dist=True)
        pl_module.log("val/bits_min", torch.stack([b.float().min() for b in bits]).min(), sync_dist=True)
        pl_module.log("val/bits_max", torch.stack([b.float().max() for b in bits]).max(), sync_dist=True)
        pl_module.log("val/bits_n_layers", float(len(bits)), sync_dist=True)


class PeriodicEBOPs(Callback):
    """Evaluate the EBOPs resource penalty every N steps instead of every step.

    EBOPs is a *regularizer*: the forward math does not depend on it, but HGQ2 computes
    it inside every quantized layer's call — an extra matmul over the bitwidth tensors
    plus an `add_loss` — on every training step. Measured on an A100 (dim-256 CLIC
    model, 250 quantized layers, micro-batch 32): **16.1% of total step time**, i.e.
    1140 ms -> 957 ms with it disabled.

    Because EBOPs moves very slowly during training (measured: 0.03% drift over 14k
    steps while beta nearly doubled), sampling it every N steps loses almost nothing.
    On skipped steps the EBOPs term is not registered, so the *time-averaged* penalty
    drops by 1/N; compensate by scaling `BetaScheduler.beta_end` by N — the Polaris
    config generator does this automatically, keeping the user-facing `beta_end`
    meaning unchanged.

    Measured on the dim-32 test model: `quant_losses()` falls from ~805 on an active
    step to ~0.38 on a skipped one (99.95% removed). The small remainder is a separate,
    always-on HGQ2 regularization term, not residual EBOPs — it is constant across
    consecutive skipped steps and is deliberately left alone.

    Note: `EBOPsMonitor` reads the value last written by a training step, so its
    reading can be up to N steps stale. Irrelevant at these drift rates.
    """

    def __init__(self, every_n_steps: int = 8):
        self.every_n_steps = max(1, int(every_n_steps))
        self._layers: list | None = None

    def _quantized_layers(self, pl_module: LightningModule) -> list:
        # Cache: keras-3 layers are torch Modules under the torch backend, and the
        # EBOPs flag lives on the QLayer instances wherever they are nested.
        if self._layers is None:
            self._layers = [m for m in pl_module.model.modules() if hasattr(m, "_enable_ebops")]
        return self._layers

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        active = (trainer.global_step % self.every_n_steps) == 0
        for layer in self._quantized_layers(pl_module):
            layer._enable_ebops = active  # noqa: SLF001  (the attribute the forward reads)
        pl_module.log("train/ebops_active", float(active), sync_dist=True)


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
