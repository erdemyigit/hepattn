"""Lightning callbacks for HGQ2 housekeeping (ports of the keras-native utilities)."""

import csv
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer


class MetricsCsvWriter(Callback):
    """Append every logged metric to a plain CSV, one row per validation epoch.

    Exists because on a compute node the repo's Comet logger is forced offline and
    writes an opaque archive, so a 4-day beta sweep produced NO readable metrics on
    disk — the trend could only be reconstructed afterwards by loading checkpoint
    tensors.

    Why a callback rather than Lightning's `CSVLogger`: the repo's CLI hardcodes
    `trainer.logger.init_args.offline_directory` and links `name` into
    `trainer.logger.init_args.name`, both of which assume `trainer.logger` is a single
    entry. Passing a *list* of loggers makes jsonargparse replace the list with a bare
    Namespace carrying no `class_path`, and the run dies at instantiation — measured on
    Polaris, the smoke test failed exactly this way. A callback never touches the logger
    machinery, so it works whatever the primary logger is.

    Captures the latest value of every metric in `trainer.callback_metrics`, so train
    metrics are end-of-epoch snapshots rather than per-step curves. That is the cadence
    the resource/accuracy question needs (is `val/bits_mean` moving between epochs?).

    Register this AFTER any callback whose metrics it should capture — Lightning runs
    callbacks in list order, so `BitwidthMonitor` must log before this writes.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._header: list[str] | None = None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:  # a sanity-check row would carry meaningless values
            return
        row: dict[str, float] = {"epoch": float(trainer.current_epoch), "step": float(trainer.global_step)}
        for key, value in trainer.callback_metrics.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue  # non-scalar metrics (e.g. confusion matrices) are not CSV-able
        self._append(row)

    def _append(self, row: dict[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._header is None:
            if self.path.exists() and self.path.stat().st_size > 0:
                # Resuming: keep the existing schema so the file stays parseable as one table.
                with self.path.open(newline="") as f:
                    self._header = next(csv.reader(f))
            else:
                self._header = ["epoch", "step", *sorted(k for k in row if k not in {"epoch", "step"})]
                with self.path.open("w", newline="") as f:
                    csv.writer(f).writerow(self._header)
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._header, extrasaction="ignore").writerow({k: row.get(k, "") for k in self._header})


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
    """Log the learned quantizer bitwidths — the actual output variable of QAT.

    Three DIFFERENT trainable parameters exist and they are not interchangeable.
    Measured on the dim-32 test model, effect of reducing each by 1 bit:

        param                       total_ebops    loss slope (E_eff)
        /b  weight bits (kbi)          -0.00%          -9.32%
        /f  activation frac (kif)     -54.17%          -9.95%
        /i  activation int  (kif)      -8.33%          -0.49%

    So the reported hardware cost is driven almost entirely by the ACTIVATION
    quantizers, while the regularizer's gradient is split roughly evenly between /b and
    /f. Logging only /b — as this callback originally did — reports a number that can
    move a long way while EBOPs does not budge, which is exactly what happened on the
    first calibrated Polaris sweep: /b went 8.0000 -> 7.8900 (min 7.30, max 8.70) while
    val/ebops stayed at 1.6755e15 to five significant figures across all four beta runs.

    Log all three separately, and never summarise them into one "bitwidth".
    """

    SUFFIXES = ("b", "f", "i")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        groups: dict[str, list[torch.Tensor]] = {s: [] for s in self.SUFFIXES}
        for name, param in pl_module.model.named_parameters():
            suffix = name.rsplit("/", 1)[-1]
            if suffix in groups:
                groups[suffix].append(param.detach().float())
        for suffix, params in groups.items():
            if not params:
                continue
            pl_module.log(f"val/bits_{suffix}_mean", torch.stack([p.mean() for p in params]).mean(), sync_dist=True)
            pl_module.log(f"val/bits_{suffix}_min", torch.stack([p.min() for p in params]).min(), sync_dist=True)
            pl_module.log(f"val/bits_{suffix}_max", torch.stack([p.max() for p in params]).max(), sync_dist=True)
            pl_module.log(f"val/bits_{suffix}_n", float(len(params)), sync_dist=True)


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
