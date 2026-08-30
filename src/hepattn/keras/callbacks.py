"""Lightning callbacks for HGQ2 housekeeping (ports of the keras-native utilities)."""

import csv
import math
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


class BetaPID(Callback):
    """Drive beta with a PID controller so total EBOPs converges on a target.

    Lightning port of ``hgq.utils.sugar.BetaPID``, and the method used by the only
    published transformer-on-FPGA result that shares our stack (Laatu, Sun et al.,
    arXiv:2510.24784, which trains every model to a fixed target of 350k EBOPs --
    roughly one Super Logic Region of an XCU250). The control law here is
    arithmetically identical to HGQ2's; only the framework hooks differ, because
    this repo trains through Lightning rather than ``keras.Model.fit``.

    Why this replaces `BetaScheduler`: an open-loop ramp requires knowing the right
    beta in advance. We do not. Calibrating it from the *reported* ``total_ebops``
    overstated the true loss slope by 10,410x, and four decades of beta then moved
    the mean learned bitwidth by 0.01%. A controller does not need the calibration --
    it measures the cost each epoch and corrects.

    The control law works in log space (the default, and the only mode ported):

        err       = log10(ebops / target)      > 0 when the model is too expensive
        integral += err
        beta      = 10 ** (p*err + i*integral + d*(err - prev_err))

    At ``epoch == warmup_epochs`` the integral is seeded so the controller's first
    output is exactly ``init_beta``; the ramp therefore starts from a known point
    rather than a discontinuity.

    KNOWN RISK, measured on this model: ``total_ebops`` is 99.98% QSoftmax and behaves
    as a step function -- every activation quantizer must lose ~0.6 bits before the
    reported total changes at all. Across nine epochs and four beta values spanning
    50x, it held at 1.6755e15 to 0.0002%. A controller facing a process variable that
    does not respond will keep integrating, so `max_beta` is a required safety valve
    rather than a formality. Saturating it is itself an informative outcome: it is
    closed-loop evidence that the blocker is the metric's granularity and not the
    choice of beta.

    Interaction with `PeriodicEBOPs`: when EBOPs is evaluated every N steps the
    time-averaged penalty is 1/N of nominal, so the beta the controller settles on is
    ~N times the value an every-step run would need. This needs no compensation --
    the controller finds it -- but `init_beta` is in the same (config) units, so scale
    it by N when carrying a value over from a `BetaScheduler` run.

    Args:
        target_ebops: Reported total EBOPs to hold. For orientation: an XCVU13P holds
            ~2.15e6 EBOPs under ``LUTs ~ exp(0.985 * log(EBOPs))`` against its 1.728M
            LUTs. Set a target the run can plausibly reach -- an unreachable one
            saturates the controller on the first post-warmup epoch and degenerates
            into training at ``max_beta``.
        init_beta: Beta held during warmup and reproduced exactly by the controller's
            first output.
        p: Proportional gain.
        i: Integral gain. HGQ2's default of 2e-3 is tuned for hundred-epoch runs; at
            an err of ~0.2 it moves beta by a factor of 1.001 per epoch. QAT here costs
            12.7 h/epoch, so the default below is set for a ~1 decade/10 epoch ramp.
        d: Derivative gain. EBOPs is noisy; leave at 0.
        warmup_epochs: Epochs to hold `init_beta` before the controller engages.
        max_beta: Hard ceiling. Bounds how far an unresponsive metric can drive beta.
        min_beta: Hard floor.
        damp_beta_on_target: When under target, beta *= (1 - this). Mitigates overshoot.

    Raises:
        ValueError: If `target_ebops` or `i` is non-positive, or if a `BetaScheduler`
            is also registered -- it writes `_beta` every batch, i.e. after this
            callback's epoch-start write, and would silently erase the controller.
    """

    def __init__(
        self,
        target_ebops: float,
        init_beta: float,
        p: float = 1.0,
        i: float = 0.5,
        d: float = 0.0,
        warmup_epochs: int = 2,
        max_beta: float = 1e-7,
        min_beta: float = 0.0,
        damp_beta_on_target: float = 0.0,
    ):
        # Coerce BEFORE comparing. YAML 1.1 requires a sign in the exponent, so a config
        # writing `target_ebops: 1.0e15` yields the *string* "1.0e15" and every numeric
        # comparison below would raise TypeError instead of validating. Configs here use
        # `1.0e+15`, but accepting the other spelling costs nothing and the failure mode
        # is otherwise a TypeError several hours into a run.
        target_ebops, init_beta = float(target_ebops), float(init_beta)
        p, i, d = float(p), float(i), float(d)
        if target_ebops <= 0:
            raise ValueError(f"target_ebops must be > 0, got {target_ebops}")
        if i <= 0:
            raise ValueError(f"integral gain must be > 0 (the integral seeding divides by it), got {i}")
        if init_beta <= 0:
            raise ValueError(f"init_beta must be > 0 (the controller works in log space), got {init_beta}")
        self.target_ebops = target_ebops
        self.init_beta = init_beta
        self.p, self.i, self.d = p, i, d
        self.warmup_epochs = int(warmup_epochs)
        self.max_beta, self.min_beta = float(max_beta), float(min_beta)
        self.damp_beta_on_target = float(damp_beta_on_target)
        self.integral = 0.0
        self.prev_error = 0.0
        self.beta = float(init_beta)
        self._seeded = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        clashing = [c for c in trainer.callbacks if isinstance(c, BetaScheduler)]
        if clashing:
            raise ValueError(
                "BetaPID and BetaScheduler both write every quantized layer's _beta. "
                "BetaScheduler writes on_train_batch_start, which runs after this "
                "callback's on_train_epoch_start, so the controller would have no "
                "effect at all. Remove BetaScheduler from the config."
            )

    def read_ebops(self, pl_module: LightningModule) -> float:
        total = 0.0
        for layer in pl_module.model.keras_layers():
            ebops = getattr(layer, "ebops", None)
            if ebops is not None:
                total += float(torch.as_tensor(ebops))
        return total

    def write_beta(self, pl_module: LightningModule, beta: float) -> None:
        for layer in pl_module.model.keras_layers():
            beta_var = getattr(layer, "_beta", None)
            if beta_var is not None:
                beta_var.assign(beta)

    def step(self, ebops: float) -> float:
        """Advance the controller one epoch and return the new beta.

        Split out from the Lightning hook so the control law can be tested against
        HGQ2's implementation without constructing a model.
        """
        if not self._seeded:
            # Seed the integral so this first call reproduces init_beta exactly:
            # the call below does integral += err, after which
            #   p*err + i*integral == log10(init_beta).
            err = math.log10(ebops / self.target_ebops + 1e-9)
            self.integral = (math.log10(self.beta) - self.p * err) / self.i - err
            self._seeded = True

        error = math.log10(ebops / self.target_ebops)
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        beta = 10.0 ** (self.p * error + self.i * self.integral + self.d * derivative)
        if ebops < self.target_ebops:
            beta *= 1.0 - self.damp_beta_on_target
        self.beta = max(min(beta, self.max_beta), self.min_beta)
        return self.beta

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ebops = self.read_ebops(pl_module)

        if trainer.current_epoch < self.warmup_epochs:
            self.write_beta(pl_module, self.init_beta)
        elif ebops <= 0.0:
            # No training-mode forward has populated the trackers yet (or EBOPs is
            # switched off). log10 would blow up; hold and try again next epoch.
            self.write_beta(pl_module, self.beta)
        else:
            self.write_beta(pl_module, self.step(ebops))

        pl_module.log("train/quant_beta", self.beta, sync_dist=True)
        pl_module.log("train/pid_ebops", ebops, sync_dist=True)
        pl_module.log("train/pid_saturated", float(self.beta >= self.max_beta), sync_dist=True)
