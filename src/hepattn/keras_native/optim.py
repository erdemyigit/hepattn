"""Optimizer factory for the native JAX trainer: AdamW or Lion, OneCycle LR, global-norm clipping, gradient accumulation.

It reproduces ``MPflowHGQ.configure_optimizers`` (``experiments/clic/lightning_module_hgq.py``) and the Lightning trainer
settings (``gradient_clip_val``, ``accumulate_grad_batches``) with Keras optimizers, which ``train_step`` drives through
``optimizer.stateless_apply`` (plan KERAS_JAX_PLAN.md section 3.7):

- ``AdamW`` (``keras.optimizers.AdamW``, epsilon 1e-8 as torch) or ``Lion`` (``keras.optimizers.Lion``, betas 0.9/0.99 as
  ``lion_pytorch.Lion``). Both decay weights decoupled, ``w -= lr * wd * w`` before the update, as torch AdamW and
  lion-pytorch do.
- Weight decay skips every trainable variable whose PATH contains ``quantizer`` (the hybrid's no-decay group), through
  ``exclude_from_weight_decay(var_list=...)``. ``var_names=["quantizer"]`` would match ``variable.name`` (HGQ2 names
  its bitwidths ``b``, ``i``, ``f``) and exclude nothing. HGQ2's ``beta`` is non-trainable, so no update touches it.
- ``global_clipnorm`` = ``gradient_clip_val`` (``gradient_clip_algorithm: norm``).
- ``lrs_config.quantizer_lr_scale`` (native only, default 1: the stock Keras optimizer): the same trainable quantizer
  variables (bitwidths, in bit units) step with ``learning_rate x quantizer_lr_scale``. AdamW and Lion move a variable
  by about the LR per step whatever its gradient, so at a fine-tuning LR a bitwidth moves a fraction of a bit per
  thousand steps; the scale lets the bitwidths move at their own rate (``ScaledLRMixin``). The hybrid ignores the key.
- ``accumulate_grad_batches = k > 1`` -> ``gradient_accumulation_steps = k``: Keras AVERAGES the k micro-batch gradients,
  then clips and applies (``base_optimizer.apply``); Lightning also averages (it divides each micro-batch loss by k,
  ``lightning/pytorch/loops/optimization/automatic.py``) and clips at the optimizer step.
- ``OneCycleLR``: torch ``OneCycleLR`` (cos anneal, two phases) at the OPTIMIZER step. Keras evaluates a schedule at its
  micro-step counter ``_iterations`` (``_get_current_learning_rate``), so the schedule divides by
  ``steps_per_update = k``; Adam's bias correction uses ``iterations`` = ``_iterations // k``.

Deviations from torch (documented in docs/hgq/port_log/S5.md):
- beta1 is FIXED (0.9). torch ``OneCycleLR(cycle_momentum=True)`` also cycles beta1 between 0.95 and 0.85 for AdamW and
  Lion; a callable ``beta_1`` fails under the jitted ``stateless_apply`` (plan section 3.7). User decision: fixed beta1.
- Keras Adam adds epsilon to ``sqrt(v)`` before the bias correction of v (``alpha = lr * sqrt(1 - b2^t) / (1 - b1^t)``,
  update ``alpha * m / (sqrt(v) + eps)``); torch adds it after (``m / bc1 / (sqrt(v) / sqrt(bc2) + eps)``). Same for
  ``eps = 0``; otherwise the effective epsilon is ``eps * sqrt(bc2)`` in torch's terms (0.032x at step 1, -> 1x).
- Clipping: Keras scales by ``clip / max(norm, clip)``, torch by ``min(1, clip / (norm + 1e-6))``.
- Past ``total_steps`` torch raises; this schedule holds the value at ``total_steps``, the last step torch defines
  (its anneal there runs past ``min_lr`` by ``pct = (total - p1) / (total - 1 - p1)``, about 1.7e-5 * max_lr above
  ``min_lr`` at 400 steps). The last LR Lightning applies is torch's value at ``total_steps - 1`` = ``min_lr``.
"""

import math

import keras
from keras import ops

SUPPORTED_OPTIMIZERS = ("adamw", "lion")


class OneCycleLR(keras.optimizers.schedules.LearningRateSchedule):
    """torch ``OneCycleLR(max_lr, total_steps, div_factor, final_div_factor, pct_start)``, cos anneal, two phases.

    Args:
        max_lr: Peak learning rate.
        total_steps: Optimizer steps in the cycle.
        div_factor: ``initial_lr = max_lr / div_factor``.
        final_div_factor: ``min_lr = initial_lr / final_div_factor``.
        pct_start: Fraction of the cycle spent increasing the LR.
        steps_per_update: Micro-steps per optimizer step (gradient accumulation); the schedule reads
            ``floor(step / steps_per_update)``.

    Raises:
        ValueError: For ``total_steps <= 0``, ``pct_start`` outside [0, 1], ``pct_start * total_steps <= 1`` (torch divides
            by zero at step 0 then) or ``steps_per_update < 1``.
    """

    def __init__(self, max_lr, total_steps, div_factor=25.0, final_div_factor=1e4, pct_start=0.3, steps_per_update=1):
        if int(total_steps) <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if not 0.0 <= float(pct_start) <= 1.0:
            raise ValueError(f"pct_start must be in [0, 1], got {pct_start}")
        if float(pct_start) * int(total_steps) <= 1.0:
            raise ValueError(f"pct_start * total_steps must be > 1 (torch divides by zero at step 0), got {pct_start} * {total_steps}")
        if int(steps_per_update) < 1:
            raise ValueError(f"steps_per_update must be >= 1, got {steps_per_update}")
        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.pct_start = float(pct_start)
        self.steps_per_update = int(steps_per_update)
        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor
        self.phase1_end = float(self.pct_start * self.total_steps) - 1
        self.phase2_end = float(self.total_steps - 1)

    @staticmethod
    def _cos(start, end, pct):
        return end + (start - end) / 2.0 * (ops.cos(math.pi * pct) + 1.0)

    def __call__(self, step):
        n = ops.floor(ops.cast(step, "float32") / self.steps_per_update)
        n = ops.minimum(n, float(self.total_steps))
        rising = self._cos(self.initial_lr, self.max_lr, n / self.phase1_end)
        falling = self._cos(self.max_lr, self.min_lr, (n - self.phase1_end) / (self.phase2_end - self.phase1_end))
        return ops.where(n <= self.phase1_end, rising, falling)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
            "pct_start": self.pct_start,
            "steps_per_update": self.steps_per_update,
        }


def quantizer_variables(model) -> list:
    """Trainable variables with ``quantizer`` in their path: the hybrid's weight-decay-free group."""
    return [v for v in model.trainable_variables if "quantizer" in v.path]


class ScaledLRMixin:
    """``update_step`` with the learning rate times ``lr_scale`` for the variables given to ``scale_variables``."""

    lr_scale: float = 1.0

    def scale_variables(self, var_list, lr_scale: float) -> None:
        self.lr_scale = float(lr_scale)
        self._scaled_keys = {self._var_key(v) for v in var_list}

    def update_step(self, gradient, variable, learning_rate):
        if self._var_key(variable) in getattr(self, "_scaled_keys", ()):
            learning_rate = ops.multiply(learning_rate, self.lr_scale)
        return super().update_step(gradient, variable, learning_rate)


class ScaledAdamW(ScaledLRMixin, keras.optimizers.AdamW):
    pass


class ScaledLion(ScaledLRMixin, keras.optimizers.Lion):
    pass


def learning_rate(lrs_config: dict, total_steps: int, steps_per_update: int = 1):
    """The hepattn ``lrs_config`` LR: a ``OneCycleLR`` (initial -> max -> end), or the constant ``initial`` with ``skip_scheduler``."""
    if lrs_config.get("skip_scheduler"):
        return float(lrs_config["initial"])
    initial, peak, end = float(lrs_config["initial"]), float(lrs_config["max"]), float(lrs_config["end"])
    return OneCycleLR(
        max_lr=peak,
        total_steps=total_steps,
        div_factor=peak / initial,
        final_div_factor=initial / end,
        pct_start=float(lrs_config["pct_start"]),
        steps_per_update=steps_per_update,
    )


def build_optimizer(
    model,
    *,
    optimizer: str,
    lrs_config: dict,
    total_steps: int,
    clip_norm: float | None = None,
    accumulate: int = 1,
    adam_epsilon: float = 1e-8,
):
    """A built Keras optimizer for ``model.trainable_variables`` (see the module docstring).

    Args:
        model: The built native model.
        optimizer: ``AdamW`` or ``Lion`` (case-insensitive).
        lrs_config: hepattn ``lrs_config`` (``initial``, ``max``, ``end``, ``pct_start``, ``weight_decay``, ``skip_scheduler``;
            optional ``quantizer_lr_scale``).
        total_steps: Optimizer steps of the whole run (Lightning ``estimated_stepping_batches``).
        clip_norm: Global gradient-norm clip, or None.
        accumulate: Micro-batches per optimizer step.
        adam_epsilon: AdamW epsilon (torch default 1e-8; Keras default 1e-7).

    Raises:
        ValueError: For an unknown optimizer or ``accumulate < 1``.
    """
    kind = optimizer.lower()
    if kind not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; supported: AdamW, Lion")
    if int(accumulate) < 1:
        raise ValueError(f"accumulate must be >= 1, got {accumulate}")
    accumulate = int(accumulate)
    common = {
        "learning_rate": learning_rate(lrs_config, total_steps, steps_per_update=accumulate),
        "weight_decay": float(lrs_config["weight_decay"]),
        "global_clipnorm": None if clip_norm is None else float(clip_norm),
        "gradient_accumulation_steps": accumulate if accumulate > 1 else None,
    }
    scale = float(lrs_config.get("quantizer_lr_scale", 1.0))
    if not scale > 0:
        raise ValueError(f"quantizer_lr_scale must be > 0, got {scale}")
    if kind == "adamw":
        opt = (keras.optimizers.AdamW if scale == 1.0 else ScaledAdamW)(beta_1=0.9, beta_2=0.999, epsilon=float(adam_epsilon), **common)
    else:
        opt = (keras.optimizers.Lion if scale == 1.0 else ScaledLion)(beta_1=0.9, beta_2=0.99, **common)
    opt.exclude_from_weight_decay(var_list=quantizer_variables(model))
    if scale != 1.0:
        opt.scale_variables(quantizer_variables(model), scale)
    opt.build(model.trainable_variables)
    return opt


def optimizer_from_config(model_section: dict, trainer_section: dict, model, total_steps: int):
    """``build_optimizer`` from the ``model:`` and ``trainer:`` sections of a hepattn YAML (``polaris_hgq.yaml``, ``base.yaml``).

    Reads ``model.optimizer`` (default AdamW, as ``ModelWrapper``), ``model.lrs_config``, ``trainer.gradient_clip_val``,
    ``trainer.gradient_clip_algorithm`` (``norm`` only) and ``trainer.accumulate_grad_batches``.

    Raises:
        NotImplementedError: For ``gradient_clip_algorithm`` other than ``norm``.
    """
    algorithm = trainer_section.get("gradient_clip_algorithm") or "norm"  # None (explicit null) is Lightning's "norm"
    if algorithm != "norm":
        raise NotImplementedError(f"gradient_clip_algorithm={algorithm!r} is not supported (only 'norm')")
    return build_optimizer(
        model,
        optimizer=model_section.get("optimizer", "AdamW"),
        lrs_config=model_section["lrs_config"],
        total_steps=total_steps,
        clip_norm=trainer_section.get("gradient_clip_val"),
        accumulate=int(trainer_section.get("accumulate_grad_batches", 1) or 1),
    )
