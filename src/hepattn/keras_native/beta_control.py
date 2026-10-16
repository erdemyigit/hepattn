"""Host-side control of the HGQ2 EBOPs penalty weight beta: the BetaPID control law and the BetaScheduler ramp.

Pure Python: this module imports no keras, jax, torch or lightning, so the JAX driver and the Lightning hybrid share
one implementation. ``hepattn.keras.callbacks.BetaPID`` and ``BetaScheduler`` wrap these (plan KERAS_JAX_PLAN.md
section 3.6: the hybrid module imports torch and lightning, and ``hepattn.keras`` raises under ``KERAS_BACKEND=jax``).

Both run between steps on the host. The JAX driver writes the returned beta into the state with
``ebops.EbopsLeaves.set_beta`` (no retrace).

BetaPID control law (log mode of ``hgq.utils.sugar.BetaPID``; checked against it in ``tests/keras/test_beta_pid.py`` and
``tests/keras_native/test_beta_control.py``)::

    err       = log10(ebops / target)      > 0 when the model is too expensive
    integral += err
    beta      = 10 ** (p * err + i * integral + d * (err - prev_err))
    beta     *= 1 - damp_beta_on_target    only when ebops < target
    beta      = clip(beta, min_beta, max_beta)

The first ``step`` seeds the integral so that it returns ``init_beta`` exactly. ``on_epoch_start`` is the epoch hook of
the Lightning callback: ``init_beta`` during warm-up, the held beta while no EBOPs reading exists (``ebops <= 0``),
``step(ebops)`` otherwise.

The JAX EBOPs read-out is ``penalty / beta`` (``ebops.ebops_readout``), so the native driver needs beta > 0 at every
EBOPs-active step: give it ``min_beta > 0`` and a ramp that starts above 0. The defaults here stay those of the hybrid
callbacks (``min_beta = 0``, ``beta_start = 0``) so that both keep the same control law.
"""

import math


class BetaPIDController:
    """The BetaPID control law and its epoch hook, without framework hooks (see the module docstring).

    Args:
        target_ebops: EBOPs setpoint (> 0).
        init_beta: Beta held during warm-up and returned by the first ``step`` (> 0).
        p: Proportional gain.
        i: Integral gain (> 0; the integral seeding divides by it).
        d: Derivative gain.
        warmup_epochs: Epochs that hold ``init_beta`` before the controller engages.
        max_beta: Upper clamp.
        min_beta: Lower clamp.
        damp_beta_on_target: Below target, beta is multiplied by ``1 - damp_beta_on_target``.

    Raises:
        ValueError: If ``target_ebops``, ``init_beta`` or ``i`` is not positive.
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
        # float() first: YAML 1.1 reads `1.0e15` (no exponent sign) as a string
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
        self.beta = init_beta
        self.seeded = False

    def step(self, ebops: float) -> float:
        """Advance the controller by one epoch on an EBOPs reading (> 0) and return the new beta."""
        if not self.seeded:
            # after the `integral += error` below, p * err + i * integral == log10(init_beta) (HGQ2's seeding, incl. 1e-9)
            err = math.log10(ebops / self.target_ebops + 1e-9)
            self.integral = (math.log10(self.beta) - self.p * err) / self.i - err
            self.seeded = True

        error = math.log10(ebops / self.target_ebops)
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        beta = 10.0 ** (self.p * error + self.i * self.integral + self.d * derivative)
        if ebops < self.target_ebops:
            beta *= 1.0 - self.damp_beta_on_target
        self.beta = max(min(beta, self.max_beta), self.min_beta)
        return self.beta

    def on_epoch_start(self, epoch: int, ebops: float) -> float:
        """The beta to write at the start of ``epoch``, given the latest EBOPs reading (0 or less: none yet)."""
        if epoch < self.warmup_epochs:
            return self.init_beta
        if ebops <= 0.0:
            return self.beta  # log10 needs a reading; hold, and seed once one exists
        return self.step(ebops)

    @property
    def saturated(self) -> bool:
        return self.beta >= self.max_beta

    def state_dict(self) -> dict:
        """The controller state that a checkpoint must keep (the gains are configuration)."""
        return {"integral": self.integral, "prev_error": self.prev_error, "beta": self.beta, "seeded": self.seeded}

    def load_state_dict(self, state: dict) -> None:
        self.integral = float(state["integral"])
        self.prev_error = float(state["prev_error"])
        self.beta = float(state["beta"])
        self.seeded = bool(state["seeded"])


def ramp_beta(step: int, beta_start: float, beta_end: float, warmup_steps: int) -> float:
    """BetaScheduler: linear ramp from ``beta_start`` to ``beta_end`` over ``warmup_steps`` optimizer steps, then flat."""
    fraction = min(1.0, step / max(1, warmup_steps))
    return beta_start + fraction * (beta_end - beta_start)
