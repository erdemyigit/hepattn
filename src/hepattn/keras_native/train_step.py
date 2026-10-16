"""Jitted, stateless train / eval / predict steps for the keras_native MaskFormer on JAX (plan KERAS_JAX_PLAN.md 3.5-3.7).

One training step is one XLA program::

    outputs, ntv, qlosses = model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)
    task, losses = loss_fn(outputs, targets)        # matched loss: stop_gradient costs -> 1 pure_callback -> losses
    loss = task + sum(EBOPs terms) + sum(regularizer terms)      # each HGQ2 layer once (stateless_call)
    grads = d loss / d tv                            # value_and_grad over the trainable variables only (beta is not one)
    tv, ov = optimizer.stateless_apply(ov, grads, tv)

API for the S6 driver
---------------------
::

    model = maskformer.from_config(cfg["model"]); model(inputs, training=False)   # build eagerly BEFORE any jit (plan 3.1)
    optimizer = optim.optimizer_from_config(cfg["model"], cfg["trainer"], model, total_steps)
    steps = TrainStep(model, optimizer, maskformer_loss_from_config(cfg["model"]["model"]["init_args"]))
    state = steps.init_state()                       # TrainState(tv, ntv, ov): COPIES of the model/optimizer variables
    state = steps.set_beta(state, beta)              # host, between steps; same shapes, no retrace
    state, out = steps.train_step(state, (inputs, targets), ebops_on=True)
    #   ``state`` is DONATED: its buffers are reused for the new state, so never touch the old one again.
    #   ``out`` is a StepOutput of device scalars; ``steps.ebops(out, beta)`` is the EBOPs read-out (active steps only).
    #   ``beta`` is the host value last given to ``set_beta``: keep it on the host. ``steps.read_beta(state)`` copies
    #   every leaf's beta to the host (a device sync that stalls async dispatch), so use it for checks only.
    out = steps.eval_step(state, (inputs, targets))  # training=False: no EBOPs term, state unchanged; EvalOutput
    #   ``out.loss`` = ``out.task_loss + out.regularization`` is the hybrid's val/loss: ``MPflowHGQ.aggregate_losses``
    #   adds ``quant_losses()``, which at training=False holds only the HGQ2 weight-regularizer terms.
    outputs = steps.predict_step(state, inputs)      # training=False output tree
    steps.assign_to_model(state)                     # copy the state into the Keras variables (save_weights, export)
    state = steps.init_state()                       # after loading weights into the model

Traces (``compile_counts``): ``train_step`` one per ``ebops_on`` value (PeriodicEBOPs: 2), ``eval_step`` 1, ``predict_step`` 1,
for a fixed batch shape. Each trace is one compiled program. ``jax.jit``'s dispatch cache is keyed on the Python argument
types too, so mixing numpy and ``jax.Array`` batches adds cache entries without a retrace; the counts do not see those.

Gradient accumulation is the optimizer's (``optim.build_optimizer``): call ``train_step`` once per micro-batch; every k-th
call applies the averaged gradient. Each call makes one host matching call.

PeriodicEBOPs with accumulation: compute ``ebops_on`` from the OPTIMIZER-step counter, ``(micro_step // k) % every == 0``.
The hybrid callback uses ``trainer.global_step % every_n_steps`` (``hepattn/keras/callbacks.py`` ``PeriodicEBOPs``), and
Lightning's ``global_step`` counts optimizer steps, so all k micro-batches of every ``every``-th update carry the EBOPs
term. A micro-step counter gives the same mean pull but a different cadence and read-out schedule.

Speed: Keras' compiled trainer donates its state to ``jax.jit`` (``keras/src/backend/jax/trainer.py``
``make_train_function``: ``jit(train_step, donate_argnums=0)``); the G1 spike's custom step did not. ``train_step``
donates ``state`` by default. docs/hgq/port_log/S5.md has the measurements.

Precision: the traced bodies run under ``jax.default_matmul_precision(matmul_precision)``, "highest" by default, so XLA
does not lower fp32 products to TF32 on Ampere GPUs: HGQ2 emulates fixed-point arithmetic, and a TF32 product rounds
operands to an 11-bit significand (S5.md, TF32 decision).
"""

import contextlib
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hepattn.keras_native.ebops import EbopsLeaves, ebops_readout
from hepattn.keras_native.maskformer import _modules
from hepattn.keras_native.objective import matched_loss, matcher_n_jobs_from_config, specs_from_config

LossFn = Callable[[dict, dict], tuple[Any, Any]]


class TrainState(NamedTuple):
    """Explicit training state: trainable, non-trainable (incl. HGQ2 beta/ebops) and optimizer variable values."""

    tv: list
    ntv: list
    ov: list


class StepOutput(NamedTuple):
    """Device values of one training step (computed before the update, from the step's forward)."""

    loss: Any  # task + penalty + regularization: what was differentiated
    task_loss: Any
    losses: Any  # the loss_fn aux, e.g. {layer: {task: {name: scalar}}}
    penalty: Any  # sum of the HGQ2 EBOPs terms (beta * EBOPs); 0 when ebops_on is False
    regularization: Any  # sum of the weight-regularizer terms
    grad_norm: Any  # global norm of this (micro-)batch's gradient before clipping; NaN unless TrainStep(grad_norm=True)


class EvalOutput(NamedTuple):
    """Device values of one ``training=False`` step."""

    outputs: Any
    loss: Any  # task + regularization: the hybrid's val/loss
    task_loss: Any
    losses: Any
    regularization: Any  # sum of the weight-regularizer terms (no EBOPs term exists at training=False)


def maskformer_loss(specs, *, n_jobs: int = 1, host_fn=None, target_object: str = "particle") -> LossFn:
    """``loss_fn(outputs, targets) -> (total, losses)``: the matched MaskFormer loss with one host matching call."""

    def loss_fn(outputs, targets):
        total, losses, _ = matched_loss(outputs, targets, specs, n_jobs=n_jobs, host_fn=host_fn, target_object=target_object)
        return total, losses

    return loss_fn


def maskformer_loss_from_config(init_args: dict, *, host_fn=None) -> LossFn:
    """``maskformer_loss`` from the MaskFormer ``init_args`` of a hepattn YAML (its ``tasks`` and ``matcher``)."""
    specs = specs_from_config(_modules(init_args["tasks"]))
    n_jobs = matcher_n_jobs_from_config(init_args["matcher"]) if init_args.get("matcher") else 1
    return maskformer_loss(specs, n_jobs=n_jobs, host_fn=host_fn, target_object=init_args.get("target_object", "particle"))


def _global_norm(grads) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in grads))


class TrainStep:
    """The jitted steps of one built model, optimizer and loss (see the module docstring).

    Args:
        model: A built Keras model; ``model(inputs)`` returns the output tree ``loss_fn`` reads.
        optimizer: A Keras optimizer built for ``model.trainable_variables`` (``optim.build_optimizer``).
        loss_fn: ``loss_fn(outputs, targets) -> (scalar, aux pytree)``, traceable (``maskformer_loss``).
        matmul_precision: ``jax.default_matmul_precision`` inside the steps ("highest"), or None to keep the global one.
        donate: Donate the state to ``train_step`` (as Keras' trainer does).
        grad_norm: Also return the global gradient norm (one more reduction over every gradient; the optimizer's clipping
            computes its own).

    Raises:
        ValueError: If the model is not built or the optimizer is not built for its trainable variables.
    """

    def __init__(self, model, optimizer, loss_fn: LossFn, *, matmul_precision: str | None = "highest", donate: bool = True, grad_norm: bool = False):
        if not model.built:
            raise ValueError("build the model eagerly (one call on a real batch) before creating the jitted steps")
        if not optimizer.built or len(optimizer._trainable_variables) != len(model.trainable_variables):  # noqa: SLF001
            raise ValueError("the optimizer must be built for model.trainable_variables (optim.build_optimizer)")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.matmul_precision = matmul_precision
        self.with_grad_norm = grad_norm
        self.ebops_leaves = EbopsLeaves(model)
        self._traces = {"train": 0, "eval": 0, "predict": 0}  # incremented at trace time only (Python side effect)
        self._train = jax.jit(self._train_impl, static_argnames=("ebops_on",), donate_argnums=(0,) if donate else ())
        self._eval = jax.jit(self._eval_impl)
        self._predict = jax.jit(self._predict_impl)

    # ----------------------------------------------------------------------------------------------- traced bodies

    def _precision(self):
        if self.matmul_precision is None:
            return contextlib.nullcontext()
        return jax.default_matmul_precision(self.matmul_precision)

    def _objective(self, tv, ntv, batch, ebops_on: bool):
        inputs, targets = batch
        with self.ebops_leaves.enabled(ebops_on):
            outputs, ntv, qlosses = self.model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)
        penalty_terms, reg_terms = self.ebops_leaves.split_losses(qlosses)
        task, losses = self.loss_fn(outputs, targets)
        penalty = sum(penalty_terms, jnp.zeros((), jnp.float32))
        regularization = sum(reg_terms, jnp.zeros((), jnp.float32))
        return task + penalty + regularization, (ntv, task, losses, penalty, regularization)

    def _grads_impl(self, state: TrainState, batch, ebops_on: bool):
        with self._precision():
            (loss, aux), grads = jax.value_and_grad(self._objective, has_aux=True)(state.tv, state.ntv, batch, ebops_on)
        ntv, task, losses, penalty, regularization = aux
        norm = _global_norm(grads) if self.with_grad_norm else jnp.full((), jnp.nan, jnp.float32)
        return grads, ntv, StepOutput(loss, task, losses, penalty, regularization, norm)

    def _train_impl(self, state: TrainState, batch, ebops_on: bool):
        self._traces["train"] += 1
        grads, ntv, out = self._grads_impl(state, batch, ebops_on)
        with self._precision():
            tv, ov = self.optimizer.stateless_apply(state.ov, grads, state.tv)
        return TrainState(tv, ntv, ov), out

    def _eval_impl(self, state: TrainState, batch):
        self._traces["eval"] += 1
        inputs, targets = batch
        with self._precision():
            outputs, _, qlosses = self.model.stateless_call(state.tv, state.ntv, inputs, training=False, return_losses=True)
            task, losses = self.loss_fn(outputs, targets)
        regularization = sum(qlosses, jnp.zeros((), jnp.float32))
        return EvalOutput(outputs, task + regularization, task, losses, regularization)

    def _predict_impl(self, state: TrainState, inputs):
        self._traces["predict"] += 1
        with self._precision():
            outputs, _ = self.model.stateless_call(state.tv, state.ntv, inputs, training=False)
        return outputs

    # ----------------------------------------------------------------------------------------------- public API

    def init_state(self) -> TrainState:
        """Copies of the current model and optimizer variables (copies, so that donation never frees a Keras variable)."""

        def copy(variables):
            return [jnp.array(v.value, copy=True) for v in variables]

        return TrainState(copy(self.model.trainable_variables), copy(self.model.non_trainable_variables), copy(self.optimizer.variables))

    def assign_to_model(self, state: TrainState) -> None:
        """Write ``state`` into the Keras model and optimizer variables (as copies: ``state`` stays donatable)."""
        for variables, values in (
            (self.model.trainable_variables, state.tv),
            (self.model.non_trainable_variables, state.ntv),
            (self.optimizer.variables, state.ov),
        ):
            for var, value in zip(variables, values, strict=True):
                var.assign(jnp.array(value, copy=True))

    def train_step(self, state: TrainState, batch, *, ebops_on: bool = True) -> tuple[TrainState, StepOutput]:
        """One (micro-)batch ``(inputs, targets)``: forward, matched loss, gradients, optimizer update. Donates ``state``."""
        return self._train(state, batch, ebops_on=bool(ebops_on))

    def grads(self, state: TrainState, batch, *, ebops_on: bool = True) -> tuple[list, StepOutput]:
        """The gradients of ``train_step`` without the update (not jitted on its own; wrap in ``jax.jit`` to reuse)."""
        grads, _, out = self._grads_impl(state, batch, bool(ebops_on))
        return grads, out

    def eval_step(self, state: TrainState, batch) -> EvalOutput:
        return self._eval(state, batch)

    def predict_step(self, state: TrainState, inputs):
        return self._predict(state, inputs)

    def set_beta(self, state: TrainState, beta: float) -> TrainState:
        return state._replace(ntv=self.ebops_leaves.set_beta(state.ntv, beta))

    def read_beta(self, state: TrainState) -> float:
        return self.ebops_leaves.read_beta(state.ntv)

    @staticmethod
    def ebops(out: StepOutput, beta: float) -> float:
        """Total EBOPs of an EBOPs-active step, ``penalty / beta`` (the common beta the step ran with)."""
        return ebops_readout(float(out.penalty), beta)

    def compile_counts(self) -> dict[str, int]:
        """Number of traces (= compiled programs) of each jitted step so far, not ``jax.jit`` dispatch-cache entries."""
        return dict(self._traces)
