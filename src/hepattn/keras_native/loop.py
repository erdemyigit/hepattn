"""Single-device training driver of the keras_native MaskFormer (plan KERAS_JAX_PLAN.md 3.5, 3.8-3.10).

``Trainer`` runs the hybrid's Lightning fit loop (``MPflowHGQ`` + the CLIC callbacks) with the S5 jitted steps:

    config (``config.load_config``) -> model (eager build on one batch) -> optimizer (``optim.build_optimizer`` from
    ``RunConfig.optimizer``) -> ``TrainStep`` -> epochs of ``train_step`` over a ``DataSource`` -> validation ->
    checkpoints, CSV logs, beta control.

Order of one epoch (the Lightning hook order of the hybrid callbacks):

1. epoch start: ``BetaPID`` (``beta_control.BetaPIDController.on_epoch_start`` with the latest EBOPs read-out);
2. every micro-batch: ``BetaScheduler`` ramp at the optimizer step (``ramp_beta``), ``PeriodicEBOPs``
   (``ebops_on = global_step % every == 0``: all micro-batches of an optimizer step carry the term), ``train_step``;
3. after each optimizer step: train log every ``log_every_n_steps`` optimizer steps (Lightning's logger cadence,
   ``(steps done) % n == 0``), the step checkpoint (``ModelCheckpoint(every_n_train_steps)``);
4. epoch end: validation (``check_val_every_n_epoch``), then ``EBOPsMonitor`` / ``BitwidthMonitor`` values, the
   ``MetricsCsvWriter`` row, the epoch checkpoint (hepattn ``Checkpoint``, monitor ``val/loss``) and ``last``.

Logged values (names as the hybrid logs them):

- ``train/loss``, ``val/loss``: the TASK loss (sum of every layer / task / loss term). The hybrid logs ``{stage}/loss``
  in ``ModelWrapper.aggregate_losses`` (``models/wrapper.py``) BEFORE ``MPflowHGQ.aggregate_losses`` adds
  ``quant_losses()`` (``experiments/clic/lightning_module_hgq.py``), so the monitor of the epoch checkpoints is the task
  loss, not the differentiated total. ``{stage}/quant_ebops_loss`` is the HGQ2 term (penalty + regularizers in
  training, regularizers only at ``training=False``); ``train/total_loss`` is what was differentiated.
- ``val/*``: epoch means weighted by the number of real events of each batch (Lightning infers ``batch_size`` from the
  batch). Padded val batches give the real events bit-identical outputs (S6a); their LOSS would not be the unpadded
  loss (padding events add null-class queries to the CE), so the loss of a short batch is recomputed on its real events
  by a loss-only program (``tail`` trace) from the same forward.
- ``train/ebops`` / ``val/ebops``: ``penalty / beta`` of the latest EBOPs-active step (plan 3.6; HGQ2's uint32
  ``_ebops`` saturates on JAX). NaN while no active step with beta > 0 has run.
- ``{stage}/bits_{b,f,i}_{mean,min,max,n}``: ``BitwidthMonitor`` over every model variable whose last path segment is
  ``b``, ``f`` or ``i`` (trainable and non-trainable: the hybrid reads ``named_parameters``, which includes both).
- val metrics: ``metrics.ClicMetrics("val", pool_layers=...)`` on ``predict`` of the matched outputs; train task
  metrics at each logged step from the ``training=False`` forward of that micro-batch (the hybrid uses the training
  forward; documented deviation).

Checkpoints (``checkpoint.CheckpointManager.from_callback`` per configured callback, all in ``<run_dir>/ckpts``): the
state ``(tv, ntv, ov)`` and the metadata of ``Trainer.meta`` (counters, data position = next micro-batch, beta, the PID
state, the last EBOPs read-out, the last val metrics, the last train row and BetaPID values for the next
``metrics.csv`` row, seed, model hash). They are written only at optimizer-step boundaries, so no accumulation is in
flight. ``Trainer.restore`` + ``fit`` continue bit for bit on CPU; in the same run directory the CSV rows after the
checkpoint and stale checkpoint temporaries are dropped first.

Deviations from Lightning (docs/hgq/port_log/S6b.md): an epoch uses ``(batches // k) * k`` micro-batches (Lightning
steps once more on a partial accumulation) and ``total_steps`` is ``batches // k * epochs`` (Lightning: ceil); no
sanity validation; one device.

Data: a ``DataSource`` gives ``Batch`` objects (``keras_native.data``) of one epoch from a micro-batch index on; the
CLIC torch-DataLoader source is ``hepattn.experiments.clic.main_jax`` (torch stays outside this package).
"""

import csv
import json
import math
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import jax
import jax.numpy as jnp
import keras
import numpy as np

from hepattn.keras_native import checkpoint as ckpt
from hepattn.keras_native import profile
from hepattn.keras_native.beta_control import BetaPIDController, ramp_beta
from hepattn.keras_native.checkpoint import CheckpointError
from hepattn.keras_native.config import RunConfig
from hepattn.keras_native.data import Batch, prefetch
from hepattn.keras_native.matching import host_match
from hepattn.keras_native.metrics import ClicMetrics
from hepattn.keras_native.objective import matched_loss, permute_outputs
from hepattn.keras_native.optim import build_optimizer, learning_rate
from hepattn.keras_native.predict_writer import PredictionWriter, predict
from hepattn.keras_native.train_step import TrainState, TrainStep, maskformer_loss

BIT_SUFFIXES = ("b", "f", "i")
FP32_PRECISIONS = {None, 32, "32", "32-true"}
_END = object()


class DataSource(Protocol):
    """Batches of one split. ``num_batches``: batches of one epoch (train: full batches only, ``drop_last``)."""

    num_batches: int

    def epoch(self, epoch: int, start_batch: int = 0) -> Iterable[Batch]: ...


@dataclass
class LoopSettings:
    """Run options (Lightning trainer arguments) that the YAML or the command line gives."""

    run_dir: str
    max_epochs: int
    max_steps: int = -1  # optimizer steps (Lightning ``max_steps``); -1: no limit
    limit_train_batches: int | None = None  # micro-batches per epoch
    limit_val_batches: int | None = None
    log_every_n_steps: int = 50
    check_val_every_n_epoch: int = 1
    seed: int = 42
    matmul_precision: str | None = "highest"
    donate: bool = True
    pool_layers: bool = True  # ClicMetrics: the hybrid's pooled values (S6a decision 5)
    prefetch: int = 0  # train batches read ahead in a background thread (data.prefetch); 0: none

    @classmethod
    def from_run_config(cls, rc: RunConfig, run_dir: str | Path, **overrides) -> "LoopSettings":
        """The trainer keys of ``rc`` (``max_epochs``, ``max_steps``, ``limit_*_batches``, ...), then ``overrides``.

        Raises:
            ValueError: For a fractional ``limit_*_batches`` (only a batch count is supported).
        """
        t = rc.trainer
        values: dict[str, Any] = {"run_dir": str(run_dir), "max_epochs": int(t.get("max_epochs") or 1)}
        if t.get("max_steps") is not None:
            values["max_steps"] = int(t["max_steps"])
        for key in ("limit_train_batches", "limit_val_batches"):
            value = t.get(key)
            if isinstance(value, float) and value != 1.0:
                raise ValueError(f"trainer.{key}={value}: only an integer batch count is supported")
            if isinstance(value, int) and not isinstance(value, bool):
                values[key] = value
        if t.get("log_every_n_steps") is not None:
            values["log_every_n_steps"] = int(t["log_every_n_steps"])
        if t.get("check_val_every_n_epoch") is not None:
            values["check_val_every_n_epoch"] = int(t["check_val_every_n_epoch"])
        if rc.seed is not None:
            values["seed"] = int(rc.seed)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def enable_compilation_cache(directory: str | Path) -> Path:
    """Persistent JAX compilation cache in ``directory`` (programs that compile for >= 1 s, JAX's default)."""
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    return path


class CompileEvents:
    """Counts JAX's persistent-cache events in this process (``jax.monitoring``)."""

    EVENTS: ClassVar[dict[str, str]] = {
        "/jax/compilation_cache/cache_hits": "cache_hits",
        "/jax/compilation_cache/cache_misses": "cache_misses",
        "/jax/compilation_cache/compile_requests_use_cache": "requests",
    }

    def __init__(self) -> None:
        self.counts = dict.fromkeys(self.EVENTS.values(), 0)
        jax.monitoring.register_event_listener(self._on_event)

    def _on_event(self, event: str, **_kwargs) -> None:
        name = self.EVENTS.get(event)
        if name is not None:
            self.counts[name] += 1


class CsvLog:
    """Append rows to a CSV in the ``hepattn.keras.callbacks.MetricsCsvWriter`` layout (columns ``epoch``, ``step``,
    then the other keys sorted; an existing file's header is kept on resume).

    Unlike ``MetricsCsvWriter`` (header fixed by the first row), a row with new keys rewrites the file with those
    columns appended (sorted), so a first row without the train values (no step logged before the first validation)
    does not drop them for the whole run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.header: list[str] | None = None

    def _rows(self) -> list[dict[str, str]]:
        with self.path.open(newline="") as f:
            return list(csv.DictReader(f))

    def _rewrite(self, header: list[str], rows: list[dict]) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k: r.get(k, "") for k in header} for r in rows)
        tmp.replace(self.path)
        self.header = header

    def append(self, row: dict[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.header is None and self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="") as f:
                self.header = next(csv.reader(f))
        new = sorted(k for k in row if k not in {"epoch", "step"} and k not in (self.header or ()))
        if self.header is None:
            self._rewrite(["epoch", "step", *new], [])
        elif new:
            self._rewrite([*self.header, *new], self._rows())
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore").writerow({k: row.get(k, "") for k in self.header})

    def keep(self, predicate) -> int:
        """Drop the rows for which ``predicate(row)`` (values as floats, ``read_csv``) is false. Returns the rows dropped."""
        if not (self.path.exists() and self.path.stat().st_size > 0):
            return 0
        rows = self._rows()
        kept = [r for r in rows if predicate({k: float(v) if v else math.nan for k, v in r.items()})]
        if len(kept) < len(rows):
            with self.path.open(newline="") as f:
                header = next(csv.reader(f))
            self._rewrite(header, kept)
        return len(rows) - len(kept)


def read_csv(path: str | Path) -> list[dict[str, float]]:
    """Rows of a ``CsvLog`` file as floats (empty cells: NaN)."""
    with Path(path).open(newline="") as f:
        return [{k: float(v) if v else math.nan for k, v in row.items()} for row in csv.DictReader(f)]


def _flat_losses(losses: dict, stage: str) -> dict[str, Any]:
    """``ModelWrapper.aggregate_losses`` names: ``{stage}/{layer}_{task}_{loss}``, ``{stage}/{layer}_loss`` (device values)."""
    out: dict[str, Any] = {}
    for layer, tasks in losses.items():
        layer_total = 0.0
        for task, parts in tasks.items():
            for name, value in parts.items():
                out[f"{stage}/{layer}_{task}_{name}"] = value
                layer_total = layer_total + value
        out[f"{stage}/{layer}_loss"] = layer_total
    return out


def _init_args(rc: RunConfig) -> dict:
    entry = rc.model_section["model"]
    return entry.get("init_args", entry)


def lrs_config(rc: RunConfig) -> dict:
    """``RunConfig.optimizer`` as the hepattn ``lrs_config`` dict that ``optim`` reads."""
    o = rc.optimizer
    return {
        "initial": o.lr_initial,
        "max": o.lr_max,
        "end": o.lr_end,
        "pct_start": o.pct_start,
        "weight_decay": o.weight_decay,
        "skip_scheduler": o.skip_scheduler,
    }


def build_run_optimizer(rc: RunConfig, model, total_steps: int):
    """The optimizer of ``rc`` for ``model`` (one reader of the YAML: ``RunConfig.optimizer``; S56 open issue 1)."""
    o = rc.optimizer
    return build_optimizer(
        model,
        optimizer=o.name,
        lrs_config=lrs_config(rc),
        total_steps=total_steps,
        clip_norm=o.clip_norm,
        accumulate=o.accumulate_grad_batches,
        adam_epsilon=o.epsilon,
    )


class Trainer:
    """One run: model, optimizer, jitted steps, counters, beta control, logs and checkpoints (module docstring).

    Args:
        rc: The run configuration.
        settings: Run options.
        sample: One batch of the train split (builds the model eagerly, plan 3.1).
        train_batches: Full micro-batches of one train epoch (before ``limit_train_batches``).
        host_fn: Optional host matching function (``matching.match``), e.g. a timing wrapper.
        profiler: Optional ``profile.PhaseProfiler``: ``fit`` records its phases (and, without ``host_fn``, the matching
            callback's wall time). None: no recording, no added sync.
        trace: Optional ``profile.JaxTrace``: a ``jax.profiler`` trace of a window of micro-batches.

    Raises:
        ValueError: If an epoch holds fewer micro-batches than one accumulation, or the config asks for a precision
            other than fp32 (the user decision: fp32 first).
    """

    def __init__(self, rc: RunConfig, settings: LoopSettings, *, sample: Batch, train_batches: int, host_fn=None, profiler=None, trace=None) -> None:
        precision = rc.trainer.get("precision")
        if precision not in FP32_PRECISIONS:
            raise ValueError(f"trainer.precision={precision!r}: the native driver runs fp32 only (generate the config with HGQ_PRECISION=32)")
        if int(rc.trainer.get("devices", 1) or 1) > 1:
            warnings.warn(f"trainer.devices={rc.trainer['devices']}: the native driver uses one device", stacklevel=2)
        self.rc, self.s = rc, settings
        self.prof = profiler if profiler is not None else profile.NULL
        self.trace = trace
        if self.prof.enabled and host_fn is None:  # the function match() would use, timed
            host_fn = self.prof.time_host_fn(lambda costs, valid, n=rc.matcher_n_jobs: host_match(costs, valid, n_jobs=n))
        self.k = rc.optimizer.accumulate_grad_batches
        n = train_batches if settings.limit_train_batches is None else min(train_batches, settings.limit_train_batches)
        self.steps_per_epoch = n // self.k
        if self.steps_per_epoch == 0:
            raise ValueError(f"{n} micro-batches per epoch < accumulate_grad_batches {self.k}")
        self.micro_per_epoch = self.steps_per_epoch * self.k
        self.total_steps = self.steps_per_epoch * settings.max_epochs
        if settings.max_steps > 0:
            self.total_steps = min(self.total_steps, settings.max_steps)
        self.run_dir = Path(settings.run_dir)
        self.batch_size = int(next(iter(sample.inputs.values())).shape[0])

        keras.utils.set_random_seed(settings.seed)  # after the datasets: CLICDataset seeds 42 in its constructor
        model = rc.build_model()
        model(sample.inputs, training=False)
        optimizer = build_run_optimizer(rc, model, self.total_steps)
        target_object = _init_args(rc).get("target_object", "particle")
        self.specs, self.target_object = rc.task_specs, target_object
        self.n_jobs, self.host_fn = rc.matcher_n_jobs, host_fn
        loss_fn = maskformer_loss(rc.task_specs, n_jobs=rc.matcher_n_jobs, host_fn=host_fn, target_object=target_object)
        self.steps = TrainStep(model, optimizer, loss_fn, matmul_precision=settings.matmul_precision, donate=settings.donate)
        self.schedule = learning_rate(lrs_config(rc), self.total_steps, steps_per_update=self.k)
        self._traces = {"val": 0, "tail": 0, "bits": 0}
        self._val = jax.jit(self._val_impl)
        self._tail = jax.jit(self._tail_impl)
        self._bits = jax.jit(self._bits_impl)
        self._bit_index = self._bit_positions(model)

        self.has_ebops = len(self.steps.ebops_leaves) > 0
        self.ebops_every = rc.quant.ebops_every_n_steps
        self.pid = BetaPIDController(**rc.quant.beta_pid) if rc.quant.beta_pid else None
        self.ramp = rc.quant.beta_scheduler
        if (self.pid or self.ramp) and not self.has_ebops:
            raise ValueError("a beta callback is configured but the model has no HGQ2 EBOPs leaves (no quant spec)")
        if self.pid is not None and self.pid.min_beta <= 0:
            warnings.warn("BetaPID min_beta <= 0: a beta of 0 has no EBOPs read-out (penalty / beta); the PID then holds", stacklevel=2)

        self.state = self.steps.init_state()
        self.beta = self.steps.read_beta(self.state) if self.has_ebops else None
        self.global_step = 0  # optimizer steps done
        self.micro_step = 0
        self.epoch = 0  # data position: the next micro-batch is (epoch, batch)
        self.batch = 0
        self._reading: tuple[Any, float] | None = None  # (device penalty, beta) of the latest EBOPs-active micro-batch
        self._ebops_value: float | None = None
        self.last_val: dict[str, float] | None = None
        self.epoch_logs: dict[str, float] = {}
        self.train_logs: dict[str, float] = {}
        self.resumed_from: Path | None = None
        self.warm_started_from: Path | None = None
        self.first_step_s: float | None = None

    # ------------------------------------------------------------------------------------------- jitted helpers

    @property
    def seed(self) -> int:
        return self.s.seed

    def _precision(self):
        return self.steps._precision()  # noqa: SLF001  (the same context as the train / eval steps)

    def _val_impl(self, state: TrainState, inputs, targets):
        """``training=False`` forward, matched task loss, regularizers, ``predict`` of the matched outputs."""
        self._traces["val"] += 1
        model = self.steps.model
        with self._precision():
            outputs, _, qlosses = model.stateless_call(state.tv, state.ntv, inputs, training=False, return_losses=True)
            task, losses, idx = matched_loss(outputs, targets, self.specs, n_jobs=self.n_jobs, host_fn=self.host_fn, target_object=self.target_object)
            permuted = permute_outputs(outputs, idx, self.specs)
            preds = predict(permuted, model.tasks)
        regularization = sum(qlosses, jnp.zeros((), jnp.float32))
        return {"task": task, "losses": losses, "regularization": regularization, "preds": preds, "final": permuted["final"], "outputs": outputs}

    def _tail_impl(self, outputs, targets):
        """The matched task loss alone, on the real events of a short batch (outputs of the padded forward)."""
        self._traces["tail"] += 1
        with self._precision():
            task, losses, _ = matched_loss(outputs, targets, self.specs, n_jobs=self.n_jobs, host_fn=self.host_fn, target_object=self.target_object)
        return task, losses

    @staticmethod
    def _bit_positions(model) -> dict[str, tuple[list[int], list[int]]]:
        def last(var) -> str:
            return var.path.rsplit("/", 1)[-1]

        return {
            s: (
                [i for i, v in enumerate(model.trainable_variables) if last(v) == s],
                [i for i, v in enumerate(model.non_trainable_variables) if last(v) == s],
            )
            for s in BIT_SUFFIXES
        }

    def _bits_impl(self, tv, ntv):
        self._traces["bits"] += 1
        out = {}
        for suffix, (ti, ni) in self._bit_index.items():
            arrays = [tv[i] for i in ti] + [ntv[i] for i in ni]
            if not arrays:
                continue
            out[suffix] = (
                jnp.mean(jnp.stack([jnp.mean(a.astype(jnp.float32)) for a in arrays])),
                jnp.min(jnp.stack([jnp.min(a.astype(jnp.float32)) for a in arrays])),
                jnp.max(jnp.stack([jnp.max(a.astype(jnp.float32)) for a in arrays])),
            )
        return out

    def bit_stats(self, stage: str) -> dict[str, float]:
        """``BitwidthMonitor`` values of the current state."""
        stats = self._bits(self.state.tv, self.state.ntv)
        out = {}
        for suffix, (mean, lo, hi) in stats.items():
            ti, ni = self._bit_index[suffix]
            out.update({
                f"{stage}/bits_{suffix}_mean": float(mean),
                f"{stage}/bits_{suffix}_min": float(lo),
                f"{stage}/bits_{suffix}_max": float(hi),
                f"{stage}/bits_{suffix}_n": float(len(ti) + len(ni)),
            })
        return out

    def compile_counts(self) -> dict[str, int]:
        """Traces of every jitted program: ``train`` (one per ``ebops_on``), ``val`` (eval + predict), ``tail``, ``bits``."""
        return {**self.steps.compile_counts(), **self._traces}

    # ------------------------------------------------------------------------------------------- beta and EBOPs

    def _set_beta(self, beta: float) -> None:
        if self.beta is None or beta != self.beta:
            self.state = self.steps.set_beta(self.state, beta)
            self.beta = float(beta)

    def ebops_active(self) -> bool:
        """``PeriodicEBOPs``: the optimizer step ``global_step`` carries the EBOPs term (all of its micro-batches)."""
        return self.has_ebops and self.global_step % self.ebops_every == 0

    @property
    def last_ebops(self) -> float | None:
        """EBOPs read-out of the latest EBOPs-active micro-batch with beta > 0 (a device sync on first access)."""
        if self._reading is not None:
            penalty, beta = self._reading
            self._ebops_value = float(penalty) / beta
            self._reading = None
        return self._ebops_value

    # ------------------------------------------------------------------------------------------- checkpoints

    def meta(self) -> dict:
        return {
            "step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": self.epoch,
            "data_position": {"epoch": self.epoch, "batch": self.batch},
            "beta": self.beta,
            "last_ebops": self.last_ebops,
            "beta_controller": self.pid.state_dict() if self.pid else None,
            "config_hash": self.rc.model_hash,
            "val_metrics": self.last_val,
            "total_steps": self.total_steps,
            "micro_per_epoch": self.micro_per_epoch,
            "seed": self.s.seed,
            # the values Lightning's callback_metrics would still hold for the next MetricsCsvWriter row
            "epoch_logs": self.epoch_logs,
            "train_logs": self.train_logs,
        }

    def restore(self, path: str | Path, *, check_seed: bool = True) -> dict:
        """Load a checkpoint of this run configuration and continue from its data position.

        Args:
            path: The checkpoint.
            check_seed: Refuse a checkpoint written with another seed (the train order is ``EpochSampler(seed +
                epoch)``); the ``test`` command, which does not train, passes False.

        Raises:
            CheckpointError: For another model config (hash), a missing or corrupted file, or a beta in the state other
                than the metadata's.
            ValueError: If the epoch length, the total optimizer steps or the seed differ from the checkpoint's (the LR
                schedule or the data order would not continue).
        """
        path = Path(path)
        model, optimizer = self.steps.model, self.steps.optimizer
        (tv, ntv, ov), meta = ckpt.load_checkpoint(path, model, optimizer, expect_config_hash=self.rc.model_hash)
        for key in ("micro_per_epoch", "total_steps", *(("seed",) if check_seed else ())):
            if meta.get(key) != getattr(self, key):
                raise ValueError(f"{path}: {key} {meta.get(key)} differs from this run's {getattr(self, key)}")
        self.state = TrainState(*([jnp.asarray(x) for x in group] for group in (tv, ntv, ov)))
        self.global_step, self.micro_step = int(meta["step"]), int(meta["micro_step"])
        self.epoch, self.batch = int(meta["data_position"]["epoch"]), int(meta["data_position"]["batch"])
        if self.has_ebops:
            # the host keeps the float64 value it gave set_beta (the read-out divides by it); the state holds its float32
            stored = self.steps.read_beta(self.state)
            if meta["beta"] is None or np.float32(meta["beta"]) != np.float32(stored):
                raise CheckpointError(f"{path}: beta in the state {stored} differs from the metadata {meta['beta']}")
            self.beta = float(meta["beta"])
        if self.pid is not None:
            self.pid.load_state_dict(meta["beta_controller"])
        self._reading, self._ebops_value = None, meta["last_ebops"]
        self.last_val = meta.get("val_metrics")
        self.epoch_logs = dict(meta.get("epoch_logs") or {})
        self.train_logs = dict(meta.get("train_logs") or {})
        self.resumed_from = path.resolve()
        return meta

    def warm_start(self, path: str | Path) -> dict:
        """Load the model variables (weights, quantizer state, beta) of a checkpoint and keep everything else fresh.

        For a checkpoint of another run or a converted Lightning checkpoint
        (``hepattn.experiments.clic.convert_to_native``): the optimizer state, step, LR schedule, beta controller and
        data position start as in a new run; an optimizer state in the file is not read. Call before ``fit``.

        ``checkpoint.load_checkpoint`` refuses another model config (hash) and a missing or corrupted file.

        Raises:
            ValueError: If called after a ``restore`` or after training started.
        """
        if self.resumed_from is not None or self.global_step or self.micro_step:
            raise ValueError("warm_start needs a fresh trainer (no restore, no step)")
        path = Path(path)
        model = self.steps.model
        (tv, ntv, _), meta = ckpt.load_checkpoint(path, model, None, expect_config_hash=self.rc.model_hash)
        self.state = TrainState([jnp.asarray(x) for x in tv], [jnp.asarray(x) for x in ntv], self.state.ov)
        if self.has_ebops:
            self.beta = self.steps.read_beta(self.state)
        self.warm_started_from = path.resolve()
        return meta

    def _managers(self) -> list:
        directory = self.run_dir / "ckpts"
        # temporaries of a write that a kill interrupted (checkpoint._atomic removes its own only on an exception)
        for stale in directory.glob(".*.tmp"):
            stale.unlink(missing_ok=True)
        cfg = self.rc.checkpoint
        self.epoch_manager = ckpt.CheckpointManager.from_callback(directory, cfg.epoch, resume_path=self.resumed_from) if cfg.epoch else None
        self.step_manager = ckpt.CheckpointManager.from_callback(directory, cfg.step, resume_path=self.resumed_from) if cfg.step else None
        return [m for m in (self.epoch_manager, self.step_manager) if m is not None]

    def _save_args(self):
        return self.steps.model, self.steps.optimizer, self.state

    # ------------------------------------------------------------------------------------------- fit

    def fit(self, train: DataSource, val: DataSource | None = None) -> dict:
        """Train to ``max_epochs`` / ``max_steps`` from the current position. Returns the run summary.

        With a profiler: its phases between ``start`` and ``stop`` (``profile`` module docstring). A train source
        that ends before ``micro_per_epoch`` micro-batches raises ``RuntimeError`` (in ``_fit``).
        """
        try:
            if not self.prof.enabled:
                return self._fit(train, val)
            with self.prof.active():
                self.prof.start()
                try:
                    return self._fit(train, val)
                finally:
                    self.prof.stop()
        finally:
            if self.trace is not None:  # a window past the end of the run
                self.trace.stop(self.state)

    def _jit_phase(self, name: str):
        """Phase ``name`` around a jitted call, labelled ``+compile`` if the call traced (profiler only)."""
        return _JitPhase(self, name) if self.prof.enabled else profile.NULL.phase(name)

    def _batches(self, source: DataSource, epoch: int, start: int, *, train: bool):
        """The batches of an epoch: read ahead (``prefetch``, train only) and timed as phase ``data`` (profiler)."""
        batches = source.epoch(epoch, start)
        if train and self.s.prefetch > 0:
            batches = prefetch(batches, self.s.prefetch)
        return self._timed(batches, train=train) if self.prof.enabled else batches

    def _timed(self, iterable, *, train: bool):
        it = iter(iterable)
        try:
            while True:
                if train:
                    self.prof.at(self.micro_step + 1, self.global_step, self.epoch)
                with self.prof.phase("data"):
                    item = next(it, _END)
                if item is _END:
                    return
                yield item
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()

    def _fit(self, train: DataSource, val: DataSource | None) -> dict:
        prof = self.prof
        with prof.phase("setup"):
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._managers()
            steps_csv, epochs_csv = CsvLog(self.run_dir / "steps.csv"), CsvLog(self.run_dir / "metrics.csv")
            if self.resumed_from is not None:  # same run directory: drop the rows written after the checkpoint
                steps_csv.keep(lambda r: r["step"] <= self.global_step)
                epochs_csv.keep(lambda r: r["epoch"] < self.epoch)  # epoch self.epoch's end hooks have not run
        self._t_log, self._micro_since_log = time.perf_counter(), 0
        # data position (epoch, batch) = the next micro-batch; (e, micro_per_epoch) = epoch e trained, its end hooks due
        while self.epoch < self.s.max_epochs:
            epoch, start = self.epoch, self.batch
            if start == 0 and self.global_step >= self.total_steps:
                break
            if start == 0:
                with prof.phase("epoch_start"):
                    self._on_epoch_start(epoch)
            if start < self.micro_per_epoch and self.global_step < self.total_steps:
                batches = self._batches(train, epoch, start, train=True)
                try:
                    for b, batch in enumerate(batches, start=start):
                        if b >= self.micro_per_epoch:
                            break
                        self._train_micro(batch)
                        self.batch = b + 1
                        if self.micro_step % self.k:
                            continue
                        self.global_step += 1
                        if self.global_step % self.s.log_every_n_steps == 0:
                            with prof.phase("train/log"):
                                row = self._log_step(batch, epoch)
                                with prof.phase("csv"):
                                    steps_csv.append(row)
                        if self.step_manager is not None and self.step_manager.policy.step_due(self.global_step):
                            with prof.phase("train/ckpt"):
                                self._save(self.step_manager.save)
                        if self.global_step >= self.total_steps:
                            break
                finally:
                    close = getattr(batches, "close", None)  # a generator (prefetch / profiler); HostBatches has none
                    if close is not None:
                        close()
            ended = self.batch >= self.micro_per_epoch
            if not ended and self.global_step < self.total_steps:
                raise RuntimeError(f"epoch {epoch} ended after {self.batch} of {self.micro_per_epoch} micro-batches: the data source is short")
            if ended:
                self.epoch, self.batch = epoch + 1, 0
            with prof.phase("epoch_end"):
                prof.wait(self.state, "drain")
                self._on_epoch_end(epoch, val if ended else None, epochs_csv)
            if not ended:  # max_steps inside an epoch: Lightning runs no validation there
                break
        if self.step_manager is not None and self.step_manager.write_last:
            with prof.phase("train_end/ckpt_last"):
                self._save(self.step_manager.save_last)
        with prof.phase("train_end/drain"):
            prof.wait(self.state, "wait")
        with prof.phase("summary"):
            return self.summary()

    def _save(self, save) -> None:
        """``save(step, model, optimizer, state, meta)`` of a checkpoint manager; ``meta()`` syncs (EBOPs read-out)."""
        self.prof.wait(self.state)  # the save copies the state to the host next: blocks on it anyway
        with self.prof.phase("meta"):
            meta = self.meta()
        save(self.global_step, *self._save_args(), meta)

    def _on_epoch_start(self, epoch: int) -> None:
        if self.pid is None:
            return
        reading = self.last_ebops
        self._set_beta(self.pid.on_epoch_start(epoch, reading if reading is not None else 0.0))
        self.epoch_logs = {
            "train/quant_beta": self.beta,
            "train/pid_ebops": reading if reading is not None else math.nan,
            "train/pid_saturated": float(self.pid.saturated),
        }

    def _train_micro(self, batch: Batch) -> None:
        if self.global_step >= self.total_steps:  # torch OneCycleLR raises here ("Tried to step N times")
            raise RuntimeError(f"optimizer step {self.global_step + 1} exceeds total_steps {self.total_steps}")
        prof = self.prof
        if self.ramp is not None:
            with prof.phase("train/beta"):
                self._set_beta(ramp_beta(self.global_step, **self.ramp))
        ebops_on = self.ebops_active()
        if self.trace is not None:
            with prof.phase("trace"):
                self.trace.before(self.micro_step)
        t0 = time.perf_counter() if self.first_step_s is None else None
        with self._jit_phase("train/step"):
            self.state, out = self.steps.train_step(self.state, (batch.inputs, batch.targets), ebops_on=ebops_on)
        if t0 is not None:
            with prof.phase("train/first_step_wait"):
                jax.block_until_ready(out.loss)
            self.first_step_s = time.perf_counter() - t0
        else:
            prof.block(out.loss, "train/device")
        self.micro_step += 1
        self._micro_since_log += 1
        if ebops_on and self.beta > 0:
            self._reading = (out.penalty, self.beta)
        self._last = (out, ebops_on)
        if self.trace is not None:
            with prof.phase("trace"):
                self.trace.after(self.micro_step, self.state)

    def _log_step(self, batch: Batch, epoch: int) -> dict[str, float]:
        """The train row of a logged optimizer step (its last micro-batch); syncs the device."""
        prof = self.prof
        out, ebops_on = self._last
        with prof.phase("device_wait"):
            jax.block_until_ready(out.loss)
        now = time.perf_counter()
        row: dict[str, float] = {"epoch": float(epoch), "step": float(self.global_step)}
        with prof.phase("device_get"):
            host = jax.device_get({"losses": _flat_losses(out.losses, "train"), "parts": (out.task_loss, out.penalty, out.regularization, out.loss)})
        row.update({k: float(v) for k, v in host["losses"].items()})
        task, penalty, regularization, total = (float(v) for v in host["parts"])
        row["train/loss"] = task
        row["train/quant_ebops_loss"] = penalty + regularization
        row["train/total_loss"] = total
        row["train/ebops_active"] = float(ebops_on)
        row["train/quant_beta"] = math.nan if self.beta is None else self.beta
        with prof.phase("ebops"):
            ebops = self.last_ebops
        row["train/ebops"] = math.nan if ebops is None else ebops
        row["train/lr"] = float(self.schedule((self.global_step - 1) * self.k)) if callable(self.schedule) else float(self.schedule)
        row["time/ms_per_micro_batch"] = 1e3 * (now - self._t_log) / max(1, self._micro_since_log)
        with self._jit_phase("bits"):
            row.update(self.bit_stats("train"))
        with self._jit_phase("metrics_forward"):
            res = self._val(self.state, batch.inputs, batch.targets)
        prof.wait(res, "metrics_device")  # metrics.update reads the predictions next
        with prof.phase("metrics_host"):
            metrics = ClicMetrics("train", pool_layers=self.s.pool_layers)
            metrics.update(res["preds"], batch.targets, event_valid=batch.event_valid)
            row.update(metrics.compute())
        self.train_logs = row
        self._t_log, self._micro_since_log = time.perf_counter(), 0
        return row

    def validate(self, val: DataSource) -> dict[str, float]:
        """One validation epoch: event-weighted losses, ``ClicMetrics``, EBOPs and bitwidth monitors.

        Raises:
            RuntimeError: If the split gives no events.
        """
        prof = self.prof
        metrics = ClicMetrics("val", pool_layers=self.s.pool_layers)
        sums: dict[str, float] = defaultdict(float)
        events = 0
        for i, batch in enumerate(self._batches(val, 0, 0, train=False)):
            if self.s.limit_val_batches is not None and i >= self.s.limit_val_batches:
                break
            with self._jit_phase("forward"):
                res = self._val(self.state, batch.inputs, batch.targets)
            prof.wait(res, "device")  # the losses are read on the host below
            n = batch.num_real
            task, losses = res["task"], res["losses"]
            if n < self.batch_size:
                with self._jit_phase("tail"):
                    outputs = jax.tree_util.tree_map(lambda x, n=n: x[:n], res["outputs"])
                    targets = {k: v[:n] for k, v in batch.targets.items()}
                    task, losses = self._tail(outputs, targets)
            with prof.phase("device_get"):
                terms = jax.device_get({"val/loss": task, "val/quant_ebops_loss": res["regularization"], **_flat_losses(losses, "val")})
            for key, value in terms.items():
                sums[key] += float(value) * n
            events += n
            with prof.phase("metrics"):
                metrics.update(res["preds"], batch.targets, event_valid=batch.event_valid)
        if events == 0:
            raise RuntimeError("the validation split gave no events")
        out = {key: total / events for key, total in sums.items()}
        with prof.phase("metrics_compute"):
            out.update(metrics.compute())
        ebops = self.last_ebops
        out["val/ebops"] = math.nan if ebops is None else ebops
        with self._jit_phase("bits"):
            out.update(self.bit_stats("val"))
        self.val_events = events
        return out

    def _on_epoch_end(self, epoch: int, val: DataSource | None, epochs_csv: CsvLog) -> None:
        """Validation (a complete epoch only), the ``MetricsCsvWriter`` row, then the epoch / last checkpoints."""
        prof = self.prof
        validated = val is not None and (epoch + 1) % self.s.check_val_every_n_epoch == 0
        if validated:
            with prof.phase("val"):
                self.last_val = self.validate(val)
            with prof.phase("csv"):
                epochs_csv.append({**self.train_logs, **self.epoch_logs, **self.last_val, "epoch": float(epoch), "step": float(self.global_step)})
        # Lightning ModelCheckpoint._should_save_on_train_epoch_end: with check_val_every_n_epoch != 1 it saves only at
        # a validation end, so an epoch without validation writes no epoch file (it would carry the previous val/loss)
        if self.epoch_manager is not None and (validated or self.s.check_val_every_n_epoch == 1):
            if self.last_val is not None:
                with prof.phase("ckpt_epoch"):
                    self._save(lambda step, *args: self.epoch_manager.maybe_save_epoch(epoch, step, self.last_val, *args))
            else:
                warnings.warn(f"epoch {epoch}: no validation has run, so no epoch checkpoint (Lightning saves one without val/loss)", stacklevel=2)
        if self.step_manager is not None and self.step_manager.write_last:
            with prof.phase("ckpt_last"):
                self._save(self.step_manager.save_last)

    def summary(self) -> dict:
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": self.epoch,
            "total_steps": self.total_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "compile_counts": self.compile_counts(),
            "first_step_s": self.first_step_s,
            "resumed_from": None if self.resumed_from is None else str(self.resumed_from),
            "warm_started_from": None if self.warm_started_from is None else str(self.warm_started_from),
            "settings": asdict(self.s),
            "last_val": self.last_val,
            "beta": self.beta,
            "last_ebops": self.last_ebops,
        }

    # ------------------------------------------------------------------------------------------- test

    def test(self, test: DataSource, path: str | Path, num_events: int, limit_batches: int | None = None) -> dict[str, float]:
        """``PflowPredictionWriter`` h5 of the test split (matched outputs, padding events dropped) and test metrics."""
        writer = PredictionWriter(path, num_events, self.rc.data.dataset_kwargs["scale_dict_path"])
        metrics = ClicMetrics("test", pool_layers=self.s.pool_layers)
        for i, batch in enumerate(test.epoch(0)):
            if limit_batches is not None and i >= limit_batches:
                break
            res = self._val(self.state, batch.inputs, batch.targets)
            final_preds = res["preds"]["final"]
            writer.write_batch(res["final"], final_preds, batch.targets, batch.host["event_number"], batch.event_valid)
            metrics.update(res["preds"], batch.targets, event_valid=batch.event_valid)
        writer.close()
        return metrics.compute()


class _JitPhase:
    """``Trainer._jit_phase``: a profiler phase that is relabelled ``+compile`` when the trainer's trace counts change."""

    def __init__(self, trainer: Trainer, name: str) -> None:
        self.trainer, self.phase = trainer, trainer.prof.phase(name)

    def __enter__(self):
        self.before = sum(self.trainer.compile_counts().values())
        return self.phase.__enter__()

    def __exit__(self, *exc) -> None:
        self.phase.__exit__(*exc)
        if sum(self.trainer.compile_counts().values()) != self.before:
            self.trainer.prof.mark_compile()


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def host_state(state: TrainState) -> tuple[list, list, list]:
    """Host copies of a state (copies: a later donation never aliases them)."""
    return tuple([np.array(x, copy=True) for x in group] for group in state)
