"""Hybrid (Lightning, torch backend) side of gate G4 (``polaris/19_g4_compare.py``): ``main_hgq`` with G4 additions.

Runs in the hybrid env (``.venv``, ``KERAS_BACKEND=torch``). Two commands; every other argument goes to the
LightningCLI of ``main_hgq`` (``MPflowHGQ``) with ``G4DataModule`` as the datamodule:

    python polaris/19_g4_hybrid.py init --config <g4 config> --trainer.default_root_dir <dir>    (env G4_INIT_DIR)
    python polaris/19_g4_hybrid.py fit  --config <arm config> --data.data_seed <s> ...           (env G4_ARM_DIR)

``init`` builds the model (``seed_everything`` of the config), writes the full Lightning checkpoint
``$G4_INIT_DIR/hybrid_init.ckpt`` at ``on_train_start`` (no batch trained: step 0), and the model's eval outputs on the
first validation batch (``hybrid_forward.npz``: ``in::<key>`` / ``out::<path>``, as the S7 fixtures), then stops.

``fit`` adds these callbacks (``G4_*`` environment variables, set by ``19_g4_compare.py run``):

- ``G4WarmStart`` (``G4_WARM_START``): load the model state of the init checkpoint at ``on_fit_start`` of a fresh run
  (weights, quantizer state, beta; optimizer, schedule, loops, BetaPID and data position fresh), as ``main_jax fit
  --init-from``. A resumed run (``--ckpt_path``) does not get this variable.
- ``G4StepLog`` (``G4_ARM_DIR``): ``steps.csv`` with the native driver's train columns at the native cadence (every
  ``log_every_n_steps`` optimizer steps: task loss of the step's last micro-batch, the HGQ2 term, beta, EBOPs, the LR
  of that update, wall ms per micro-batch since the previous row) and ``progress.json`` (latest step, epoch).
- ``G4Kill`` (``G4_KILL_AT_STEP``, tests only): ``os._exit(75)`` at the first train batch after that step, i.e. right
  after its step checkpoint, as a walltime kill.

Data order (``G4DataModule``): the train order of epoch ``e`` is the native driver's ``EpochSampler(seed + e)``
(``hepattn.experiments.clic.native_data``) with ``seed = data.data_seed``, so a hybrid arm and a JAX arm with the same
seed see the same events in the same order. The sampler reads the data position from the trainer when an epoch's
iterator is made: ``start = global_step * accumulate_grad_batches - epoch * limit_train_batches`` micro-batches, so a
resume from a step checkpoint continues the epoch where it stopped (Lightning restarts the iterator of a non-stateful
loader at the start of the epoch otherwise). ``limit_train_batches`` must be an integer multiple of the accumulation.
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import comet_ml  # noqa: F401  (before lightning and torch, as main_hgq: the CLI resolves the Comet logger)
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from hepattn.experiments.clic.pflow_data import PflowDataModule

STEP_COLUMNS = (
    "epoch",
    "step",
    "train/loss",
    "train/quant_ebops_loss",
    "train/quant_beta",
    "train/ebops",
    "train/lr",
    "time/ms_per_micro_batch",
)


def data_position(global_step: int, epoch: int, accum: int, micro_per_epoch: int) -> int:
    """The next micro-batch of ``epoch`` after ``global_step`` optimizer steps (epochs of ``micro_per_epoch``).

    Raises:
        ValueError: If the position is outside the epoch (the trainer counters do not fit the epoch length).
    """
    start = global_step * accum - epoch * micro_per_epoch
    if not 0 <= start <= micro_per_epoch:
        raise ValueError(f"step {global_step} x accum {accum} is not in epoch {epoch} of {micro_per_epoch} micro-batches")
    return start


def epoch_order(num_events: int, seed: int, epoch: int, batch_size: int, start_batch: int = 0) -> list[int]:
    """The train indices of ``epoch`` from micro-batch ``start_batch`` on: ``native_data.EpochSampler`` (shuffle, drop_last).

    A copy, because ``native_data`` imports ``hepattn.keras_native`` (jax), which the hybrid env does not have;
    ``tests/polaris/test_g4_compare.py`` checks it against ``EpochSampler`` (JAX env).
    """
    order = torch.randperm(num_events, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    order = order[: len(order) - len(order) % batch_size]
    return order[start_batch * batch_size :]


class TrainerEpochSampler(Sampler[int]):
    """The native train order, with the epoch and start batch read from the trainer at ``iter`` (module docstring)."""

    def __init__(self, num_events: int, *, batch_size: int, seed: int, position) -> None:
        self.num_events, self.batch_size, self.seed = num_events, batch_size, seed
        self._position = position  # () -> (epoch, start_batch) or None (no trainer yet)
        self.epoch, self.start_batch = 0, 0
        self.history: list[tuple[int, int]] = []

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        """Ignored: Lightning calls it with the epoch only, after the epoch's iterator may already exist."""

    def __iter__(self):
        pos = self._position()
        if pos is not None:
            self.epoch, self.start_batch = pos
        self.history.append((self.epoch, self.start_batch))
        return iter(epoch_order(self.num_events, self.seed, self.epoch, self.batch_size, self.start_batch))

    def __len__(self) -> int:
        return max(0, self.num_events - self.num_events % self.batch_size - self.start_batch * self.batch_size)


class G4DataModule(PflowDataModule):
    """``PflowDataModule`` whose train loader follows the native driver's order (module docstring)."""

    def __init__(
        self,
        train_path: str,
        valid_path: str,
        batch_size: int,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        scale_dict_path: str,
        test_path: str | None = None,
        pin_memory: bool = True,
        test_suff: str | None = None,
        data_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            train_path=train_path,
            valid_path=valid_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            scale_dict_path=scale_dict_path,
            test_path=test_path,
            pin_memory=pin_memory,
            test_suff=test_suff,
            **kwargs,
        )
        self.data_seed = int(data_seed)
        self.sampler: TrainerEpochSampler | None = None

    def _position(self) -> tuple[int, int] | None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        limit = trainer.limit_train_batches
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError(f"G4 needs an integer trainer.limit_train_batches, got {limit!r}")
        accum = trainer.accumulate_grad_batches
        if limit % accum:
            raise ValueError(f"limit_train_batches {limit} is not a multiple of accumulate_grad_batches {accum}")
        epoch = trainer.current_epoch
        return epoch, data_position(trainer.global_step, epoch, accum, limit)

    def train_dataloader(self):
        self.sampler = TrainerEpochSampler(len(self.train_dset), batch_size=self.batch_size, seed=self.data_seed, position=self._position)
        return DataLoader(
            dataset=self.train_dset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def flatten(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        out.update(flatten(value, f"{name}.") if isinstance(value, dict) else {name: value})
    return out


def _beta(model) -> float:
    from hepattn.keras.callbacks import quantized_leaves  # noqa: PLC0415

    for layer in quantized_leaves(model):
        var = getattr(layer, "_beta", None)
        if var is not None:
            return float(torch.as_tensor(var.value))
    return math.nan


class _InitSaved(Exception):  # noqa: N818  (control flow: stops the init fit after the checkpoint)
    pass


class G4SaveInit(pl.Callback):
    """``init``: the step-0 checkpoint and the eval outputs on the first validation batch, then stop."""

    def __init__(self, out_dir: str) -> None:
        self.out_dir = Path(out_dir)

    @torch.no_grad()
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if trainer.global_step != 0:
            raise RuntimeError(f"init checkpoint at step {trainer.global_step}, expected 0")
        trainer.save_checkpoint(self.out_dir / "hybrid_init.ckpt")
        inputs, _ = next(iter(trainer.datamodule.val_dataloader()))
        model = pl_module.model
        was_training = model.training
        model.eval()
        outputs = flatten(model({k: v.to(pl_module.device) for k, v in inputs.items()}))
        model.train(was_training)
        arrays = {f"in::{k}": v.numpy() for k, v in inputs.items()}
        arrays |= {f"out::{k}": v.detach().cpu().numpy() for k, v in outputs.items()}
        np.savez(self.out_dir / "hybrid_forward.npz", **arrays)
        raise _InitSaved


class G4WarmStart(pl.Callback):
    """Load the init checkpoint's model state into a fresh run (module docstring)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_step != 0:
            raise RuntimeError(f"G4_WARM_START on a run at step {trainer.global_step}: a resumed run must not be warm-started")
        state = torch.load(self.path, map_location="cpu", weights_only=True)["state_dict"]
        pl_module.load_state_dict(state, strict=True)
        print(f"[G4] warm start from {self.path}: {len(state)} tensors", flush=True)


class G4StepLog(pl.Callback):
    """``steps.csv`` and ``progress.json`` of a hybrid arm (module docstring)."""

    def __init__(self, arm_dir: str) -> None:
        self.dir = Path(arm_dir)
        self.lr = math.nan
        self._step_seen = -1
        self._t, self._micro = time.perf_counter(), 0

    def _rows_up_to(self, step: int) -> None:
        """A resumed process drops the rows its predecessor wrote after the checkpoint (as the native driver)."""
        path = self.dir / "steps.csv"
        if not path.is_file():
            return
        with path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if float(r["step"]) <= step]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, STEP_COLUMNS)
            w.writeheader()
            w.writerows(rows)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._rows_up_to(trainer.global_step)
        self._step_seen = trainer.global_step
        self._t, self._micro = time.perf_counter(), 0

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer) -> None:
        self.lr = float(optimizer.param_groups[0]["lr"])  # the LR of this update (the scheduler steps after it)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        from hepattn.keras.callbacks import total_ebops  # noqa: PLC0415

        self._micro += 1
        step = trainer.global_step
        if step == self._step_seen:
            return
        self._step_seen = step
        if step % trainer.log_every_n_steps:
            return
        now = time.perf_counter()
        metrics = trainer.callback_metrics
        row = {
            "epoch": trainer.current_epoch,
            "step": step,
            "train/loss": float(metrics["train/loss"]),
            "train/quant_ebops_loss": float(metrics.get("train/quant_ebops_loss", math.nan)),
            "train/quant_beta": _beta(pl_module.model),
            "train/ebops": total_ebops(pl_module.model),
            "train/lr": self.lr,
            "time/ms_per_micro_batch": 1e3 * (now - self._t) / max(1, self._micro),
        }
        path = self.dir / "steps.csv"
        new = not path.is_file()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, STEP_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        (self.dir / "progress.json").write_text(json.dumps({"global_step": step, "epoch": trainer.current_epoch}) + "\n")
        if os.environ.get("G4_LOG_TERMS"):  # every scalar train/ value of the row (per-layer / per-task loss terms)
            terms = {k: float(v) for k, v in metrics.items() if k.startswith("train/") and getattr(v, "numel", lambda: 1)() == 1}
            with (self.dir / "terms.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **terms}) + "\n")
        self._t, self._micro = time.perf_counter(), 0

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        (self.dir / "progress.json").write_text(json.dumps({"global_step": trainer.global_step, "epoch": trainer.current_epoch}) + "\n")


class G4Kill(pl.Callback):
    """Tests: die like a walltime kill at the first train batch after step ``at`` (its checkpoint is written)."""

    def __init__(self, at: int) -> None:
        self.at = at

    def on_train_batch_start(self, trainer: pl.Trainer, *_args) -> None:
        if trainer.global_step >= self.at:
            print(f"[G4] simulated kill at step {trainer.global_step}", flush=True)
            sys.stdout.flush()
            os._exit(75)


def g4_callbacks(command: str) -> list[pl.Callback]:
    if command == "init":
        return [G4SaveInit(os.environ["G4_INIT_DIR"])]
    cbs: list[pl.Callback] = []
    if os.environ.get("G4_WARM_START"):
        cbs.append(G4WarmStart(os.environ["G4_WARM_START"]))
    cbs.append(G4StepLog(os.environ["G4_ARM_DIR"]))
    if os.environ.get("G4_KILL_AT_STEP"):
        cbs.append(G4Kill(int(os.environ["G4_KILL_AT_STEP"])))
    return cbs


_ORIG_TRAINER_INIT = pl.Trainer.__init__
_EXTRA_CALLBACKS: list[pl.Callback] = []


def _init_with_g4(self, *args, **kwargs):
    # module level and calling the original by its global name, as polaris/13_step_time_fit.py: jsonargparse reads the
    # Trainer's arguments through this call (a closure hides them: "No action for key trainer.logger...")
    cbs = kwargs.get("callbacks") or []
    kwargs["callbacks"] = [*(cbs if isinstance(cbs, list) else [cbs]), *_EXTRA_CALLBACKS]
    return _ORIG_TRAINER_INIT(self, *args, **kwargs)


def main(argv: list[str]) -> int:
    command, rest = argv[0], argv[1:]
    if command not in {"init", "fit"}:
        raise SystemExit(f"usage: 19_g4_hybrid.py init|fit <LightningCLI arguments>, got {command!r}")
    _EXTRA_CALLBACKS[:] = g4_callbacks(command)
    sys.argv[1:] = []  # the CLI gets its arguments through args= only
    pl.Trainer.__init__ = _init_with_g4

    from hepattn.experiments.clic.lightning_module_hgq import MPflowHGQ  # noqa: PLC0415
    from hepattn.experiments.clic.main_hgq import config_dir  # noqa: PLC0415
    from hepattn.utils.cli import CLI  # noqa: PLC0415

    try:
        CLI(
            model_class=MPflowHGQ,
            datamodule_class=G4DataModule,
            args=["fit", *rest],
            parser_kwargs={"default_env": True, "fit": {"default_config_files": [f"{config_dir}/pflow_hgq.yaml"]}},
        )
    except _InitSaved:
        print(f"[G4] init checkpoint written to {os.environ['G4_INIT_DIR']}", flush=True)
    print(f"[G4-TORCH-MATMUL-PRECISION] {torch.get_float32_matmul_precision()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
