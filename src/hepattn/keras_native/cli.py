"""Command line of the keras_native driver (``fit`` and ``test``); the data source is injected by the experiment.

``hepattn.experiments.clic.main_jax`` is the entry point: it supplies the CLIC data (``CLICDataset`` + torch
``DataLoader``, which this torch-free package must not import) and calls ``main``.

    python -m hepattn.experiments.clic.main_jax fit --config polaris_hgq.yaml --run-dir runs/r1 [--resume last]
    python -m hepattn.experiments.clic.main_jax test --config polaris_hgq.yaml --ckpt runs/r1/ckpts/epoch-....npz

Several ``--config`` files merge in order, as LightningCLI does (``config.load_config``). The options below override
the YAML (they are the Lightning trainer / data arguments a Polaris job sets on the command line). Everything a run
writes goes under ``--run-dir``: ``config.yaml`` (the merged config with the command-line data, trainer, seed and
step-checkpoint options written in, so ``fit --config <run-dir>/config.yaml`` repeats the run; ``--matmul-precision``,
``--no-donate`` and ``--metrics-per-layer`` are in ``run.json`` only), ``run.json`` (options, versions, model hash),
``ckpts/``, ``steps.csv`` (every logged step), ``metrics.csv`` (every validation epoch, the hybrid
``MetricsCsvWriter`` layout) and ``summary.json`` (counters, compile counts, persistent-cache events of the latest
process). A later process in the same directory (a resume) writes ``config-r<n>.yaml`` / ``run-r<n>.json`` and keeps
the first ones, and drops the CSV rows written after its checkpoint. A run directory belongs to one run and its
resumes (checkpoint pruning looks at every file in ``ckpts``); the YAML's checkpoint and ``MetricsCsvWriter`` paths
are not used.

The persistent JAX compilation cache (``--compile-cache``, default ``$JAX_COMPILATION_CACHE_DIR`` or
``~/.cache/hepattn/jax``) lets a resumed or repeated run skip the ~2 min compile per program on an A100.
"""

import argparse
import copy
import dataclasses
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from hepattn.keras_native.config import CheckpointCallback, RunConfig, load_config

# data_factory(rc, split, *, seed, dummy_data) -> DataSource (with ``num_batches`` and ``num_events``)
DataFactory = Callable[..., object]
MATMUL_PRECISIONS = {"highest": "highest", "high": "high", "default": "default", "none": None}


def default_cache_dir() -> str:
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return os.environ["JAX_COMPILATION_CACHE_DIR"]
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return str(Path(base) / "hepattn" / "jax")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", action="append", required=True, help="YAML config; repeat to merge in order")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--dummy-data", action="store_true", help="CLICDataset dummy events instead of the ROOT files")
    p.add_argument("--seed", type=int, help="model init and data order seed (default: seed_everything of the YAML)")
    p.add_argument("--matmul-precision", choices=sorted(MATMUL_PRECISIONS), default="highest")
    p.add_argument("--compile-cache", default=default_cache_dir(), help="persistent JAX compilation cache ('' disables)")
    p.add_argument("--metrics-per-layer", action="store_true", help="ClicMetrics per layer (default: pooled, as the hybrid logs)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main_jax", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="train")
    _common(fit)
    fit.add_argument("--run-dir", required=True)
    fit.add_argument("--resume", help="checkpoint .npz, or 'last' (the run directory's last checkpoint)")
    fit.add_argument("--max-epochs", type=int)
    fit.add_argument("--max-steps", type=int, help="optimizer steps")
    fit.add_argument("--limit-train-batches", type=int, help="micro-batches per epoch")
    fit.add_argument("--limit-val-batches", type=int)
    fit.add_argument("--log-every-n-steps", type=int)
    fit.add_argument("--check-val-every-n-epoch", type=int)
    fit.add_argument("--num-train", type=int)
    fit.add_argument("--num-val", type=int)
    fit.add_argument("--train-path")
    fit.add_argument("--valid-path")
    fit.add_argument("--ckpt-every-n-steps", type=int, help="step checkpoints (adds a step callback if the YAML has none)")
    fit.add_argument("--ckpt-save-top-k", type=int, help="step checkpoints kept (-1: all)")
    fit.add_argument("--no-donate", action="store_true", help="do not donate the state to the train step")
    test = sub.add_parser("test", help="write the prediction h5 of a checkpoint")
    _common(test)
    test.add_argument("--ckpt", required=True)
    test.add_argument("--out", help="h5 path (default: <ckpt dir>/<ckpt stem>__test.h5)")
    test.add_argument("--num-test", type=int)
    test.add_argument("--test-path")
    test.add_argument("--limit-test-batches", type=int)
    test.add_argument("--run-dir", help="where test_metrics.json goes (default: the checkpoint's directory)")
    return parser


DATA_OPTIONS = ("batch_size", "num_workers", "num_train", "num_val", "num_test", "train_path", "valid_path", "test_path")
TRAINER_OPTIONS = ("max_epochs", "max_steps", "limit_train_batches", "limit_val_batches", "log_every_n_steps", "check_val_every_n_epoch")
STEP_CALLBACK = "lightning.pytorch.callbacks.ModelCheckpoint"


def apply_overrides(rc: RunConfig, args: argparse.Namespace) -> RunConfig:
    """``rc`` with the data and checkpoint options of ``args``."""
    data = {}
    for key in DATA_OPTIONS:
        value = getattr(args, key, None)
        if value is not None:
            data[key] = value
    rc = dataclasses.replace(rc, data=dataclasses.replace(rc.data, **data)) if data else rc
    every, top_k = getattr(args, "ckpt_every_n_steps", None), getattr(args, "ckpt_save_top_k", None)
    if every is not None or top_k is not None:
        step = rc.checkpoint.step or CheckpointCallback(
            monitor=None, mode="min", save_top_k=1, save_last=True, every_n_train_steps=1000, every_n_epochs=0, dirpath=None
        )
        step = dataclasses.replace(
            step, **({"every_n_train_steps": every} if every is not None else {}), **({"save_top_k": top_k} if top_k is not None else {})
        )
        rc = dataclasses.replace(rc, checkpoint=dataclasses.replace(rc.checkpoint, step=step))
    return rc


def effective_raw(rc: RunConfig, args: argparse.Namespace) -> dict:
    """``rc.raw`` with the command-line options written in: the YAML that ``load_config`` reads back to the same data,
    checkpoint and trainer settings (``rc`` is ``apply_overrides``'s result).
    """
    raw = copy.deepcopy(rc.raw)
    data = raw.setdefault("data", {})
    for key in DATA_OPTIONS:
        if getattr(args, key, None) is not None:
            data[key] = getattr(args, key)
    trainer = raw.setdefault("trainer", {})
    for key in TRAINER_OPTIONS:
        if getattr(args, key, None) is not None:
            trainer[key] = getattr(args, key)
    if getattr(args, "seed", None) is not None:
        raw["seed_everything"] = args.seed
    if getattr(args, "ckpt_every_n_steps", None) is not None or getattr(args, "ckpt_save_top_k", None) is not None:
        step = rc.checkpoint.step
        values = {"every_n_train_steps": step.every_n_train_steps, "save_top_k": step.save_top_k, "save_last": step.save_last}
        callbacks = trainer.setdefault("callbacks", []) or []
        trainer["callbacks"] = callbacks
        for i, entry in enumerate(callbacks):
            name = entry["class_path"] if isinstance(entry, dict) else str(entry)
            if name.rsplit(".", 1)[-1] == "ModelCheckpoint":
                entry = entry if isinstance(entry, dict) else {"class_path": name}
                entry["init_args"] = {**(entry.get("init_args") or {}), **values}
                callbacks[i] = entry
                break
        else:
            callbacks.append({"class_path": STEP_CALLBACK, "init_args": {**values, "monitor": step.monitor}})
    return raw


def _revision(path: Path) -> Path:
    """``path``, or ``<stem>-r<n><suffix>`` (next free ``n``) if it exists."""
    n, out = 0, path
    while out.exists():
        n += 1
        out = path.with_name(f"{path.stem}-r{n}{path.suffix}")
    return out


def _versions() -> dict:
    import hgq  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import keras  # noqa: PLC0415

    import hepattn  # noqa: PLC0415

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "keras": keras.__version__,
        "hgq2": getattr(hgq, "__version__", "?"),
        "device": str(jax.devices()[0]),
        "hepattn_file": hepattn.__file__,
    }


def _first(source) -> object:
    return next(iter(source.epoch(0, 0)))


def fit(args: argparse.Namespace, data_factory: DataFactory) -> dict:
    """The ``fit`` command. Returns the run summary (also written to ``summary.json``).

    Raises:
        SystemExit: For ``--resume last`` without a ``last`` pointer in the run directory.
    """
    from hepattn.keras_native import loop  # noqa: PLC0415  (imports jax; only after the backend check in main)

    events = loop.CompileEvents()
    cache = loop.enable_compilation_cache(args.compile_cache) if args.compile_cache else None
    rc = apply_overrides(load_config(args.config), args)
    run_dir = Path(args.run_dir)
    settings = loop.LoopSettings.from_run_config(
        rc,
        run_dir,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        seed=args.seed,
        matmul_precision=MATMUL_PRECISIONS[args.matmul_precision],
        donate=not args.no_donate,
        pool_layers=not args.metrics_per_layer,
    )
    dummy = True if args.dummy_data else None
    train = data_factory(rc, "train", seed=settings.seed, dummy_data=dummy)
    val = data_factory(rc, "val", seed=settings.seed, dummy_data=dummy) if rc.data.valid_path or dummy else None
    t0 = time.perf_counter()
    trainer = loop.Trainer(rc, settings, sample=_first(train), train_batches=train.num_batches)
    build_s = time.perf_counter() - t0
    resume = None
    if args.resume:
        resume = Path(args.resume)
        if args.resume == "last":
            pointer = run_dir / "ckpts" / "last.json"
            if not pointer.is_file():
                # the hepattn Checkpoint (epoch) callback forces save_last off, as in the hybrid: only a step callback
                # (the YAML's ModelCheckpoint or --ckpt-every-n-steps) writes ``last``
                raise SystemExit(f"--resume last: {pointer} does not exist (no step checkpoint callback?); pass an epoch checkpoint path")
            resume = run_dir / "ckpts" / json.loads(pointer.read_text())["file"]
        trainer.restore(resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    _revision(run_dir / "config.yaml").write_text(yaml.safe_dump(effective_raw(rc, args), sort_keys=False))
    loop.write_json(
        _revision(run_dir / "run.json"),
        {
            "argv": sys.argv,
            "args": vars(args),
            "settings": dataclasses.asdict(settings),
            "model_hash": rc.model_hash,
            "versions": _versions(),
            "compile_cache": None if cache is None else str(cache),
            "total_steps": trainer.total_steps,
            "resume": None if resume is None else str(resume),
        },
    )
    t1 = time.perf_counter()
    summary = trainer.fit(train, val)
    summary.update(
        build_s=build_s, fit_s=time.perf_counter() - t1, compile_cache=None if cache is None else str(cache), cache_events=dict(events.counts)
    )
    loop.write_json(run_dir / "summary.json", summary)
    return summary


def test(args: argparse.Namespace, data_factory: DataFactory) -> dict:
    """The ``test`` command: prediction h5 of ``--ckpt`` and the test metrics."""
    from hepattn.keras_native import loop  # noqa: PLC0415
    from hepattn.keras_native.predict_writer import output_path  # noqa: PLC0415

    if args.compile_cache:
        loop.enable_compilation_cache(args.compile_cache)
    rc = apply_overrides(load_config(args.config), args)
    ckpt = Path(args.ckpt)
    run_dir = Path(args.run_dir) if args.run_dir else ckpt.parent
    settings = loop.LoopSettings.from_run_config(
        rc, run_dir, seed=args.seed, matmul_precision=MATMUL_PRECISIONS[args.matmul_precision], pool_layers=not args.metrics_per_layer
    )
    dummy = True if args.dummy_data else None
    source = data_factory(rc, "test", seed=settings.seed, dummy_data=dummy)
    meta = loop.ckpt.read_meta(ckpt)
    trainer = loop.Trainer(rc, settings, sample=_first(source), train_batches=meta["micro_per_epoch"])
    trainer.total_steps = meta["total_steps"]  # restore() checks the schedule length; the test stage does not step
    trainer.micro_per_epoch = meta["micro_per_epoch"]
    trainer.restore(ckpt, check_seed=False)
    out = Path(args.out) if args.out else output_path(ckpt)
    metrics = trainer.test(source, out, source.num_events, limit_batches=args.limit_test_batches)
    result = {"h5": str(out), "ckpt": str(ckpt), "metrics": metrics, "compile_counts": trainer.compile_counts()}
    loop.write_json(run_dir / "test_metrics.json", result)
    return result


def main(argv: list[str] | None, data_factory: DataFactory) -> int:
    """Parse ``argv`` and run the command.

    Raises:
        SystemExit: If keras is not on the JAX backend.
    """
    args = build_parser().parse_args(argv)
    import keras  # noqa: PLC0415

    if keras.backend.backend() != "jax":
        raise SystemExit(f"the keras_native driver needs KERAS_BACKEND=jax, keras runs {keras.backend.backend()!r}")
    result = fit(args, data_factory) if args.command == "fit" else test(args, data_factory)
    print("RESULT " + json.dumps(result, sort_keys=True, default=str), flush=True)
    return 0
