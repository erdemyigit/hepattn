"""S6b gate: the command line from a clean shell (``python -m hepattn.experiments.clic.main_jax``) on CLIC dummy events.

Each run is a fresh process with only ``PATH``, ``HOME`` and ``PYTHONPATH`` set (no ``KERAS_BACKEND``: the entry point
selects JAX), on the tiny generated Polaris config:

1. ``fit`` 3 epochs with 2 spawned DataLoader workers (the ``__main__`` guard of the entry point is what makes spawn
   work), a step checkpoint every 2 optimizer steps, all kept, and a new compilation cache directory: checkpoints,
   ``steps.csv`` / ``metrics.csv``, ``summary.json``; the loss falls, the quantizer variables move, 0 cache hits;
2. ``fit --resume`` from the step-6 checkpoint of run 1 in a new process and run directory: the final checkpoint equals
   run 1's bit for bit, the logged rows after the resume equal run 1's, and the compiles are persistent-cache hits
   (control: run 1, with an empty cache, has none);
3. ``test`` on run 1's best checkpoint: the prediction h5 holds the 10 test events in order, without padding rows;
4. ``fit --resume last`` in a copy of run 1's finished directory: ``last`` is the step-12 checkpoint, written before
   epoch 2's end hooks, so the resume re-runs that validation (deviation 6); it writes ``run-r1.json`` /
   ``config-r1.yaml`` next to run 1's files and replaces, not repeats, the epoch-2 CSV row;
5. run 1's ``config.yaml`` holds the command-line options: ``load_config`` of it gives run 1's effective settings.
"""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from hepattn.keras_native import loop
from hepattn.keras_native.cli import apply_overrides
from hepattn.keras_native.config import load_config

from .s6b_utils import REPO, TINY, differing, npz_arrays

CLEAN_ENV_KEYS = ("PATH", "HOME", "TMPDIR")


def run_cli(*args, timeout=900) -> dict:
    env = {k: os.environ[k] for k in CLEAN_ENV_KEYS if k in os.environ}
    env["PYTHONPATH"] = str(REPO / "src")
    cmd = [sys.executable, "-m", "hepattn.experiments.clic.main_jax", *map(str, args)]
    proc = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("cli")
    cache = base / "jax_cache"
    common = ["--config", TINY, "--dummy-data", "--compile-cache", cache, "--ckpt-every-n-steps", 2, "--ckpt-save-top-k", -1]
    first = run_cli("fit", *common, "--run-dir", base / "run1", "--num-workers", 2)
    second = run_cli("fit", *common, "--run-dir", base / "run2", "--resume", base / "run1/ckpts/step-000000006.npz")
    best = base / "run1/ckpts" / json.loads((base / "run1/ckpts/best.json").read_text())["file"]
    tested = run_cli("test", "--config", TINY, "--dummy-data", "--compile-cache", cache, "--ckpt", best, "--out", base / "pred.h5")
    shutil.copytree(base / "run1", base / "run1_again")
    again = run_cli("fit", *common, "--run-dir", base / "run1_again", "--resume", "last")  # finished: no step runs
    return {"base": base, "first": first, "second": second, "test": tested, "again": again}


def test_fit_writes_checkpoints_logs_and_summary(runs):
    run1, first = runs["base"] / "run1", runs["first"]
    assert {p.name for p in run1.iterdir()} >= {"ckpts", "config.yaml", "run.json", "steps.csv", "metrics.csv", "summary.json"}
    assert sorted(p.name for p in (run1 / "ckpts").glob("step-*.npz")) == [f"step-{s:09d}.npz" for s in (2, 4, 6, 8, 10, 12)]
    assert len(list((run1 / "ckpts").glob("epoch-*.npz"))) == 3
    assert (first["global_step"], first["epoch"]) == (12, 3)
    assert first["compile_counts"] == {"train": 2, "eval": 0, "predict": 0, "val": 1, "tail": 1, "bits": 1}
    run = json.loads((run1 / "run.json").read_text())
    assert run["versions"]["hepattn_file"].startswith(str(REPO / "src"))
    assert run["settings"]["matmul_precision"] == "highest"


def test_loss_falls_and_quantizers_move(runs):
    run1 = runs["base"] / "run1"
    rows = loop.read_csv(run1 / "steps.csv")
    losses = [r["train/loss"] for r in rows]
    assert np.mean(losses[-4:]) < np.mean(losses[:4])
    epochs = loop.read_csv(run1 / "metrics.csv")
    assert epochs[-1]["val/loss"] < epochs[0]["val/loss"]
    changed = differing(npz_arrays(run1 / "ckpts/step-000000002.npz"), npz_arrays(run1 / "ckpts/step-000000012.npz"))
    quantizers = [k for k in changed if k.startswith("tv/") and k.rsplit("/", 1)[-1] in {"b", "f", "i"}]
    assert len(quantizers) > 10
    assert rows[-1]["train/bits_b_mean"] != rows[0]["train/bits_b_mean"]


def test_resume_in_a_new_process_is_bitwise_and_hits_the_compilation_cache(runs):
    base, first, second = runs["base"], runs["first"], runs["second"]
    final = "ckpts/step-000000012.npz"
    assert differing(npz_arrays(base / "run2" / final), npz_arrays(base / "run1" / final)) == []
    a = {r["step"]: r for r in loop.read_csv(base / "run1/steps.csv")}
    b = loop.read_csv(base / "run2/steps.csv")
    assert [r["step"] for r in b] == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    assert all(r["train/loss"] == a[r["step"]]["train/loss"] for r in b)
    assert second["resumed_from"].endswith("step-000000006.npz")
    # the train (EBOPs on / off) and val programs come from the cache written by run 1; run 1 started with none
    assert first["cache_events"]["cache_hits"] == 0
    assert second["cache_events"]["cache_hits"] >= 3
    assert second["cache_events"]["cache_misses"] == 0


def test_test_command_writes_the_prediction_file(runs):
    path = runs["base"] / "pred.h5"
    assert runs["test"]["h5"] == str(path)
    with h5py.File(path) as f:
        assert set(f) == {"events", "object_class", "object_masks", "incidence", "regression"}
        numbers = f["events"]["event_number"][:]
        assert numbers.shape == (10,)
        assert (numbers == np.arange(10)).all()  # dummy events are numbered by index; padding rows (-1) are dropped
        assert f["object_masks"].shape[0] == 10
        assert {"truth_masks", "mask_logits"} <= set(f["object_masks"].dtype.names)
    assert "test/final_obj_class_accuracy_micro" in runs["test"]["metrics"]


def test_resume_in_the_run_directory_keeps_the_first_run_files(runs):
    run1, copy, again = runs["base"] / "run1", runs["base"] / "run1_again", runs["again"]
    assert again["resumed_from"].endswith("step-000000012.npz") and again["global_step"] == 12
    assert (copy / "run.json").read_text() == (run1 / "run.json").read_text()  # the resume wrote its own file
    revision = json.loads((copy / "run-r1.json").read_text())
    assert revision["resume"].endswith("step-000000012.npz")
    # the resume's own options (no --num-workers 2); run 1's config.yaml stays as run 1 wrote it
    assert load_config(copy / "config-r1.yaml").data.num_workers == 0
    assert load_config(copy / "config.yaml").data.num_workers == 2
    assert len(loop.read_csv(copy / "steps.csv")) == 12  # nothing after the checkpoint, nothing repeated
    a, b = loop.read_csv(run1 / "metrics.csv"), loop.read_csv(copy / "metrics.csv")
    assert [r["epoch"] for r in b] == [0.0, 1.0, 2.0]  # epoch 2's row dropped, then written again
    assert b[-1]["val/loss"] == a[-1]["val/loss"]
    assert (copy / "ckpts/epoch-002-step-000000012-r1.npz").is_file()  # deviation 6: the re-run epoch-end save


def test_saved_config_holds_the_command_line_options(runs):
    """``fit --config <run>/config.yaml`` repeats the run: data, checkpoint and trainer settings and the model."""
    run1 = runs["base"] / "run1"
    saved = load_config(run1 / "config.yaml")
    args = argparse.Namespace(ckpt_every_n_steps=2, ckpt_save_top_k=-1, num_workers=2)
    effective = apply_overrides(load_config(TINY), args)
    assert saved.data == effective.data
    assert saved.checkpoint == effective.checkpoint
    assert saved.model_hash == effective.model_hash
    settings = json.loads((run1 / "run.json").read_text())["settings"]
    reread = dataclasses.asdict(loop.LoopSettings.from_run_config(saved, settings["run_dir"]))
    yaml_keys = ("max_epochs", "max_steps", "limit_train_batches", "limit_val_batches", "log_every_n_steps", "check_val_every_n_epoch", "seed")
    assert {k: reread[k] for k in yaml_keys} == {k: settings[k] for k in yaml_keys}
    # control: the YAML without the options (what config.yaml held before) is another run
    plain = load_config(TINY)
    assert plain.data.num_workers != saved.data.num_workers
    assert plain.checkpoint.step != saved.checkpoint.step
    assert dataclasses.replace(saved.checkpoint.step, every_n_train_steps=2, save_top_k=-1) == saved.checkpoint.step
