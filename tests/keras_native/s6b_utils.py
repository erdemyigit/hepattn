"""Shared setup of the S6b driver tests: the tiny generated Polaris config on CLIC dummy events."""

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from hepattn.experiments.clic.main_jax import ClicSource
from hepattn.keras_native import loop
from hepattn.keras_native.cli import apply_overrides
from hepattn.keras_native.config import load_config

REPO = Path(__file__).resolve().parents[2]
TINY = REPO / "tests/keras_native/s6b_fixtures/tiny_polaris.yaml"


def tiny_config(**overrides):
    """The tiny config with CLI-style overrides (``ckpt_every_n_steps``, ``ckpt_save_top_k``, ``num_val``, ...)."""
    args = argparse.Namespace(**overrides)
    return apply_overrides(load_config(TINY), args)


def sources(rc, seed: int = 42) -> tuple[ClicSource, ClicSource]:
    return ClicSource(rc, "train", seed=seed, dummy_data=True), ClicSource(rc, "val", seed=seed, dummy_data=True)


def trainer(rc, run_dir, train: ClicSource, **settings) -> loop.Trainer:
    s = loop.LoopSettings.from_run_config(rc, run_dir, **settings)
    return loop.Trainer(rc, s, sample=next(iter(train.epoch(0))), train_batches=train.num_batches)


def npz_arrays(path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k != "__meta__"}


def differing(a: dict, b: dict) -> list[str]:
    """Keys whose arrays differ in value, dtype or shape (NaN equal to NaN); a key set difference is an error."""
    assert set(a) == set(b), set(a) ^ set(b)
    return [k for k in a if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]


def replace_quant(rc, **changes):
    return dataclasses.replace(rc, quant=dataclasses.replace(rc.quant, **changes))
