r"""S6a checks on real CLIC events (``requiresdata``; run on Polaris or falcon, where the ROOT files are).

Data: ``$CLIC_DATA_DIR/val_clic_fix.root`` (default ``/eagle/hgcal-maskformer-fpga/clic_data``), the first
``$CLIC_NUM_EVENTS`` (default 2000) events, loaded and batched as the native driver does (``host_batches``, val split,
padded to batch 32). Skipped when the file is missing, unless ``CLIC_DATA_REQUIRED=1`` (then it fails).

Gates (carried from S4, plan section 4 / port_log README):
- padded nodes carry no target: incidence 0 and ``particle_node_valid`` False wherever ``node_valid`` is False;
- ``particle_valid`` is prefix-contiguous per event (the matcher solves the first ``n_valid`` target rows, as torch's
  ``Matcher`` does); ``particle_class`` is in [0, 5], and 5 exactly at the invalid particles;
- static int32 / float32 / bool batches.
Recorded, not gated (RETARGET_ORG_MAIN.md section 7, risk 6): the rate of equal phi values among valid nodes.

    CLIC_DATA_DIR=/eagle/hgcal-maskformer-fpga/clic_data KERAS_BACKEND=jax PYTHONPATH=src \\
        python -m pytest -m requiresdata tests/keras_native/test_data_real.py -s
"""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from hepattn.experiments.clic.native_data import host_batches, make_dataset
from hepattn.keras_native.config import load_config

from .golden.mf_cases import CONFIGS

DATA_DIR = Path(os.environ.get("CLIC_DATA_DIR", "/eagle/hgcal-maskformer-fpga/clic_data"))
NUM_EVENTS = int(os.environ.get("CLIC_NUM_EVENTS", "2000"))
VAL_FILE = DATA_DIR / "val_clic_fix.root"
STATIC = {np.dtype("int32"), np.dtype("float32"), np.dtype("bool")}

pytestmark = pytest.mark.requiresdata


@pytest.fixture(scope="module")
def batches():
    if not VAL_FILE.is_file():
        if os.environ.get("CLIC_DATA_REQUIRED") == "1":
            pytest.fail(f"{VAL_FILE} not found and CLIC_DATA_REQUIRED=1")
        pytest.skip(f"{VAL_FILE} not found (set CLIC_DATA_DIR)")
    cfg = load_config([CONFIGS / "base.yaml", CONFIGS / "linformer.yaml"]).data
    cfg = dataclasses.replace(cfg, valid_path=str(VAL_FILE), batch_size=32, num_workers=0)
    dataset = make_dataset(cfg, "val", num_events=NUM_EVENTS)
    return list(host_batches(cfg, "val", dataset=dataset, device_put=False))


def real_events(batches, tree: str, key: str) -> np.ndarray:
    return np.concatenate([getattr(b, tree)[key][b.event_valid] for b in batches])


def test_padded_nodes_carry_no_target(batches):
    node_valid = real_events(batches, "targets", "node_valid")
    pad = ~node_valid[:, None, :]
    incidence = real_events(batches, "targets", "particle_incidence")
    node_target = real_events(batches, "targets", "particle_node_valid")
    print(f"events {len(node_valid)}, padded node slots {int((~node_valid).sum())}")
    assert (~node_valid).any(axis=1).all(), "every event has a padded node (at most 159 of 160 are valid)"
    assert int(np.count_nonzero(np.where(pad, incidence, 0))) == 0
    assert int(np.count_nonzero(node_target & pad)) == 0


def test_particle_valid_is_prefix_contiguous_and_classes_match(batches):
    valid = real_events(batches, "targets", "particle_valid")
    classes = real_events(batches, "targets", "particle_class")
    n_valid = valid.sum(axis=1)
    prefix = np.arange(valid.shape[1])[None] < n_valid[:, None]
    broken = int((valid != prefix).any(axis=1).sum())
    print(f"events with non-prefix particle_valid: {broken} of {len(valid)}; class counts {np.bincount(classes.ravel(), minlength=6).tolist()}")
    assert broken == 0
    assert classes.min() >= 0 and classes.max() <= 5
    np.testing.assert_array_equal(classes == 5, ~valid)


def test_real_batches_are_static(batches):
    shapes = None
    for batch in batches:
        arrays = {**{f"in/{k}": v for k, v in batch.inputs.items()}, **{f"tgt/{k}": v for k, v in batch.targets.items()}}
        assert {np.dtype(v.dtype) for v in arrays.values()} <= STATIC
        current = {k: v.shape for k, v in arrays.items()}
        shapes = shapes or current
        assert current == shapes


def test_record_duplicate_phi_rate(batches):
    """Risk 6: equal phi values among valid nodes can be ordered differently by torch and JAX argsort."""
    events = with_dup = 0
    for batch in batches:
        phi, valid = batch.inputs["node_phi"], batch.inputs["node_valid"]
        for i in np.flatnonzero(batch.event_valid):
            values = phi[i][valid[i]]
            events += 1
            with_dup += int(len(np.unique(values)) < len(values))
    print(f"events with equal phi among valid nodes: {with_dup} of {events} ({with_dup / max(events, 1):.4%})")
