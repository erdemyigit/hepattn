"""CLICDataset(sectors=...) against the reference dataset and an independent reading of the ROOT branches.

Needs the CLIC files; set CLIC_DATA_DIR to the directory with val_clic_fix.root and test_clic_common_infer.root.
"""

import importlib.util
import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import torch
import uproot

from hepattn.experiments.clic.pflow_data import CLICDataset

ROOT = Path(__file__).resolve().parents[3]
CLIC = ROOT / "src/hepattn/experiments/clic"
DATA = Path(os.environ.get("CLIC_DATA_DIR", "data/clic"))
NUM_EVENTS = 400
SECTORS3 = {"num_sectors": 3, "scheme": "cap", "max_nodes_per_sector": 53, "track_phi": "calo", "overflow": "drop_lowest_energy"}
FIELDS = ["e", "pt", "eta", "sinphi", "cosphi"]

pytestmark = pytest.mark.requiresdata


def dataset(name, sectors=None, **kwargs):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"no {path}")
    return CLICDataset(
        str(path),
        inputs={},
        targets={"particle": FIELDS},
        scale_dict_path=str(CLIC / "configs/clic_var_transform.yaml"),
        num_events=NUM_EVENTS,
        incidence_cutval=0.01,
        sectors=sectors,
        **kwargs,
    )


def source_slots(ref_labels, sec_labels):
    """Reference slot of each valid sector slot, matched on the regression targets (unique per event)."""
    ref = torch.stack([ref_labels[f"particle_{f}"] for f in FIELDS], -1)
    sec = torch.stack([sec_labels[f"particle_{f}"] for f in FIELDS], -1)
    ref_valid = torch.nonzero(ref_labels["particle_valid"]).flatten()
    keys = {tuple(ref[p].tolist()): int(p) for p in ref_valid}
    assert len(keys) == len(ref_valid), "regression targets do not identify the particles"
    return {int(k): keys[tuple(sec[k].tolist())] for k in torch.nonzero(sec_labels["particle_valid"]).flatten()}


def test_one_sector_matches_the_reference():
    ref = dataset("val_clic_fix.root")
    sec = dataset("val_clic_fix.root", {**SECTORS3, "num_sectors": 1, "max_nodes_per_sector": 160, "track_phi": "vtx"})
    for i in range(len(ref)):
        ref_in, ref_lab = ref[i]
        sec_in, sec_lab = sec[i]
        for key, value in ref_in.items():
            assert torch.equal(sec_in[key], value), key
        valid, n = ref_in["node_valid"], int(ref_in["node_valid"].sum())
        assert (sec_in["node_sector"][:n] == 0).all()
        assert (sec_in["node_sector"][n:] == -1).all()
        # one sector: the rank is the phi order of the encoder sort
        assert torch.equal(sec_in["node_sector_rank"][:n], torch.argsort(torch.argsort(ref_in["node_phi"][valid])))
        target = ref_lab["particle_valid"]
        assert torch.equal(sec_lab["particle_valid"], target)
        for key, value in ref_lab.items():
            if key.startswith("particle_"):
                assert torch.equal(sec_lab[key][target], value[target]), key
        assert (sec_lab["particle_class"][~target] == 5).all()
        assert not sec_lab["particle_node_valid"][~target].any()
        assert sec_lab["sector_dropped_targets"] == 0


def test_blocks_restrict_the_reference_targets():
    ref = dataset("val_clic_fix.root")
    sec = dataset("val_clic_fix.root", SECTORS3)
    for i in range(len(ref)):
        _, ref_lab = ref[i]
        sec_in, sec_lab = sec[i]
        node_sector = sec_in["node_sector"]
        for j in range(3):
            block = sec_lab["particle_valid"][50 * j : 50 * (j + 1)]
            assert block[: int(block.sum())].all(), "valid targets are not a prefix of the block"
        slots = source_slots(ref_lab, sec_lab)
        assert len(slots) + int(sec_lab["sector_dropped_targets"]) == int(ref_lab["particle_valid"].sum())
        for k, p in slots.items():
            in_sector = node_sector == k // 50
            assert torch.equal(sec_lab["particle_node_valid"][k], ref_lab["particle_node_valid"][p] & in_sector)
            assert torch.equal(sec_lab["particle_incidence"][k], torch.where(in_sector, ref_lab["particle_incidence"][p], 0.0))
            assert sec_lab["particle_class"][k] == ref_lab["particle_class"][p]


def load_measure_sectors():
    spec = importlib.util.spec_from_file_location("measure_sectors", CLIC / "studies/sector_decoder/measure_sectors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MCAP[3] = SECTORS3["max_nodes_per_sector"]
    return module


def test_owners_match_measure_sectors():
    """Owner sectors and dropped targets against measure_sectors.py, which rebuilds them from the raw branches with
    its own partition (angle edges) and share code.
    """
    ms = load_measure_sectors()
    sec = dataset("val_clic_fix.root", SECTORS3)
    ref = dataset("val_clic_fix.root")
    entries = sec.event_number[: len(sec)]
    arrays = uproot.open(DATA / "val_clic_fix.root")["EventTree"].arrays(ms.BRANCHES, entry_stop=int(entries.max()) + 1)
    total_dropped = 0
    for i, entry in enumerate(entries):
        ev = {k: ak.to_numpy(arrays[k][entry]) for k in ms.BRANCHES}
        n_tr, n_p = len(ev["track_phi"]), len(ev["particle_e"])
        phi = np.concatenate([ev["track_phi_int"], ev["topo_phi"]]).astype(np.float64)
        node_sector_ms = ms.sector_index(phi, ms.boundaries(phi, 3, "cap"))
        w = np.zeros((n_p, len(phi)))
        tpi = ev["track_particle_idx"].astype(np.int64)
        w[tpi, np.arange(n_tr)] = ev["particle_e"][tpi]
        w[ev["topo2particle_particle_idx"].astype(np.int64), ev["topo2particle_topo_idx"].astype(np.int64) + n_tr] = ev["topo2particle_energy"]
        owner_ms = (w @ np.eye(3)[node_sector_ms]).argmax(1)

        sec_in, sec_lab = sec[i]
        _, ref_lab = ref[i]
        valid = ref_lab["particle_valid"][:n_p].numpy()
        e = ev["particle_e"].astype(np.float64)
        expected_drop = set()
        for j in range(3):
            members = np.flatnonzero(valid & (owner_ms == j))
            expected_drop |= set(members[np.argsort(-e[members], kind="stable")[50:]].tolist())
        slots = source_slots(ref_lab, sec_lab)
        assert set(np.flatnonzero(valid).tolist()) - set(slots.values()) == expected_drop
        total_dropped += len(expected_drop)
        node_sector = sec_in["node_sector"][: len(phi)].numpy()
        for k, p in slots.items():
            assert set(np.flatnonzero(node_sector == k // 50)) == set(np.flatnonzero(node_sector_ms == owner_ms[p])), (i, p)
    print(f"{len(entries)} events, {total_dropped} dropped targets")


def test_inference_file_keeps_every_event():
    ref = dataset("test_clic_common_infer.root", is_inference=True)
    sec = dataset("test_clic_common_infer.root", SECTORS3, is_inference=True)
    np.testing.assert_array_equal(sec.event_number, ref.event_number)
    for i in range(len(sec)):
        _, sec_lab = sec[i]
        _, ref_lab = ref[i]
        assert int(sec_lab["particle_valid"].sum()) + int(sec_lab["sector_dropped_targets"]) == int(ref_lab["particle_valid"].sum())
