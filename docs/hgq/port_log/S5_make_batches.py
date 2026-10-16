"""3 batches x 2 CLIC dummy events (CLICDataset dummy_data, offsets 0/2/4), padded-node targets zeroed (real-data-like)."""

import sys

import numpy as np
import torch

from hepattn.experiments.clic.pflow_data import CLICDataset

out, scale = sys.argv[1], sys.argv[2]
ds = CLICDataset(
    filepath="dummy",
    inputs={"node": ["features"]},
    targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
    scale_dict_path=scale,
    num_events=6,
    num_objects=150,
    max_nodes=160,
    dummy_data=True,
)
arrays = {"n": np.array(3)}
for b in range(3):
    events = [ds[2 * b + j] for j in range(2)]
    inputs = {k: torch.stack([e[0][k] for e in events]).numpy() for k in events[0][0]}
    targets = {k: torch.stack([e[1][k] for e in events]).numpy() for k in events[0][1]}
    nv = targets["node_valid"][:, None, :]
    targets["particle_node_valid"] = targets["particle_node_valid"] & nv
    targets["particle_incidence"] = np.where(nv, targets["particle_incidence"], 0.0).astype(np.float32)
    arrays.update({f"{b}::in::{k}": v for k, v in inputs.items()})
    arrays.update({f"{b}::tg::{k}": v for k, v in targets.items()})
np.savez(out, **arrays)
print(sorted(arrays)[:6], len(arrays))
