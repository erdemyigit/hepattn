"""S4 end to end: torch ``MaskFormer.loss`` vs the keras_native objective on real model outputs.

The torch model is the CLIC test configuration of ``tests/keras/test_maskformer_parity.py`` (dim 32,
2 encoder + 2 decoder layers, 150 queries, the four CLIC tasks) on ``CLICDataset`` dummy batches. Its
outputs are copied to numpy BEFORE ``MaskFormer.loss`` (which permutes them in place), then the native
costs, one host matching call, the permutation and the losses run in one ``jax.jit``.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from torch import nn

from hepattn.experiments.clic.pflow_data import CLICDataset
from hepattn.keras_native import objective as O
from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder
from hepattn.models.input import InputNet
from hepattn.models.maskformer import MaskFormer
from hepattn.models.matcher import Matcher
from hepattn.models.posenc import FourierPositionEncoder
from hepattn.models.wrapper import ModelWrapper

from .objective_utils import SCALE_DICT, assert_close_same_nonfinite, flatten, to_numpy, torch_tasks

DIM = 32
COST_TOL = {"atol": 1e-5, "rtol": 1e-5}
LOSS_TOL = {"atol": 1e-4, "rtol": 1e-3}


def _torch_model(seed: int) -> MaskFormer:
    torch.manual_seed(seed)
    input_nets = nn.ModuleList([
        InputNet(
            input_name="node",
            fields=["features"],
            net=Dense(input_size=27, output_size=DIM),
            posenc=FourierPositionEncoder(input_name="node", dim=DIM, fields=["eta", "phi"], scale=0.1),
        )
    ])
    return MaskFormer(
        input_nets=input_nets,
        encoder=Encoder(
            num_layers=2, dim=DIM, attn_type="torch", hybrid_norm=True, value_residual=True, num_register_tokens=8, attn_kwargs={"num_heads": 16}
        ),
        decoder=MaskFormerDecoder(
            num_decoder_layers=2,
            num_queries=150,
            mask_attention=True,
            use_query_masks=False,
            decoder_layer_config={"dim": DIM, "hybrid_norm": True, "attn_kwargs": {"num_heads": 16}},
        ),
        tasks=torch_tasks(DIM),
        dim=DIM,
        matcher=Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False),
    ).eval()


def _dummy_batch(num_events: int, seed: int) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    ds = CLICDataset(
        filepath="dummy",
        inputs={"node": ["features"]},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=SCALE_DICT,
        num_events=num_events,
        num_objects=150,
        max_nodes=160,
        dummy_data=True,
    )
    events = [ds[i] for i in range(num_events)]
    inputs = {k: torch.stack([e[0][k] for e in events]) for k in events[0][0]}
    targets = {k: torch.stack([e[1][k] for e in events]) for k in events[0][1]}
    return inputs, targets


def _native_targets(targets_t: dict) -> dict:
    out = to_numpy(targets_t)
    return {k: v.astype(np.int32) if v.dtype == np.int64 else v for k, v in out.items()}


def _real_like(targets: dict) -> dict:
    """Zero the mask targets at padded nodes and invalid particles (as real data has; dummy data does not)."""
    targets = dict(targets)
    keep = targets["node_valid"][:, None, :] & targets["particle_valid"][:, :, None]
    targets["particle_node_valid"] = targets["particle_node_valid"] & keep
    return targets


@pytest.mark.parametrize(("seed", "num_events", "real_like"), [(31, 2, False), (32, 4, False), (33, 4, True)])
def test_maskformer_loss_equals_torch(seed, num_events, real_like):
    model = _torch_model(seed)
    inputs, targets_t = _dummy_batch(num_events, seed)
    if real_like:
        targets_t = _real_like(targets_t)
    with torch.no_grad():
        outputs_t = model(inputs)
    outputs = {k: v for k, v in to_numpy(outputs_t).items() if k != "encoder"}
    targets = _native_targets(targets_t)
    specs = O.clic_specs()

    with torch.no_grad():
        costs_t = model._compute_decoder_costs({k: outputs_t[k] for k in outputs}, targets_t)  # noqa: SLF001
        _, _, losses_t = model.loss(outputs_t, dict(targets_t))

    costs = O.model_costs(outputs, targets, specs)
    assert set(costs) == set(costs_t) == {"layer_0", "layer_1", "final"}
    for layer, ref in costs_t.items():
        abs_err, rel_err = assert_close_same_nonfinite(ref.numpy(), np.asarray(costs[layer]), **COST_TOL, what=f"cost {layer}")
        print(f"[seed {seed}, B={num_events}] layer cost {layer}: max abs {abs_err:.2e}, max rel {rel_err:.2e}")

    total, losses, idx = jax.jit(lambda o, t: O.matched_loss(o, t, specs))(outputs, targets)
    flat_t = {k: v.numpy() for k, v in flatten(losses_t).items()}
    flat_n = {k: np.asarray(v) for k, v in flatten(losses).items()}
    assert set(flat_t) == set(flat_n), set(flat_t) ^ set(flat_n)
    assert len(flat_t) == 10, sorted(flat_t)
    for key in sorted(flat_t):
        r, g = flat_t[key], flat_n[key]
        # dummy data puts target 1 at padded nodes; since org a4263c1 mask_bce stays finite there (was inf)
        assert np.isfinite(r), f"{key}: torch {r}"
        abs_err, rel_err = assert_close_same_nonfinite(r, g, **LOSS_TOL, what=key)
        print(f"[seed {seed}, B={num_events}] loss {key}: {float(r):.6f}, max abs {abs_err:.2e}, max rel {rel_err:.2e}")
    # the scalar S5 trains on equals the hybrid's total (ModelWrapper.aggregate_losses, logging stubbed out)
    total_t = ModelWrapper.aggregate_losses(SimpleNamespace(model=model, log=lambda *_a, **_k: None), losses_t, stage="train").numpy()
    assert np.isfinite(total_t)
    abs_err, rel_err = assert_close_same_nonfinite(total_t, np.asarray(total), **LOSS_TOL, what="total loss")
    print(f"[seed {seed}, B={num_events}] total loss {float(total_t):.6f}: max abs {abs_err:.2e}, max rel {rel_err:.2e}")
    assert all(np.asarray(v).shape == (num_events, 150) for v in idx.values())
