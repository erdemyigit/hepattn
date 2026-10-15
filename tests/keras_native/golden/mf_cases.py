"""S3 MaskFormer cases: the pure-torch reference model, the matching native config, and the CLIC dummy batch.

Imports torch and ``hepattn.models`` only (no keras), so the hybrid golden writer (torch env) and the native tests
(JAX env) share it. The model is the CLIC test config of ``tests/keras/test_maskformer_parity.py`` (dim 32, 2 encoder
+ 2 decoder layers, 16 heads, 8 registers, 150 queries, 160 nodes, the four CLIC tasks), in two attention variants:
``torch`` (dense attention, as the hybrid tests) and ``lin`` (Linformer k = seq_len = 256 in encoder and decoder, as
``pflow_hgq.yaml``).
"""

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

import hepattn
from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder
from hepattn.models.input import InputNet
from hepattn.models.maskformer import MaskFormer
from hepattn.models.posenc import FourierPositionEncoder
from hepattn.models.task import IncidenceBasedRegressionTask, IncidenceRegressionTask, ObjectClassificationTask, ObjectHitMaskTask

try:  # imported as tests.keras_native.golden.mf_cases (native tests) or as a script sibling (make_golden_maskformer.py)
    from .cases import perturb_affine_params
except ImportError:
    from cases import perturb_affine_params

DIM, HEADS, NUM_QUERIES, MAX_NODES, REGS, LIN = 32, 16, 150, 160, 8, 256
SCALE_DICT = str(Path(hepattn.__file__).parent / "experiments/clic/configs/clic_var_transform.yaml")
FIELDS = ["e", "pt", "eta", "sinphi", "cosphi"]

LIN_KWARGS = {"num_heads": HEADS, "linformer_seq_len": LIN, "linformer_proj_dim": LIN}
VARIANTS = {
    "torch": {
        "encoder": {
            "num_layers": 2,
            "attn_type": "torch",
            "hybrid_norm": True,
            "value_residual": True,
            "num_register_tokens": REGS,
            "attn_kwargs": {"num_heads": HEADS},
        },
        "decoder_attn": {"num_heads": HEADS},
    },
    "lin": {
        "encoder": {
            "num_layers": 2,
            "attn_type": "linformer",
            "hybrid_norm": True,
            "value_residual": True,
            "num_register_tokens": REGS,
            "attn_kwargs": dict(LIN_KWARGS),
        },
        "decoder_attn": {**LIN_KWARGS, "attn_type": "linformer"},
    },
}

# Task specs as in the YAML (class_path + init_args); the Dense nets are plain kwargs. Shared by both frameworks.
TASK_SPECS = [
    {
        "class_path": "hepattn.models.task.ObjectClassificationTask",
        "init_args": {
            "name": "classification",
            "input_object": "query",
            "output_object": "pflow",
            "target_object": "particle",
            "num_classes": 5,
            "losses": {"object_ce": 2},
            "costs": {"object_ce": 2},
            "net": {"input_size": DIM, "output_size": 6, "hidden_layers": [256, 128, 32], "activation": "SiLU"},
            "null_weight": 0.5,
            "class_weights": [1.0, 3.0, 8.0, 1.5, 1.0],
            "mask_queries": False,
            "has_intermediate_loss": True,
        },
    },
    {
        "class_path": "hepattn.models.task.ObjectHitMaskTask",
        "init_args": {
            "name": "mask",
            "input_constituent": "node",
            "input_object": "query",
            "output_object": "pflow",
            "target_object": "particle",
            "pred_threshold": 0.1,
            "logit_scale": 4,
            "losses": {"mask_bce": 5.0, "mask_dice": 1.0},
            "costs": {"mask_dice": 1.0},
            "dim": DIM,
            "null_weight": 1.0,
            "has_intermediate_loss": True,
        },
    },
    {
        "class_path": "hepattn.models.task.IncidenceRegressionTask",
        "init_args": {
            "name": "incidence",
            "input_constituent": "node",
            "input_object": "query",
            "output_object": "pflow",
            "target_object": "particle",
            "losses": {"kl_div": 1.0},
            "costs": {"kl_div": 1.0},
            "net": {"input_size": DIM, "hidden_layers": 2, "activation": "SiLU"},
            "node_net": {"input_size": DIM, "hidden_layers": 1},
            "has_intermediate_loss": False,
        },
    },
    {
        "class_path": "hepattn.models.task.IncidenceBasedRegressionTask",
        "init_args": {
            "name": "regression",
            "fields": FIELDS,
            "input_constituent": "node",
            "input_object": "query",
            "output_object": "pflow",
            "target_object": "particle",
            "loss": "l1",
            "loss_weight": 10.0,
            "cost_weight": 10.0,
            "use_incidence": True,
            "use_nodes": True,
            "cost": "new",
            "mode": "scale",
            "scale_dict_path": SCALE_DICT,
            "net": {"input_size": 2 * DIM + len(FIELDS) + 1, "output_size": 5, "hidden_layers": [512, 256, 128, 64, 32], "activation": "SiLU"},
            "has_intermediate_loss": False,
        },
    },
]

INPUT_NET = {
    "input_name": "node",
    "fields": ["features"],
    "net": {"input_size": 27, "output_size": DIM},
    "posenc": {"class_path": "FourierPositionEncoder", "input_name": "node", "dim": DIM, "fields": ["eta", "phi"], "scale": 0.1},
}

TORCH_TASKS = {
    "ObjectClassificationTask": ObjectClassificationTask,
    "ObjectHitMaskTask": ObjectHitMaskTask,
    "IncidenceRegressionTask": IncidenceRegressionTask,
    "IncidenceBasedRegressionTask": IncidenceBasedRegressionTask,
}
NET_KEYS = ("net", "object_net", "constituent_net", "node_net")


def decoder_config(variant: str, **overrides) -> dict:
    cfg = {
        "num_decoder_layers": 2,
        "num_queries": NUM_QUERIES,
        "mask_attention": True,
        "use_query_masks": False,
        "decoder_layer_config": {"dim": DIM, "hybrid_norm": True, "attn_kwargs": dict(VARIANTS[variant]["decoder_attn"])},
    }
    cfg.update(overrides)
    return cfg


def task_specs(**init_overrides) -> list[dict]:
    """Deep copy of TASK_SPECS; ``init_overrides`` maps a task name to init_args updates."""
    specs = copy.deepcopy(TASK_SPECS)
    for spec in specs:
        spec["init_args"].update(init_overrides.get(spec["init_args"]["name"], {}))
    return specs


def native_kwargs(variant: str, decoder_overrides: dict | None = None, task_overrides: dict | None = None) -> dict:
    """``hepattn.keras_native.maskformer.MaskFormer`` kwargs (without ``quant``) of a case."""
    return {
        "input_nets": [copy.deepcopy(INPUT_NET)],
        "encoder": copy.deepcopy(VARIANTS[variant]["encoder"]),
        "decoder": decoder_config(variant, **(decoder_overrides or {})),
        "tasks": task_specs(**(task_overrides or {})),
        "dim": DIM,
    }


def torch_dense(cfg: dict) -> Dense:
    cfg = dict(cfg)
    act = cfg.pop("activation", None)
    if act is not None:
        cfg["activation"] = getattr(nn, act.removeprefix("torch.nn."))()
    return Dense(**cfg)


def torch_task(spec: dict) -> nn.Module:
    args = copy.deepcopy(spec["init_args"])
    for key in NET_KEYS:
        if key in args:
            args[key] = torch_dense(args[key])
    return TORCH_TASKS[spec["class_path"].rsplit(".", 1)[-1]](**args)


def torch_input_nets() -> nn.ModuleList:
    posenc = {k: v for k, v in INPUT_NET["posenc"].items() if k != "class_path"}
    return nn.ModuleList([
        InputNet(input_name="node", fields=["features"], net=torch_dense(INPUT_NET["net"]), posenc=FourierPositionEncoder(**posenc))
    ])


def torch_model(
    variant: str, seed: int = 0, decoder_overrides: dict | None = None, task_overrides: dict | None = None, perturb: bool = True
) -> MaskFormer:
    """The seeded pure-torch MaskFormer of a case (eval mode), biases and norm affines perturbed.

    No matcher: S3 compares forwards only. ``perturb=False`` with ``variant="torch"`` and seed 50 is exactly the model of
    ``tests/keras/test_quantized.py`` (the PARITY.md paired-delta rows). The perturbation (``cases.perturb_affine_params``) makes a missing port of a
    norm affine change the outputs at O(0.1) (default init would be 1 / 0 on both sides).
    """
    torch.manual_seed(seed)
    enc = copy.deepcopy(VARIANTS[variant]["encoder"])
    model = MaskFormer(
        input_nets=torch_input_nets(),
        encoder=Encoder(dim=DIM, **enc),
        decoder=MaskFormerDecoder(**decoder_config(variant, **(decoder_overrides or {}))),
        tasks=nn.ModuleList([torch_task(s) for s in task_specs(**(task_overrides or {}))]),
        dim=DIM,
    ).eval()
    if perturb:
        perturb_affine_params(model, seed)
    return model


def torch_state_digest(model: nn.Module) -> str:
    """sha256 over the sorted state_dict (key, dtype, shape, bytes); the golden files store it instead of the weights."""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        arr = value.detach().cpu().numpy()
        h.update(f"{key}|{arr.dtype}|{arr.shape}|".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def array_digest(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(f"{arr.dtype}|{arr.shape}|".encode() + arr.tobytes()).hexdigest()


def clic_dummy_batch(num_events: int = 2, offset: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Genuine CLIC dummy-data events (``CLICDataset(dummy_data=True)``) as numpy (int64 -> int32)."""
    from hepattn.experiments.clic.pflow_data import CLICDataset  # noqa: PLC0415  (imports lightning; only needed here)

    ds = CLICDataset(
        filepath="dummy",
        inputs={"node": ["features"]},
        targets={"particle": FIELDS},
        scale_dict_path=SCALE_DICT,
        num_events=num_events + offset,
        num_objects=NUM_QUERIES,
        max_nodes=MAX_NODES,
        dummy_data=True,
    )
    events = [ds[i] for i in range(offset, offset + num_events)]

    def stack(i: int) -> dict[str, np.ndarray]:
        out = {k: torch.stack([e[i][k] for e in events]).numpy() for k in events[0][i]}
        return {k: v.astype(np.int32) if v.dtype == np.int64 else v for k, v in out.items()}

    return stack(0), stack(1)


def flatten(tree: dict, prefix: str = "") -> dict:
    """Nested output dict -> {"a.b.c": leaf}; empty dicts vanish."""
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat
