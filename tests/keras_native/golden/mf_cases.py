"""MaskFormer cases: the pure-torch reference models, the matching native configs, and the CLIC dummy batch.

Imports torch and ``hepattn.models`` only (no keras), so the hybrid golden writer (torch env) and the native tests
(JAX env) share it. Variants:
- ``torch`` and ``lin``: the S3 CLIC test config of ``tests/keras/test_maskformer_parity.py`` (dim 32, 2 encoder
  + 2 decoder layers, 16 heads, 8 registers, 150 queries, 160 nodes, the four CLIC tasks) with dense attention
  (``torch``) or org/main's masked-Linformer attention (``lin``, k = 64; ``linformer_seq_len`` 168 in the encoder,
  160 in the decoder, as ``linformer.yaml``).
- ``org``: the org/main ``ac0da8f`` model itself, ``base.yaml`` with ``linformer.yaml`` merged over it, at the test
  width (``ORG_DIM`` = 32; depth, heads, k, tasks, sorting and every other key as the YAML). Built from the config
  (``org_init_args``): jsonargparse-style instantiation for torch, ``from_config`` for native, ``KerasMaskFormer``
  kwargs for the hybrid (``hybrid_org_kwargs``).
"""

import copy
import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
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

DIM, HEADS, NUM_QUERIES, MAX_NODES, REGS = 32, 16, 150, 160, 8
SCALE_DICT = str(Path(hepattn.__file__).parent / "experiments/clic/configs/clic_var_transform.yaml")
FIELDS = ["e", "pt", "eta", "sinphi", "cosphi"]

LIN_K = 64  # linformer.yaml linformer_k
# linformer.yaml: the encoder keys are the nodes + registers, the decoder's longest key axis the nodes
LIN_ENC_KWARGS = {"num_heads": HEADS, "linformer_seq_len": MAX_NODES + REGS, "linformer_k": LIN_K}
LIN_DEC_KWARGS = {"num_heads": HEADS, "linformer_seq_len": MAX_NODES, "linformer_k": LIN_K}
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
            "attn_kwargs": dict(LIN_ENC_KWARGS),
        },
        "decoder_attn": {**LIN_DEC_KWARGS, "attn_type": "linformer"},
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

ORG_VARIANTS = {"org"}
ORG_DIM = DIM
CONFIGS = Path(hepattn.__file__).parent / "experiments/clic/configs"


def deep_merge(base: dict, over: dict) -> dict:
    """``over`` merged into ``base``: dicts merge key by key, anything else (lists included) replaces."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def org_init_args(dim: int = ORG_DIM, decoder_layer_overrides: dict | None = None, sort: bool = True) -> dict:
    """``model.model.init_args`` of org ``base.yaml`` + ``linformer.yaml`` at width ``dim``, without the matcher.

    ``dim`` replaces the YAML anchor ``&dim`` (and the overlay's decoder ``dim``, the regression input 2 dim + 6);
    the scale dict path is made absolute. ``decoder_layer_overrides`` edits ``decoder.decoder_layer_config`` and
    ``sort=False`` drops ``input_sort_field`` (controls).
    """
    base_text = (CONFIGS / "base.yaml").read_text()
    over_text = (CONFIGS / "linformer.yaml").read_text()
    for old, new, which in (
        ("dim: &dim 256", f"dim: &dim {dim}", "base"),
        ("input_size: 518", f"input_size: {2 * dim + 6}", "base"),
        ("dim: 256", f"dim: {dim}", "over"),
    ):
        text = base_text if which == "base" else over_text
        assert text.count(old) == 1, f"{which}: '{old}' found {text.count(old)} times"
        if which == "base":
            base_text = base_text.replace(old, new)
        else:
            over_text = over_text.replace(old, new)
    init = deep_merge(yaml.safe_load(base_text), yaml.safe_load(over_text))["model"]["model"]["init_args"]
    init.pop("matcher")
    for task in init["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task["init_args"]:
            task["init_args"]["scale_dict_path"] = SCALE_DICT
    init["decoder"]["decoder_layer_config"].update(decoder_layer_overrides or {})
    if not sort:
        init["input_sort_field"] = None
    return init


def instantiate(spec):
    """jsonargparse-style instantiation of a YAML spec (``class_path`` / ``init_args``; ``torch.nn.*`` strings)."""
    if isinstance(spec, dict) and "class_path" in spec:
        module, name = spec["class_path"].rsplit(".", 1)
        args = {key: instantiate(value) for key, value in (spec.get("init_args") or {}).items()}
        return getattr(importlib.import_module(module), name)(**args)
    if isinstance(spec, list):
        return [instantiate(value) for value in spec]
    if isinstance(spec, str) and spec.startswith("torch.nn."):
        return getattr(nn, spec.removeprefix("torch.nn."))()
    return spec


def native_org_config(**org_kwargs) -> dict:
    """The ``model`` entry for ``hepattn.keras_native.maskformer.from_config``."""
    return {"class_path": "hepattn.models.MaskFormer", "init_args": org_init_args(**org_kwargs)}


def hybrid_org_kwargs(**org_kwargs) -> dict:
    """``KerasMaskFormer`` kwargs (without ``quant``) of the org model: fresh torch input nets and tasks."""
    init = org_init_args(**org_kwargs)
    encoder = dict(init["encoder"]["init_args"])
    assert encoder.pop("dim") == init["dim"]
    return {
        "input_nets": instantiate(init["input_nets"]),
        "encoder": encoder,
        "decoder": init["decoder"],
        "tasks": instantiate(init["tasks"]),
        "dim": init["dim"],
        "input_sort_field": init["input_sort_field"],
    }


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

    No matcher: S3 compares forwards only. For ``org``, ``decoder_overrides`` are ``org_init_args`` kwargs
    (``decoder_layer_overrides``, ``sort``). ``perturb=False`` with ``variant="torch"`` and seed 50 is exactly the model of
    ``tests/keras/test_quantized.py`` (the PARITY.md paired-delta rows). The perturbation (``cases.perturb_affine_params``) makes a missing port of a
    norm affine change the outputs at O(0.1) (default init would be 1 / 0 on both sides).

    Raises:
        ValueError: For ``task_overrides`` with the ``org`` variant.
    """
    torch.manual_seed(seed)
    if variant in ORG_VARIANTS:
        if task_overrides:
            raise ValueError("task_overrides are not supported for the org variant")
        init = org_init_args(**(decoder_overrides or {}))
        decoder = MaskFormerDecoder(**init.pop("decoder"))
        model = MaskFormer(decoder=decoder, **{key: instantiate(value) for key, value in init.items()}).eval()
        if perturb:
            perturb_affine_params(model, seed)
        return model
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
