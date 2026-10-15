"""S2 golden cases: torch reference modules and inputs shared by ``make_golden.py`` (hybrid, torch backend)
and ``tests/keras_native/test_quantized_golden.py`` (native, JAX backend).

Imports torch and ``hepattn.models`` only (no keras), so both environments can import it. Configs are the
dim-32 CLIC test configs of ``tests/keras`` (16 heads, 160 nodes + 8 registers, 150 queries, Linformer
k = seq_len = 256). Leaf names are explicit so hybrid and native leaves pair up by name.
"""

import copy

import numpy as np
import torch
from torch import nn

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder
from hepattn.models.input import InputNet
from hepattn.models.linformer import LinformerAttention
from hepattn.models.posenc import FourierPositionEncoder

DIM, HEADS, NODES, REGS, QUERIES, LIN = 32, 16, 160, 8, 150, 256

# The paired-delta configs of tests/keras/test_quantized.py (HIGH_QUANT / LOW_QUANT).
QUANT_SPECS = {
    "float": None,
    "high": {
        "weight": {"default_q_type": "kbi", "b0": 20, "i0": 2},
        "datalane": {"default_q_type": "kif", "i0": 8, "f0": 16},
        "table": {"default_q_type": "kif", "i0": 2, "f0": 16},
    },
    "low": {
        "weight": {"default_q_type": "kbi", "b0": 4, "i0": 1},
        "datalane": {"default_q_type": "kif", "i0": 2, "f0": 2},
        "table": {"default_q_type": "kif", "i0": 1, "f0": 3},
    },
}

ENC_COMMON = {"num_layers": 2, "dim": DIM, "hybrid_norm": True, "value_residual": True, "num_register_tokens": REGS}

# kind, constructor kwargs (shared by torch / hybrid / native unless noted), leaf-name prefix
CASES: dict[str, dict] = {
    "dense_ffn": {"kind": "dense", "cfg": {"input_size": DIM}, "name": "d"},
    "dense_innet": {"kind": "dense", "cfg": {"input_size": 27, "output_size": DIM}, "name": "d"},
    "attn_self": {"kind": "attn", "cfg": {"dim": DIM, "num_heads": HEADS, "qkv_norm": True, "norm": "LayerNorm"}, "name": "attn"},
    "attn_cross": {"kind": "attn", "cfg": {"dim": DIM, "num_heads": HEADS, "qkv_norm": True, "norm": "LayerNorm"}, "name": "attn"},
    "lin_self": {"kind": "lin", "cfg": {"dim": DIM, "num_heads": HEADS, "seq_len": LIN, "k": LIN}, "name": "lin"},
    "lin_cross": {"kind": "lin", "cfg": {"dim": DIM, "num_heads": HEADS, "seq_len": LIN, "k": LIN}, "name": "lin"},
    "enc_lin": {
        "kind": "enc",
        "cfg": {**ENC_COMMON, "attn_type": "linformer", "attn_kwargs": {"num_heads": HEADS, "linformer_seq_len": LIN, "linformer_proj_dim": LIN}},
        "name": "encoder",
    },
    "enc_torch": {
        "kind": "enc",
        "cfg": {**ENC_COMMON, "attn_type": "torch", "layer_scale": 0.5, "attn_kwargs": {"num_heads": HEADS}},
        "name": "encoder",
    },
    "innet": {
        "kind": "innet",
        "cfg": {"input_size": 27, "output_size": DIM, "posenc": {"dim": DIM, "fields": ["eta", "phi"], "scale": 0.1}},
        "name": "innet0",
    },
}


def golden_file(case: str) -> str:
    return f"{case}.npz"


def make_inputs(case: str, seed: int = 123) -> dict[str, np.ndarray]:
    """Deterministic float32 / bool inputs for a case (the golden file stores them too)."""
    rng = np.random.default_rng(seed)
    kind = CASES[case]["kind"]
    lengths = np.array([NODES, 117])
    node_valid = np.arange(NODES)[None, :] < lengths[:, None]
    if kind == "dense":
        return {"x": rng.standard_normal((2, NODES, CASES[case]["cfg"]["input_size"])).astype(np.float32)}
    if kind == "innet":
        return {
            "node_features": rng.standard_normal((2, NODES, 27)).astype(np.float32),
            "node_eta": rng.uniform(-3, 3, (2, NODES)).astype(np.float32),
            "node_phi": rng.uniform(-np.pi, np.pi, (2, NODES)).astype(np.float32),
        }
    if kind == "enc":
        return {"x": rng.standard_normal((2, NODES, DIM)).astype(np.float32), "kv_mask": node_valid}
    if case.endswith("_self"):
        seq = NODES + REGS
        valid = np.arange(seq)[None, :] < np.array([seq, 125])[:, None]
        out = {"q": rng.standard_normal((2, seq, DIM)).astype(np.float32)}
        if kind == "attn":
            out["kv_mask"] = valid
        return out
    attn_mask = rng.random((2, QUERIES, NODES)) > 0.3
    attn_mask[..., 0] = True
    attn_mask[:, 1, :] = False  # one fully masked query (zero attention weights in every variant)
    out = {"q": rng.standard_normal((2, QUERIES, DIM)).astype(np.float32), "k": rng.standard_normal((2, NODES, DIM)).astype(np.float32)}
    out["attn_mask"] = attn_mask
    if kind == "attn":
        out["kv_mask"] = node_valid
    return out


@torch.no_grad()
def perturb_affine_params(module: nn.Module, seed: int) -> None:
    """Non-default biases / norm affines / LayerScale (same rule as tests/keras_native/porting.py)."""
    gen = torch.Generator().manual_seed(seed)
    for name, param in module.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"bias", "in_proj_bias", "gamma"} or (leaf == "weight" and param.dim() == 1):
            param.add_(0.1 * torch.randn(param.shape, generator=gen))


def torch_module(case: str, seed: int = 7) -> nn.Module:
    """The pure-torch reference module of a case (seeded, affines perturbed, eval mode).

    Raises:
        ValueError: For an unknown case kind.
    """
    spec = CASES[case]
    cfg = copy.deepcopy(spec["cfg"])
    torch.manual_seed(seed)
    kind = spec["kind"]
    if kind == "dense":
        module = Dense(**cfg)
    elif kind == "attn":
        module = Attention(attn_type="torch", **cfg)
    elif kind == "lin":
        module = LinformerAttention(cfg["dim"], seq_len=cfg["seq_len"], k=cfg["k"], heads=cfg["num_heads"], dim_head=cfg["dim"] // cfg["num_heads"])
    elif kind == "enc":
        module = Encoder(**cfg)
    elif kind == "innet":
        posenc = cfg.pop("posenc")
        module = InputNet(input_name="node", fields=["features"], net=Dense(**cfg), posenc=FourierPositionEncoder(input_name="node", **posenc))
    else:
        raise ValueError(kind)
    perturb_affine_params(module.eval(), seed)
    return module


def torch_state(module: nn.Module) -> dict[str, np.ndarray]:
    """Parameters and buffers by state_dict key, as numpy."""
    return {key: value.detach().cpu().numpy() for key, value in module.state_dict().items()}
