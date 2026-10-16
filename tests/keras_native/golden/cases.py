"""Golden cases: torch reference modules and inputs shared by ``make_golden.py`` (hybrid, torch backend)
and ``tests/keras_native/test_quantized_golden.py`` (native, JAX backend).

Imports torch and ``hepattn.models`` only (no keras), so both environments can import it. Configs are the
dim-32 CLIC test configs of ``tests/keras`` (16 heads, 160 nodes + 8 registers, 150 queries). Leaf names are
explicit so hybrid and native leaves pair up by name.

The masked-Linformer cases (org/main ``ac0da8f``, PR #9; k = 64) cover the three org call paths
(docs/hgq/RETARGET_ORG_MAIN.md section 5.6): ``lin_enc`` is the encoder's ``project`` with a key mask (renorm),
``lin_q_ca`` the decoder query cross-attention's ``attend`` with ``kv_sort_idx``, and ``lin_kv_ca`` the decoder
constituent cross-attention's ``attend`` with tail rows (150 keys, ``linformer_seq_len`` 160). ``enc_lin`` is the
PR #8 encoder of ``base.yaml`` + ``linformer.yaml`` (FastLayerNorm, legacy placement, SwiGLU FFN, phi sorting).
The cases ``dense_*``, ``attn_*``, ``innet`` and ``enc_torch`` are kept from S2 unchanged (their golden files
must regenerate bit-identically).
"""

import copy

import numpy as np
import torch
from torch import nn

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder, sort_index
from hepattn.models.input import InputNet
from hepattn.models.posenc import FourierPositionEncoder

DIM, HEADS, NODES, REGS, QUERIES = 32, 16, 160, 8, 150
LIN_K = 64  # linformer.yaml linformer_k

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
BASE_MODES = ("float", "high", "low")
EXTRA_MODES: dict[str, tuple[str, ...]] = {}  # per-case extra modes (none since Helen's Linformer cases were removed)

# org Attention(attn_type="linformer") with q/k/v norms (affine, so a missing port shows) and no value residual
LIN_ATTN = {"dim": DIM, "num_heads": HEADS, "qkv_norm": True, "norm": "LayerNorm", "attn_type": "linformer", "linformer_k": LIN_K}

ENC_COMMON = {"num_layers": 2, "dim": DIM, "hybrid_norm": True, "value_residual": True, "num_register_tokens": REGS}

# kind, constructor kwargs (shared by torch / hybrid / native), leaf-name prefix
CASES: dict[str, dict] = {
    "dense_ffn": {"kind": "dense", "cfg": {"input_size": DIM}, "name": "d"},
    "dense_innet": {"kind": "dense", "cfg": {"input_size": 27, "output_size": DIM}, "name": "d"},
    "attn_self": {"kind": "attn", "cfg": {"dim": DIM, "num_heads": HEADS, "qkv_norm": True, "norm": "LayerNorm"}, "name": "attn"},
    "attn_cross": {"kind": "attn", "cfg": {"dim": DIM, "num_heads": HEADS, "qkv_norm": True, "norm": "LayerNorm"}, "name": "attn"},
    "dense_ffn_swiglu": {"kind": "dense", "cfg": {"input_size": DIM, "activation": "SwiGLU"}, "name": "d"},
    "dense_innet_swiglu": {"kind": "dense", "cfg": {"input_size": 27, "output_size": DIM, "activation": "SwiGLU"}, "name": "d"},
    "lin_enc": {"kind": "attn", "cfg": {**LIN_ATTN, "linformer_seq_len": NODES + REGS}, "name": "attn"},
    "lin_q_ca": {"kind": "attn", "cfg": {**LIN_ATTN, "linformer_seq_len": NODES}, "name": "attn"},
    "lin_kv_ca": {"kind": "attn", "cfg": {**LIN_ATTN, "linformer_seq_len": NODES}, "name": "attn"},
    "enc_lin": {
        "kind": "enc",
        "cfg": {
            **ENC_COMMON,
            "attn_type": "linformer",
            "norm": "FastLayerNorm",
            "dense_norm_placement": "legacy",
            "attn_kwargs": {"num_heads": HEADS, "linformer_seq_len": NODES + REGS, "linformer_k": LIN_K},
            "dense_kwargs": {"activation": "SwiGLU"},
        },
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


def modes(case: str) -> tuple[str, ...]:
    """The quantization modes stored in a case's golden file."""
    return BASE_MODES + EXTRA_MODES.get(case, ())


def golden_file(case: str) -> str:
    return f"{case}.npz"


def make_inputs(case: str, seed: int = 123) -> dict[str, np.ndarray]:
    """Deterministic float32 / bool inputs for a case (the golden file stores them too)."""
    rng = np.random.default_rng(seed)
    kind = CASES[case]["kind"]
    if case in LIN_INPUTS:
        return LIN_INPUTS[case](rng)
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


def _padded(lengths, size: int) -> np.ndarray:
    return np.arange(size)[None, :] < np.asarray(lengths)[:, None]


def _cross_mask(rng, queries: int, keys: int) -> np.ndarray:
    attn_mask = rng.random((2, queries, keys)) > 0.3
    attn_mask[..., 0] = True
    attn_mask[:, 1, :] = False  # one fully masked query: zero output in org attend
    return attn_mask


def _lin_enc_inputs(rng) -> dict[str, np.ndarray]:
    """Encoder self-attention (registers + nodes): org ``project`` with ``kv_mask``, both events padded (renorm)."""
    seq = NODES + REGS
    return {"q": rng.standard_normal((2, seq, DIM)).astype(np.float32), "kv_mask": _padded([150, 125], seq)}


def _lin_q_ca_inputs(rng) -> dict[str, np.ndarray]:
    """Decoder query cross-attention: org ``attend`` with ``kv_mask`` and ``kv_sort_idx`` (phi order, padding last)."""
    kv_mask = _padded([140, 117], NODES)
    phi = rng.uniform(-np.pi, np.pi, (2, NODES)).astype(np.float32)
    return {
        "q": rng.standard_normal((2, QUERIES, DIM)).astype(np.float32),
        "k": rng.standard_normal((2, NODES, DIM)).astype(np.float32),
        "attn_mask": _cross_mask(rng, QUERIES, NODES),
        "kv_mask": kv_mask,
        "kv_sort_idx": sort_index(torch.from_numpy(phi), torch.from_numpy(kv_mask)).numpy(),
    }


def _lin_kv_ca_inputs(rng) -> dict[str, np.ndarray]:
    """Decoder constituent cross-attention: org ``attend`` over the 150 queries, so E/F rows 150-159 are the tail."""
    return {
        "q": rng.standard_normal((2, NODES, DIM)).astype(np.float32),
        "k": rng.standard_normal((2, QUERIES, DIM)).astype(np.float32),
        "attn_mask": _cross_mask(rng, NODES, QUERIES),
    }


def _enc_lin_inputs(rng) -> dict[str, np.ndarray]:
    """PR #8 encoder with phi sorting; both events padded (plan 1.2: every event has a padded node)."""
    return {
        "x": rng.standard_normal((2, NODES, DIM)).astype(np.float32),
        "kv_mask": _padded([150, 117], NODES),
        "x_sort_value": rng.uniform(-np.pi, np.pi, (2, NODES)).astype(np.float32),
    }


LIN_INPUTS = {"lin_enc": _lin_enc_inputs, "lin_q_ca": _lin_q_ca_inputs, "lin_kv_ca": _lin_kv_ca_inputs, "enc_lin": _enc_lin_inputs}


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
        module = Attention(**{"attn_type": "torch", **cfg})
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
