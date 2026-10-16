"""S2 variable bookkeeping: which native variables train, and where they live.

The JIT gate checks that every TRAINABLE variable gets a gradient, so a weight wrongly built as non-trainable
escapes it. These tests pin trainability against two oracles, per golden case and mode:
- the torch reference: native trainable parameters (non-leaf variables + leaf kernels/biases) have exactly the
  size of the torch parameters that receive a gradient. The masked-Linformer rows (R5) add the
  sequence projections: E/F rows ``[:n]`` are leaf kernels, the tail rows past n a native-owned weight.
- the hybrid: every HGQ2 leaf variable has the hybrid's ``trainable`` flag (recorded in the golden meta).
"""

import re

import numpy as np
import pytest
import torch

from hepattn.keras_native.factory import LayerFactory

from .golden.cases import CASES, modes, torch_module
from .golden_utils import build_native, golden_meta, inputs_of, load_golden, native_leaves, native_module

PARAMETER_NAMES = {"kernel", "bias"}  # leaf variables that are torch parameters; the rest is quantizer state
# torch parameters without a gradient, per case (none: org's masked Linformer uses its qkv norms, value residual and
# every projection row, the tail through the renorm numerator)
DEAD_IN_TORCH: dict[str, re.Pattern] = {}
# path suffixes of native-owned weights that must train (torch nn.Parameter) in each case
MUST_TRAIN = {
    "attn": ["attn_q_norm/gamma", "attn_k_norm/beta", "attn_v_norm/gamma"],
    "enc": ["encoder/register_tokens"],
}
MUST_TRAIN_CASE = {
    "enc_lin": ["_attn_project_E_proj/kernel", "_attn_project_F_proj/kernel"],  # FastLayerNorm: no norm weights
    "lin_kv_ca": ["attn_attend_E/tail", "attn_attend_F/tail"],
    "enc_torch": ["_attn_norm/gamma", "_ffn_norm/beta", "_attn_ls/gamma", "_ffn_ls/gamma", "_attn_q_norm/gamma"],
}


def torch_used_parameters(case: str) -> dict[str, int]:
    """Numel of each torch reference parameter that gets a nonzero gradient from mean(out^2) on the golden inputs."""
    module = torch_module(case)
    inputs = {key: torch.from_numpy(value) for key, value in inputs_of(load_golden(case)).items()}
    kind = CASES[case]["kind"]
    if kind == "dense":
        out = module(inputs["x"])
    elif kind == "innet":
        out = module(inputs)
    elif kind == "enc":
        out = module(inputs["x"], x_sort_value=inputs.get("x_sort_value"), kv_mask=inputs["kv_mask"])
    else:
        out = module(inputs["q"], **{key: inputs[key] for key in ("k", "kv_mask", "attn_mask", "kv_sort_idx") if key in inputs})
    out.square().mean().backward()
    used = {name: p.numel() for name, p in module.named_parameters() if p.grad is not None and bool(p.grad.any())}
    dead = sorted(set(dict(module.named_parameters())) - set(used))
    pattern = DEAD_IN_TORCH.get(case)
    assert all(pattern and pattern.fullmatch(name) for name in dead), f"{case}: unexpected torch parameters without gradient {dead}"
    return used


def split_variables(module):
    leaf_ids = {id(var) for leaf in native_leaves(module).values() for var in leaf.variables}
    non_leaf = [var for var in module.variables if id(var) not in leaf_ids]
    return leaf_ids, non_leaf


@pytest.mark.parametrize(
    ("case", "mode"),
    [(case, mode) for case in CASES for mode in modes(case)],
)
def test_trainable_variables_match_torch_and_hybrid(case, mode):
    golden = load_golden(case)
    module, _ = build_native(case, mode, golden)
    leaves = native_leaves(module)
    _, non_leaf = split_variables(module)

    # hybrid oracle: every leaf variable's trainable flag (kernel/bias, HGQ2 bitwidths, quantizer state)
    hybrid_flags = golden_meta(golden)[mode]["trainable"]
    for name, leaf in leaves.items():
        native_flags = [bool(var.trainable) for var in leaf.variables]
        assert native_flags == hybrid_flags[name], f"{case}/{mode}/{name}: trainable flags differ from the hybrid"

    # native-owned weights: all train except the Fourier posenc buffer B (a torch buffer)
    frozen = [var.path for var in non_leaf if not var.trainable]
    assert all(path.endswith("_posenc/B") for path in frozen), f"{case}/{mode}: non-trainable native weights {frozen}"
    assert (CASES[case]["kind"] == "innet") == bool(frozen), f"{case}/{mode}: posenc B must exist and be non-trainable only in innet"

    # torch oracle: the trainable parameter count equals the torch parameters that get a gradient
    native_size = sum(int(np.prod(var.shape)) for var in non_leaf if var.trainable)
    native_size += sum(int(np.prod(var.shape)) for leaf in leaves.values() for var in leaf.variables if var.name in PARAMETER_NAMES and var.trainable)
    torch_size = sum(torch_used_parameters(case).values())
    assert native_size == torch_size, f"{case}/{mode}: {native_size} native trainable parameters vs {torch_size} used in torch"

    trainable_paths = [var.path for var in module.trainable_variables]
    for suffix in MUST_TRAIN.get(CASES[case]["kind"], []) + MUST_TRAIN_CASE.get(case, []):
        assert any(path.endswith(suffix) for path in trainable_paths), f"{case}/{mode}: no trainable variable ending in {suffix}"


def test_variable_paths_are_deterministic():
    """The path layout the S7 converter will rely on (enc_torch, float), identical when built by build() or a call.

    Attention / FFN containers sit directly under the encoder layer; norms and LayerScale under the Residual
    (``*_res``) that owns them; register tokens on the encoder.
    """
    golden = load_golden("enc_torch")
    called, _ = build_native("enc_torch", "float", golden)
    paths = [var.path for var in called.variables]
    assert len(set(paths)) == len(paths), "variable paths are not unique"
    expected = [
        "encoder/register_tokens",
        "encoder/encoder_l0/encoder_l0_attn/encoder_l0_attn_q_proj/kernel",
        "encoder/encoder_l0/encoder_l0_attn/encoder_l0_attn_q_norm/gamma",
        "encoder/encoder_l0/encoder_l0_attn_res/encoder_l0_attn_norm/gamma",
        "encoder/encoder_l0/encoder_l0_attn_res/encoder_l0_attn_ls/gamma",
        "encoder/encoder_l0/encoder_l0_ffn/encoder_l0_ffn_hidden0/kernel",
        "encoder/encoder_l0/encoder_l0_ffn_res/encoder_l0_ffn_norm/beta",
        "encoder/encoder_l1/encoder_l1_attn/encoder_l1_attn_value_residual_mix/kernel",
    ]
    missing = [path for path in expected if path not in paths]
    assert not missing, f"missing paths {missing}"

    built = native_module("enc_torch", LayerFactory(None))
    built.build((None, *inputs_of(golden)["x"].shape[1:]))
    assert [var.path for var in built.variables] == paths, "build() and the first call give different variable paths"
