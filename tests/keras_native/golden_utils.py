"""Build keras_native twins of the S2 golden cases and load the hybrid golden weights into them.

Weights: the torch reference state (``src::*``) is loaded into a fresh torch module and ported with
``porting.py`` (norms, LayerScale, register tokens, Linformer projections, posenc B); then every variable of every
named HGQ2/keras leaf is overwritten from ``<mode>::kv::<leaf>::<idx>::<name>`` in tracking order, with name,
shape and dtype asserted. So the native layer holds exactly the hybrid's weights and quantizer state.
"""

import copy
import json
from pathlib import Path

import keras
import numpy as np
import torch
from hgq.layers import QDense, QEinsum, QSoftmax

from hepattn.keras_native.attention import Attention
from hepattn.keras_native.dense import Dense
from hepattn.keras_native.encoder import Encoder
from hepattn.keras_native.factory import EinsumOp, LayerFactory, SoftmaxOp
from hepattn.keras_native.input import InputNet
from hepattn.keras_native.linformer import LinformerAttention

from .golden.cases import CASES, QUANT_SPECS, golden_file, torch_module
from .porting import port_attention, port_dense, port_encoder, port_input_net, port_linformer

GOLDEN_DIR = Path(__file__).parent / "golden"
LEAF_TYPES = (QDense, QEinsum, QSoftmax, keras.layers.Dense, EinsumOp, SoftmaxOp)


def load_golden(case: str) -> dict[str, np.ndarray]:
    with np.load(GOLDEN_DIR / golden_file(case)) as data:
        return {key: data[key] for key in data.files}


def golden_meta(golden: dict) -> dict:
    return json.loads(str(golden["meta"]))


def inputs_of(golden: dict) -> dict[str, np.ndarray]:
    return {key.removeprefix("in::"): value for key, value in golden.items() if key.startswith("in::")}


def native_module(case: str, factory: LayerFactory):
    spec = CASES[case]
    cfg = copy.deepcopy(spec["cfg"])
    kind = spec["kind"]
    if kind == "dense":
        return Dense(**cfg, factory=factory, name=spec["name"])
    if kind == "attn":
        return Attention(**cfg, factory=factory, name=spec["name"])
    if kind == "lin":
        return LinformerAttention(cfg["dim"], num_heads=cfg["num_heads"], seq_len=cfg["seq_len"], k=cfg["k"], factory=factory, name=spec["name"])
    if kind == "enc":
        return Encoder(factory=factory, **cfg)
    if kind == "innet":
        posenc = cfg.pop("posenc")
        return InputNet("node", net=cfg, fields=["features"], posenc=posenc, factory=factory, name=spec["name"])
    raise ValueError(kind)


def run(case: str, module, inputs: dict, training=False):
    kind = CASES[case]["kind"]
    if kind == "dense":
        return module(inputs["x"], training=training)
    if kind == "innet":
        return module(inputs, training=training)
    if kind == "enc":
        return module(inputs["x"], kv_mask=inputs["kv_mask"], training=training)
    if kind == "attn":
        return module(inputs["q"], k=inputs.get("k"), kv_mask=inputs.get("kv_mask"), attn_mask=inputs.get("attn_mask"), training=training)
    k = inputs.get("k")
    return module(inputs["q"], k=k, v=k, attn_mask=inputs.get("attn_mask"), training=training)


def native_leaves(module) -> dict:
    found = [layer for layer in module._flatten_layers(include_self=True, recursive=True) if isinstance(layer, LEAF_TYPES)]  # noqa: SLF001
    names = [layer.name for layer in found]
    assert len(set(names)) == len(names), f"native leaf names are not unique: {names}"
    return dict(zip(names, found, strict=True))


def port_reference(case: str, module, golden: dict) -> None:
    """Load the torch reference state from the golden file and port it (float weights, norms, buffers)."""
    tmod = torch_module(case)
    state = {key.removeprefix("src::"): torch.from_numpy(value) for key, value in golden.items() if key.startswith("src::")}
    tmod.load_state_dict(state, strict=True)
    kind = CASES[case]["kind"]
    porter = {"dense": port_dense, "attn": port_attention, "lin": port_linformer, "enc": port_encoder, "innet": port_input_net}[kind]
    porter(tmod, module)


def assign_leaf_variables(module, golden: dict, mode: str) -> int:
    """Overwrite every named-leaf variable from the golden file; returns the number of variables assigned."""
    prefix = f"{mode}::kv::"
    by_leaf: dict[str, list[tuple[int, str, np.ndarray]]] = {}
    for key, value in golden.items():
        if key.startswith(prefix):
            leaf, idx, name = key.removeprefix(prefix).split("::")
            by_leaf.setdefault(leaf, []).append((int(idx), name, value))
    leaves = native_leaves(module)
    assert set(leaves) == set(golden_meta(golden)[mode]["leaves"]), f"leaf sets differ: {set(leaves) ^ set(golden_meta(golden)[mode]['leaves'])}"
    count = 0
    for leaf_name, layer in leaves.items():
        entries = sorted(by_leaf.get(leaf_name, []))
        variables = layer.variables
        assert len(variables) == len(entries), f"{leaf_name}: {len(variables)} native vs {len(entries)} hybrid variables"
        for var, (_, name, value) in zip(variables, entries, strict=True):
            assert var.name == name, f"{leaf_name}: variable order differs ({var.name} vs {name})"
            assert tuple(var.shape) == value.shape, f"{leaf_name}/{name}: shape {tuple(var.shape)} vs {value.shape}"
            assert np.dtype(var.dtype) == value.dtype, f"{leaf_name}/{name}: dtype {var.dtype} vs {value.dtype}"
            var.assign(value)
            count += 1
    return count


def build_from_golden(case: str, mode: str, golden: dict | None = None):
    """A native twin of ``case`` in ``mode`` holding the hybrid golden weights; returns (module, factory, golden)."""
    golden = load_golden(case) if golden is None else golden
    factory = LayerFactory(QUANT_SPECS[mode])
    with factory.scopes():
        module = native_module(case, factory)
        run(case, module, inputs_of(golden))  # eager build from the static shapes
    port_reference(case, module, golden)
    assign_leaf_variables(module, golden, mode)
    return module, factory, golden
