"""Write the layer golden files from the HYBRID (keras on torch) layers; run it in the hybrid torch env.

    KERAS_BACKEND=torch PYTHONPATH=<worktree>/src <hepattn>/.venv/bin/python \
        tests/keras_native/golden/make_golden.py [case ...]

For every case in ``cases.CASES`` (or the named ones) it builds the pure-torch reference module (seeded) and, for
each mode in ``cases.modes(case)``, ports it into the hybrid twin with ``hepattn.keras.porting`` and runs it in eval
mode. ``GOLDEN_OUT=<dir>`` writes the files there instead of next to this script (for the regeneration check of the
kept cases: their arrays must be bit-identical, docs/hgq/port_log/R5.md). One compressed ``<case>.npz`` per case
holds: the inputs (``in::<name>``), the torch reference state (``src::<state_dict key>``, shared by the modes), and
per mode the output (``<mode>::out``) and every variable of every named keras leaf in tracking order
(``<mode>::kv::<leaf>::<idx>::<var name>``, leaves as ``golden_utils.select_leaves``), plus ``meta`` (json; per mode
the leaf names, the output abs max and each leaf variable's ``trainable`` flag).
"""

import copy
import json
import os
import sys
from pathlib import Path

if os.environ.get("KERAS_BACKEND", "torch") != "torch":
    raise SystemExit("make_golden.py needs KERAS_BACKEND=torch (the hybrid is the oracle)")
os.environ["KERAS_BACKEND"] = "torch"

import hgq
import keras
import numpy as np
import torch
from hgq.layers import QDense, QEinsum, QEinsumDense, QSoftmax, QUnaryFunctionLUT

import hepattn
from hepattn.keras import set_keras_default_device
from hepattn.keras.attention import KerasAttention
from hepattn.keras.dense import KerasDense
from hepattn.keras.encoder import KerasEncoder
from hepattn.keras.factory import EinsumOp, LayerFactory, SoftmaxOp
from hepattn.keras.porting import port_attention, port_dense, port_encoder
from hepattn.keras.tasks import kerasify_module
from hepattn.keras_native.masked_linformer import SeqProjection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES, QUANT_SPECS, golden_file, make_inputs, modes, torch_module, torch_state  # noqa: E402

LEAF_TYPES = (QDense, QEinsum, QSoftmax, QEinsumDense, QUnaryFunctionLUT, keras.layers.Dense, keras.layers.EinsumDense, EinsumOp, SoftmaxOp)


def hybrid_module(case: str, tmod, factory: LayerFactory):
    spec = CASES[case]
    cfg = copy.deepcopy(spec["cfg"])
    kind = spec["kind"]
    if kind == "dense":
        return KerasDense(**cfg, factory=factory, name=spec["name"])
    if kind == "attn":
        return KerasAttention(**cfg, factory=factory, name=spec["name"])
    if kind == "enc":
        return KerasEncoder(factory=factory, **cfg)
    if kind == "innet":
        return kerasify_module(copy.deepcopy(tmod), factory, name=spec["name"])  # Dense swapped, posenc kept
    raise ValueError(kind)


@torch.no_grad()
def port(case: str, tmod, kmod) -> None:
    kind = CASES[case]["kind"]
    if kind == "dense":
        port_dense(tmod, kmod)
    elif kind == "attn":
        port_attention(tmod, kmod)
    elif kind == "enc":
        port_encoder(tmod, kmod)
    elif kind != "innet":  # innet: kerasify_module ported (float) or ports at the first forward (quantized)
        raise ValueError(kind)


def run(case: str, module, inputs: dict[str, np.ndarray]) -> np.ndarray:
    kind = CASES[case]["kind"]
    tin = {k: torch.from_numpy(v) for k, v in inputs.items()}
    with torch.no_grad():
        if kind == "dense":
            out = module(tin["x"])
        elif kind == "innet":
            out = module(tin)
        elif kind == "enc":
            out = module(tin["x"], x_sort_value=tin.get("x_sort_value"), kv_mask=tin["kv_mask"])
        elif kind == "attn":
            out = module(tin["q"], k=tin.get("k"), kv_mask=tin.get("kv_mask"), attn_mask=tin.get("attn_mask"), kv_sort_idx=tin.get("kv_sort_idx"))
        else:
            raise ValueError(kind)
    return out.detach().cpu().numpy()


def select_leaves(layers) -> list:
    """The rule of ``tests/keras_native/golden_utils.select_leaves`` (kept in step by the native golden tests)."""
    layers = list(layers)
    found = [layer for layer in layers if isinstance(layer, LEAF_TYPES)]
    found += [layer.tail_q for layer in layers if isinstance(layer, SeqProjection) and layer.tail_q is not None]
    ids = [{id(var) for var in layer.variables} for layer in found]
    return [layer for i, layer in enumerate(found) if not (ids[i] and any(ids[i] < other for other in ids))]


def leaves(module) -> list:
    found = select_leaves(module.modules())
    names = [m.name for m in found]
    assert len(set(names)) == len(names), f"leaf names are not unique: {names}"
    return found


def main() -> None:
    set_keras_default_device("cpu")
    assert Path(hepattn.__file__).resolve().is_relative_to(HERE.parents[2] / "src"), hepattn.__file__
    out_dir = Path(os.environ.get("GOLDEN_OUT", HERE))
    selected = sys.argv[1:] or list(CASES)
    unknown = sorted(set(selected) - set(CASES))
    assert not unknown, f"unknown cases {unknown}"
    sizes = {}
    for case in selected:
        inputs = make_inputs(case)
        tmod = torch_module(case)
        arrays = {f"in::{k}": v for k, v in inputs.items()}
        arrays.update({f"src::{k}": v for k, v in torch_state(tmod).items()})
        meta = {"case": case, "versions": {"torch": torch.__version__, "keras": keras.__version__, "hgq2": getattr(hgq, "__version__", "?")}}
        for mode in modes(case):
            factory = LayerFactory(QUANT_SPECS[mode])
            keras.utils.clear_session(free_memory=False)
            set_keras_default_device("cpu")
            with factory.scopes():
                kmod = hybrid_module(case, tmod, factory)
                if hasattr(kmod, "eval"):
                    kmod.eval()
                run(case, kmod, inputs)  # builds lazy (quantized) leaves; innet ports its pending Dense here
            port(case, tmod, kmod)
            out = run(case, kmod, inputs)
            arrays[f"{mode}::out"] = out
            leaf_list = leaves(kmod)
            for leaf in leaf_list:
                for i, var in enumerate(leaf.variables):
                    arrays[f"{mode}::kv::{leaf.name}::{i:03d}::{var.name}"] = keras.ops.convert_to_numpy(var.value).astype(var.dtype)
            meta[mode] = {
                "leaves": [leaf.name for leaf in leaf_list],
                "out_absmax": float(np.abs(out).max()),
                "trainable": {leaf.name: [bool(var.trainable) for var in leaf.variables] for leaf in leaf_list},
            }
        arrays["meta"] = np.array(json.dumps(meta))
        path = out_dir / golden_file(case)
        np.savez_compressed(path, **arrays)
        sizes[case] = path.stat().st_size
        print(case, sizes[case], {m: meta[m] for m in modes(case)}, flush=True)
    print(f"total {sum(sizes.values()) / 1e6:.2f} MB in {len(sizes)} files")


if __name__ == "__main__":
    main()
