"""Write the S2 golden files from the HYBRID (keras on torch) layers; run it in the hybrid torch env.

    KERAS_BACKEND=torch PYTHONPATH=<worktree>/src <hepattn>/.venv/bin/python tests/keras_native/golden/make_golden.py

For every case in ``cases.CASES`` it builds the pure-torch reference module (seeded) and, for each mode in
``cases.modes(case)`` (float/high/low, plus mid for the single Linformer layers), ports it into the hybrid twin
with ``hepattn.keras.porting`` and runs it in eval mode. One compressed ``<case>.npz`` per case holds: the inputs
(``in::<name>``), the torch reference state (``src::<state_dict key>``, shared by the modes), and per mode the
output (``<mode>::out``) and every variable of every named keras leaf in tracking order
(``<mode>::kv::<leaf>::<idx>::<var name>``), plus ``meta`` (json; per mode the leaf names, the output abs max and
each leaf variable's ``trainable`` flag).
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
from hgq.layers import QDense, QEinsum, QSoftmax

import hepattn
from hepattn.keras import set_keras_default_device
from hepattn.keras.attention import KerasAttention
from hepattn.keras.dense import KerasDense
from hepattn.keras.encoder import KerasEncoder
from hepattn.keras.factory import EinsumOp, LayerFactory, SoftmaxOp
from hepattn.keras.linformer import KerasLinformerAttention
from hepattn.keras.porting import port_attention, port_dense, port_encoder, port_linformer, port_residual
from hepattn.keras.tasks import kerasify_module

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES, QUANT_SPECS, golden_file, make_inputs, modes, torch_module, torch_state  # noqa: E402

LEAF_TYPES = (QDense, QEinsum, QSoftmax, keras.layers.Dense, EinsumOp, SoftmaxOp)


def hybrid_module(case: str, tmod, factory: LayerFactory):
    spec = CASES[case]
    cfg = copy.deepcopy(spec["cfg"])
    kind = spec["kind"]
    if kind == "dense":
        return KerasDense(**cfg, factory=factory, name=spec["name"])
    if kind == "attn":
        return KerasAttention(**cfg, factory=factory, name=spec["name"])
    if kind == "lin":
        return KerasLinformerAttention(cfg["dim"], num_heads=cfg["num_heads"], seq_len=cfg["seq_len"], k=cfg["k"], factory=factory, name=spec["name"])
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
    elif kind == "lin":
        port_linformer(tmod, kmod)
    elif kind == "enc" and CASES[case]["cfg"]["attn_type"] == "linformer":
        # hepattn.keras.porting.port_encoder assumes torch-backend attention; port the linformer leaves directly
        kmod.register_tokens.copy_(tmod.register_tokens)
        for tl, kl in zip(tmod.layers, kmod.layers, strict=True):
            port_residual(tl.attn, kl.attn, lambda t_attn, k_attn: port_linformer(t_attn.attn, k_attn))
            port_residual(tl.dense, kl.dense, port_dense)
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
            out = module(tin["x"], kv_mask=tin["kv_mask"])
        elif kind == "attn":
            out = module(tin["q"], k=tin.get("k"), kv_mask=tin.get("kv_mask"), attn_mask=tin.get("attn_mask"))
        else:
            k = tin.get("k")
            out = module(tin["q"], k=k, v=k, attn_mask=tin.get("attn_mask"))
    return out.detach().cpu().numpy()


def leaves(module) -> list:
    found = [m for m in module.modules() if isinstance(m, LEAF_TYPES)]
    names = [m.name for m in found]
    assert len(set(names)) == len(names), f"leaf names are not unique: {names}"
    return found


def main() -> None:
    set_keras_default_device("cpu")
    assert Path(hepattn.__file__).resolve().is_relative_to(HERE.parents[2] / "src"), hepattn.__file__
    sizes = {}
    for case in CASES:
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
        path = HERE / golden_file(case)
        np.savez_compressed(path, **arrays)
        sizes[case] = path.stat().st_size
        print(case, sizes[case], {m: meta[m] for m in modes(case)}, flush=True)
    print(f"total {sum(sizes.values()) / 1e6:.2f} MB in {len(sizes)} files")


if __name__ == "__main__":
    main()
