"""Write the MaskFormer golden files from the HYBRID ``KerasMaskFormer``; run it in the hybrid torch env.

    KERAS_BACKEND=torch PYTHONPATH=<worktree>/src <hepattn>/.venv/bin/python \
        tests/keras_native/golden/make_golden_maskformer.py [case ...]

For each case of ``CASES`` (``mf_torch``: the S3 test model with dense attention; ``mf_org``: org/main
``base.yaml`` + ``linformer.yaml`` at the test width, R5) it builds the seeded pure-torch MaskFormer, and for each mode
in ``MODES`` the hybrid twin with the same weights (``hepattn.keras.porting.port_maskformer``). ``GOLDEN_OUT=<dir>``
writes the files there instead of next to this script. One ``<case>.npz`` holds the CLIC dummy
batch (``in::<key>``), per mode every output leaf (``<mode>::out::<path>``, incl. the boolean ``attn_mask`` per layer)
and every non-kernel/bias variable of every named keras leaf (``make_golden.select_leaves``;
``<mode>::kv::<leaf>::<idx>::<var name>``), plus ``meta``
(json: per mode the leaf names, trainable flags and the sha256 of every kernel/bias; the sha256 of the torch state).
Quantized modes also store ``<mode>::qlosses``: the training-mode HGQ2 loss terms, each keras layer once, and in meta the
count and sum of the hybrid's ``quant_losses()``, which counts the task-head layers twice.

Size: the weights themselves are not stored. Kernels and biases are the ported torch weights in every mode, and the
torch model is rebuilt from its seed by the native test, which asserts the stored digests (torch state and every
leaf kernel/bias after porting). This keeps the two files near 4 MB instead of 24 MB.
"""

import json
import os
import sys
from pathlib import Path

if os.environ.get("KERAS_BACKEND", "torch") != "torch":
    raise SystemExit("make_golden_maskformer.py needs KERAS_BACKEND=torch (the hybrid is the oracle)")
os.environ["KERAS_BACKEND"] = "torch"

import hgq
import keras
import numpy as np
import torch

import hepattn
from hepattn.keras import set_keras_default_device
from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.keras.porting import port_maskformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import QUANT_SPECS  # noqa: E402
from make_golden import select_leaves  # noqa: E402
from mf_cases import (  # noqa: E402
    DIM,
    ORG_VARIANTS,
    VARIANTS,
    array_digest,
    clic_dummy_batch,
    decoder_config,
    flatten,
    hybrid_org_kwargs,
    task_specs,
    torch_input_nets,
    torch_model,
    torch_state_digest,
    torch_task,
)

DIGESTED = {"kernel", "bias"}  # stored as sha256 only (they equal the ported torch weights)
SEED = 30
# pflow_hgq.yaml / polaris 03_make_config.py quantizer config (production)
PROD = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0e-12},
}
# prod with SAT_SYM datalane overflow: same bitwidths, but no WRAP (HGQ2's datalane default). Under WRAP an ulp-level
# backend difference at a rounding or wrap edge becomes a jump of up to 2**(i+1), so prod rows are chaotic across
# backends while prodsat rows are not (measured, docs/hgq/port_log/S3.md).
PROD_SAT = {**PROD, "datalane": {**PROD["datalane"], "overflow_mode": "SAT_SYM"}}
MODES = {"float": None, "high": QUANT_SPECS["high"], "low": QUANT_SPECS["low"], "prod": PROD, "prodsat": PROD_SAT}


CASES = {"mf_torch": "torch", "mf_org": "org"}


def hybrid_model(variant: str, quant) -> KerasMaskFormer:
    torch.manual_seed(SEED)
    if variant in ORG_VARIANTS:
        return KerasMaskFormer(**hybrid_org_kwargs(), quant=quant).eval()
    enc = dict(VARIANTS[variant]["encoder"])
    return KerasMaskFormer(
        input_nets=torch_input_nets(),
        encoder=enc,
        decoder=decoder_config(variant),
        tasks=torch.nn.ModuleList([torch_task(s) for s in task_specs()]),
        dim=DIM,
        quant=quant,
    ).eval()


def leaves(module) -> list:
    found = select_leaves(module.modules())
    names = [m.name for m in found]
    assert len(set(names)) == len(names), f"leaf names are not unique: {names}"
    return found


def main() -> None:
    set_keras_default_device("cpu")
    assert Path(hepattn.__file__).resolve().is_relative_to(HERE.parents[2] / "src"), hepattn.__file__
    inputs, _ = clic_dummy_batch()
    tin = {k: torch.from_numpy(v) for k, v in inputs.items()}
    out_dir = Path(os.environ.get("GOLDEN_OUT", HERE))
    selected = sys.argv[1:] or list(CASES)
    assert set(selected) <= set(CASES), f"unknown cases {sorted(set(selected) - set(CASES))}"
    sizes = {}
    for case in selected:
        variant = CASES[case]
        tmodel = torch_model(variant, seed=SEED)
        arrays = {f"in::{k}": v for k, v in inputs.items()}
        meta = {
            "case": case,
            "seed": SEED,
            "torch_state_sha256": torch_state_digest(tmodel),
            "versions": {"torch": torch.__version__, "keras": keras.__version__, "hgq2": getattr(hgq, "__version__", "?")},
        }
        for mode, quant in MODES.items():
            kmodel = hybrid_model(variant, quant)
            with torch.no_grad():
                kmodel(tin)  # builds the lazy (quantized) leaves
            port_maskformer(tmodel, kmodel)
            with torch.no_grad():
                out = flatten(kmodel.eval()(tin))
            for path, value in out.items():
                arrays[f"{mode}::out::{path}"] = value.detach().cpu().numpy()
            leaf_list = leaves(kmodel)
            digests = {}
            for leaf in leaf_list:
                for i, var in enumerate(leaf.variables):
                    value = keras.ops.convert_to_numpy(var.value).astype(var.dtype)
                    key = f"{leaf.name}::{i:03d}::{var.name}"
                    if var.name in DIGESTED:
                        digests[key] = array_digest(value)
                    else:
                        arrays[f"{mode}::kv::{key}"] = value
            meta[mode] = {
                "quant": quant,
                "leaves": [leaf.name for leaf in leaf_list],
                "variables": {leaf.name: [var.name for var in leaf.variables] for leaf in leaf_list},
                "trainable": {leaf.name: [bool(var.trainable) for var in leaf.variables] for leaf in leaf_list},
                "digests": digests,
                "quant_losses": None,
            }
            if quant is not None:
                # training-mode HGQ2 loss terms (after the eval rows above: a training call also moves the WRAP
                # quantizers' integer bits). quant_losses() walks keras_layers(), which yields every task-head layer
                # twice (the tasks are registered under model.tasks AND model.decoder.tasks), so the per-layer terms
                # are stored deduplicated, and the hybrid's own (double-counting) sum and count go to meta.
                with torch.no_grad():
                    kmodel.train()(tin)
                    raw = list(kmodel.keras_layers())
                    unique = list({id(layer): layer for layer in raw}.values())
                    arrays[f"{mode}::qlosses"] = np.array([float(t) for layer in unique for t in layer.losses], dtype=np.float64)
                    meta_q = {
                        "n_layers_raw": len(raw),
                        "n_layers_unique": len(unique),
                        "n_terms_raw": sum(len(layer.losses) for layer in raw),
                        "sum_raw": float(kmodel.quant_losses()),
                    }
                meta[mode]["quant_losses"] = meta_q
            print(variant, mode, len(leaf_list), "leaves", flush=True)
        arrays["meta"] = np.array(json.dumps(meta))
        path = out_dir / f"{case}.npz"
        np.savez_compressed(path, **arrays)
        sizes[path.name] = path.stat().st_size
        print(path.name, sizes[path.name], flush=True)
    print(f"total {sum(sizes.values()) / 1e6:.2f} MB in {len(sizes)} files")


if __name__ == "__main__":
    main()
