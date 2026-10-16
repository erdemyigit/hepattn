"""Hybrid weight-decay groups of the org test model (QUANT_SPECS['high']): the oracle for test_optim's exclusion test.

    KERAS_BACKEND=torch PYTHONPATH=<worktree>/src <hepattn>/.venv/bin/python docs/hgq/port_log/S5_hybrid_param_groups.py

Prints, for ``KerasMaskFormer.trainable_parameter_groups()`` (the groups ``MPflowHGQ.configure_optimizers`` decays /
does not decay), the tensor count, the element total and the count per last path component (kernel, bias, gamma, ...).
"""

import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
import torch

from hepattn.keras import set_keras_default_device
from hepattn.keras.maskformer import KerasMaskFormer

GOLDEN = Path(__file__).resolve().parents[3] / "tests/keras_native/golden"
sys.path.insert(0, str(GOLDEN))
from cases import QUANT_SPECS  # noqa: E402
from mf_cases import clic_dummy_batch, hybrid_org_kwargs  # noqa: E402


def main() -> None:
    set_keras_default_device("cpu")
    torch.manual_seed(30)
    model = KerasMaskFormer(**hybrid_org_kwargs(), quant=QUANT_SPECS["high"]).eval()
    inputs, _ = clic_dummy_batch(num_events=2)
    with torch.no_grad():
        model({k: torch.from_numpy(v) for k, v in inputs.items()})
    decay, quant = model.trainable_parameter_groups()
    paths = {id(v.value): v.path for layer in model.keras_layers() for v in layer.weights}
    torch_names = {id(p): n for n, p in model.named_parameters()}

    def describe(group):
        kinds = collections.Counter((paths.get(id(p)) or torch_names[id(p)]).rsplit("/", 1)[-1] for p in group)
        return {
            "count": len(group),
            "elements": int(sum(p.numel() for p in group)),
            "torch_params": sum(id(p) in torch_names for p in group),
            "kinds": dict(sorted(kinds.items())),
        }

    print(json.dumps({"decay": describe(decay), "quantizer": describe(quant)}, indent=1))


if __name__ == "__main__":
    main()
