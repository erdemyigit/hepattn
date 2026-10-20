"""Which value does the hybrid log as ``val/loss``? S5.md said task + quant_losses.

Calls the real ``MPflowHGQ.aggregate_losses`` (``experiments/clic/lightning_module_hgq.py``) -> ``ModelWrapper.
aggregate_losses`` (``models/wrapper.py``) on a stub whose ``log`` records and whose ``quant_losses()`` returns 7.
Run in the hybrid env:  KERAS_BACKEND=torch PYTHONPATH=src ../hepattn/.venv/bin/python docs/hgq/port_log/S6b_val_loss_probe.py
"""

import torch
from torch import nn

from hepattn.experiments.clic.lightning_module_hgq import MPflowHGQ


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def quant_losses(self):
        return torch.tensor(7.0)


class Probe(MPflowHGQ):
    def __init__(self):  # skip the module's own __init__ (model building); keep nn.Module state
        nn.Module.__init__(self)
        self.model = StubModel()
        self.logged = {}

    def log(self, name, value, **_kwargs):
        self.logged[name] = float(value)


losses = {"final": {"classification": {"object_ce": torch.tensor(2.0)}, "mask": {"mask_bce": torch.tensor(3.0)}}}
probe = Probe()
returned = float(probe.aggregate_losses(losses, stage="val"))
print({"returned (backpropagated)": returned, **probe.logged})
assert probe.logged["val/loss"] == 5.0, "val/loss is the task loss"
assert probe.logged["val/quant_ebops_loss"] == 7.0
assert returned == 12.0
print("val/loss = task loss only; the quant term is logged separately and only added to the returned value")
