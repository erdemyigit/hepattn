"""Net-swap: replace hepattn Dense sub-nets of task/input-net modules with Keras twins.

Task modules (loss/cost/predict/attn_mask logic) and InputNet (concat + posenc glue)
are reused as-is; only their parameterized Dense nets are swapped for factory-built
KerasDense twins. The swap ports the current weights, so a freshly initialized module
keeps its init distribution and a checkpoint-loaded module keeps its trained weights.
"""

from torch import nn

from hepattn.keras.dense import KerasDense
from hepattn.keras.factory import LayerFactory
from hepattn.keras.porting import port_dense
from hepattn.models.dense import Dense


def kerasify_module(module: nn.Module, factory: LayerFactory, name: str = "net") -> nn.Module:
    """Recursively replace every hepattn Dense child with a weight-ported KerasDense twin (in place).

    ``name`` seeds deterministic keras layer names from the attribute path (e.g.
    task_mask_object_net_hidden0) so state_dict keys are stable across processes.
    """
    for child_name, child in list(module.named_children()):
        path = f"{name}_{child_name}"
        if isinstance(child, Dense):
            kdense = KerasDense.from_torch(child, factory, name=path)
            port_dense(child, kdense)
            setattr(module, child_name, kdense)
        else:
            kerasify_module(child, factory, name=path)
    return module
