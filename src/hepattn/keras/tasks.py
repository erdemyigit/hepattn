"""Net-swap: replace hepattn Dense sub-nets of task/input-net modules with Keras twins.

Task modules (loss/cost/predict/attn_mask logic) and InputNet (concat + posenc glue)
are reused as-is; only their parameterized Dense nets are swapped for factory-built
KerasDense twins. The swap ports the current weights, so a freshly initialized module
keeps its init distribution and a checkpoint-loaded module keeps its trained weights.
"""

import torch
from torch import Tensor, nn

from hepattn.keras.dense import KerasDense
from hepattn.keras.factory import LayerFactory, apply_softmax
from hepattn.keras.porting import port_dense
from hepattn.models.dense import Dense
from hepattn.models.task import IncidenceRegressionTask, ObjectHitMaskTask


class KerasObjectHitMaskTask(ObjectHitMaskTask):
    """ObjectHitMaskTask whose mask-logit einsum is a quantized HGQ2 layer.

    Instances are produced by kerasify_module via a class swap on the already
    configured torch task (quantized mode only) — never constructed directly.
    """

    def attach_quant_layers(self, factory: LayerFactory, name: str) -> None:
        self.keras_mask_einsum = factory.einsum("bnc,bmc->bnm", name=f"{name}_mask_einsum")

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        mask_tokens = self.object_net(x[self.input_object + "_embed"])
        xs = x[self.input_constituent + "_embed"]
        if self.constituent_net:
            xs = self.constituent_net(xs)

        object_hit_logit = self.logit_scale * self.keras_mask_einsum([mask_tokens, xs], training=self.training)

        if (valid_mask := x[f"{self.input_constituent}_valid"]) is not None:
            valid_mask = valid_mask.unsqueeze(-2).expand_as(object_hit_logit)
            object_hit_logit = object_hit_logit.masked_fill(~valid_mask, torch.finfo(object_hit_logit.dtype).min)

        return {self.output_object_hit + "_logit": object_hit_logit}


class KerasIncidenceRegressionTask(IncidenceRegressionTask):
    """IncidenceRegressionTask with quantized incidence einsum and query-axis softmax.

    Instances are produced by kerasify_module via a class swap (quantized mode only).
    """

    def attach_quant_layers(self, factory: LayerFactory, name: str) -> None:
        self.keras_incidence_einsum = factory.einsum("bqe,ble->bql", name=f"{name}_incidence_einsum")
        self.keras_incidence_softmax = factory.softmax(axis=1, name=f"{name}_incidence_softmax")

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        x_object = self.net(x[self.input_object + "_embed"])
        x_hit = self.node_net(x[self.input_constituent + "_embed"])

        incidence_pred = self.keras_incidence_einsum([x_object, x_hit], training=self.training)
        incidence_pred = apply_softmax(self.keras_incidence_softmax, incidence_pred, training=self.training)
        incidence_pred = incidence_pred * x[self.input_constituent + "_valid"].unsqueeze(1).expand_as(incidence_pred)

        return {self.incidence_key: incidence_pred}


# torch task class -> quantized-forward subclass (applied by class swap in quantized mode)
QUANT_TASK_SWAPS: dict[type[nn.Module], type[nn.Module]] = {
    ObjectHitMaskTask: KerasObjectHitMaskTask,
    IncidenceRegressionTask: KerasIncidenceRegressionTask,
}


def kerasify_module(module: nn.Module, factory: LayerFactory, name: str = "net") -> nn.Module:
    """Recursively replace every hepattn Dense child with a weight-ported KerasDense twin (in place).

    In quantized mode, tasks whose forward contains einsum/softmax compute (mask
    logits, incidence) are additionally class-swapped to subclasses that route those
    ops through HGQ2 layers; the swap preserves all configured attributes and the
    inherited loss/cost/predict/attn_mask logic.

    ``name`` seeds deterministic keras layer names from the attribute path (e.g.
    task1_object_net_hidden0) so state_dict keys are stable across processes.
    """
    for child_name, child in list(module.named_children()):
        path = f"{name}_{child_name}"
        if isinstance(child, Dense):
            kdense = KerasDense.from_torch(child, factory, name=path)
            if factory.quantize:
                # quantized layers build lazily at the first forward; the weight port
                # happens there (KerasDense._materialize) from this retained source
                kdense._pending_port = child  # noqa: SLF001
            else:
                port_dense(child, kdense)
            setattr(module, child_name, kdense)
        else:
            kerasify_module(child, factory, name=path)

    if factory.quantize and type(module) in QUANT_TASK_SWAPS:
        module.__class__ = QUANT_TASK_SWAPS[type(module)]
        module.attach_quant_layers(factory, name)
    return module
