"""KerasMaskFormer: the hepattn MaskFormer with its network rebuilt from Keras/HGQ2 layers.

Orchestration (forward/loss/predict, per-layer matching, mask attention) is inherited
from MaskFormer. Input nets and tasks are the SAME torch modules as the YAML config
describes, with their Dense sub-nets swapped for factory-built Keras twins; the encoder
and decoder are keras mirrors constructed from plain config dicts so that every keras
layer is created inside the HGQ2 config scopes regardless of YAML instantiation order.

With quant=None this is the float parity-reference model; with a quant spec it is the
HGQ2 quantization-aware model on the identical graph (float weights warm-start QAT).
"""

import torch
from torch import Tensor, nn

from hepattn.keras import get_keras_default_device, keras, set_keras_default_device
from hepattn.keras.decoder import KerasMaskFormerDecoder
from hepattn.keras.encoder import KerasEncoder
from hepattn.keras.factory import LayerFactory
from hepattn.keras.tasks import kerasify_module
from hepattn.models.maskformer import MaskFormer


class KerasMaskFormer(MaskFormer):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: dict,
        decoder: dict,
        tasks: nn.ModuleList,
        dim: int,
        target_object: str = "particle",
        matcher: nn.Module | None = None,
        encoder_tasks: nn.ModuleList | None = None,
        quant: dict | None = None,
    ):
        """Build the keras-backed MaskFormer.

        Args:
            input_nets: Instantiated hepattn InputNet modules (their Dense nets are swapped in place).
            encoder: KerasEncoder configuration dict (same keys as the torch Encoder YAML section).
            decoder: KerasMaskFormerDecoder configuration dict (same keys as the torch decoder YAML section).
            tasks: Instantiated hepattn task modules (their Dense nets are swapped in place).
            dim: Embedding dimension.
            target_object: Target object name used during matching.
            matcher: Hungarian matcher module (reused unchanged, training-only).
            encoder_tasks: Optional tasks run on post-encoder features (Dense nets swapped in place).
            quant: None for the float reference model, or an HGQ2 QuantSpec dict
                (keys: weight, datalane, ebops) for the quantization-aware model.
        """
        factory = LayerFactory(quant)

        # Reset keras's process-global name counters so every layer name — including the
        # quantizer sub-layers HGQ2 creates internally, which cannot be named explicitly —
        # is determined by construction order alone. Keras embeds these names in the torch
        # state_dict keys; without the reset, a checkpoint written by the first model built
        # in a process cannot be loaded into the second (e.g. Lightning's `test` restore).
        device = get_keras_default_device()
        keras.utils.clear_session(free_memory=False)
        set_keras_default_device(device)

        with factory.scopes():
            keras_encoder = KerasEncoder(dim=dim, factory=factory, **encoder)
            keras_decoder = KerasMaskFormerDecoder(factory=factory, **decoder)
            for i, input_net in enumerate(input_nets):
                kerasify_module(input_net, factory, name=f"innet{i}")
            for i, task in enumerate(tasks):
                kerasify_module(task, factory, name=f"task{i}")
            for i, task in enumerate(encoder_tasks or []):
                kerasify_module(task, factory, name=f"enctask{i}")

        super().__init__(
            input_nets=input_nets,
            encoder=keras_encoder,
            decoder=keras_decoder,
            tasks=tasks,
            dim=dim,
            target_object=target_object,
            matcher=matcher,
            encoder_tasks=encoder_tasks,
        )
        self.factory = factory

    def keras_layers(self):
        """Yield the top-level keras layers of the model (without descending into their sublayers)."""

        def walk(module: nn.Module):
            for child in module.children():
                if isinstance(child, keras.layers.Layer):
                    yield child
                else:
                    yield from walk(child)

        yield from walk(self)

    def quant_losses(self) -> Tensor:
        """Sum of the HGQ2 EBOPs regularization terms collected from all keras layers.

        Keras layers register their EBOPs*beta losses via add_loss during training-mode
        calls; collecting from top-level layers only avoids double counting (Layer.losses
        already aggregates sublayers such as quantizers).
        """
        terms = [loss for layer in self.keras_layers() for loss in layer.losses]
        if not terms:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack([torch.as_tensor(t) for t in terms]).sum()
