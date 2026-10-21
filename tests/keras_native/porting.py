"""Weight porting for the native tests: pure-torch hepattn modules -> their hepattn.keras_native twins (imports torch).

The porters are the S7 converter's (``hepattn.experiments.clic.convert_to_native``), so every parity test also tests
the converter's mapping; kept outside ``src/hepattn/keras_native`` (that package must not import torch). Conventions:
- keras Dense kernel = torch Linear weight transposed.
- torch Attention packs Q/K/V in ``in_proj_weight`` (3*dim, dim), split in three by rows in q, k, v order.
- torch LayerNorm weight/bias -> keras LayerNormalization gamma/beta.
Every native layer must be built (``layer.build`` or one eager call) before porting.
"""

import torch
from torch import nn

from hepattn.experiments.clic.convert_to_native import (  # noqa: F401  (re-exported for the tests)
    TASK_NETS,
    assign_dense,
    port_any_attention,
    port_attention,
    port_decoder,
    port_dense,
    port_encoder,
    port_encoder_layer,
    port_input_net,
    port_maskformer,
    port_norm,
    port_residual,
    port_task,
)


@torch.no_grad()
def perturb_affine_params(module: nn.Module, seed: int) -> None:
    """Give biases, norm affines, LayerScale and register tokens non-default values.

    Default inits (zero biases, unit norm weights, constant LayerScale) would let a missing or swapped port of
    those parameters pass unnoticed; after this, a missing port changes outputs at O(0.1).
    """
    gen = torch.Generator().manual_seed(seed)
    for name, param in module.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"bias", "in_proj_bias", "gamma"} or (leaf == "weight" and param.dim() == 1):
            param.add_(0.1 * torch.randn(param.shape, generator=gen))
