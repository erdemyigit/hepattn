"""Keras twin of hepattn.models.dense.Dense with an identical layer structure."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hepattn.keras import keras
from hepattn.keras.factory import LayerFactory
from hepattn.models.activation import SwiGLU
from hepattn.models.dense import Dense

# torch activation-module class -> keras activation name
ACTIVATION_NAMES: dict[type[nn.Module], str | None] = {
    nn.SiLU: "silu",
    nn.ReLU: "relu",
    nn.GELU: "gelu",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.Identity: None,
}


def activation_name(module: nn.Module) -> str:
    if isinstance(module, SwiGLU):
        return "SwiGLU"
    for cls, name in ACTIVATION_NAMES.items():
        if type(module) is cls and name is not None:
            return name
    raise NotImplementedError(f"No keras activation mapping for torch module {type(module).__name__}")


class KerasDense(nn.Module):
    """Mirror of hepattn Dense: [Linear+act (+dropout)]*hidden + Linear (+final act).

    All keras sublayers are built eagerly at construction so their parameters exist
    before Lightning's configure_optimizers runs (keras builds lazily by default).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: int | list[int] | None = None,
        hidden_dim_scale: int = 2,
        activation: str | None = None,
        final_activation: str | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        factory: LayerFactory | None = None,
        name: str | None = None,
    ) -> None:
        """Args mirror hepattn Dense; ``name`` is a deterministic keras-name prefix.

        Explicit names matter: keras embeds the layer name in the torch parameter keys
        (``..._torch_params.<name>/kernel``), so auto-generated names (global counter)
        would make state_dict keys depend on construction order within the process.
        """
        super().__init__()
        factory = factory or LayerFactory()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]
        if isinstance(hidden_layers, int):
            hidden_layers = [input_size * hidden_dim_scale] * hidden_layers
        if activation is None:
            activation = "silu"

        self.input_size = input_size
        self.output_size = output_size
        self.gate = activation == "SwiGLU"
        if factory.quantize and self.gate:
            raise NotImplementedError("SwiGLU is not supported in quantized mode (not used by the CLIC config)")

        hidden = []
        node_list = [input_size, *hidden_layers]
        for i in range(len(node_list) - 1):
            proj_dim = node_list[i + 1] * 2 if self.gate else node_list[i + 1]
            layer = factory.dense(proj_dim, activation=None if self.gate else activation, use_bias=bias, name=f"{name}_hidden{i}" if name else None)
            layer.build((None, node_list[i]))
            hidden.append(layer)

        final = factory.dense(output_size, activation=final_activation, use_bias=bias, name=f"{name}_final" if name else None)
        final.build((None, node_list[-1]))

        self.hidden = nn.ModuleList(hidden)
        self.final = final
        self.dropout = None
        if dropout:
            if factory.quantize:
                raise NotImplementedError("dropout is not supported in quantized mode")
            self.dropout = keras.layers.Dropout(dropout, name=f"{name}_dropout" if name else None)

    @classmethod
    def from_torch(cls, dense: Dense, factory: LayerFactory | None = None, name: str | None = None) -> "KerasDense":
        """Build a structurally identical KerasDense by introspecting a torch Dense (weights not ported)."""
        modules = list(dense.net)
        linears = [m for m in modules if isinstance(m, nn.Linear)]
        gate = any(isinstance(m, SwiGLU) for m in modules)

        # the activation module directly after the first Linear (if there is a hidden block)
        act = "silu"
        if len(linears) > 1:
            act = activation_name(modules[modules.index(linears[0]) + 1])

        final_act = None
        idx_last = len(modules) - 1 - modules[::-1].index(linears[-1])
        if idx_last + 1 < len(modules):
            final_act = activation_name(modules[idx_last + 1])

        dropouts = [m for m in modules if isinstance(m, nn.Dropout)]
        hidden_sizes = [lin.out_features // (2 if gate else 1) for lin in linears[:-1]]

        return cls(
            input_size=dense.input_size,
            output_size=dense.output_size,
            hidden_layers=hidden_sizes,
            activation=act,
            final_activation=final_act,
            dropout=dropouts[0].p if dropouts else 0.0,
            bias=linears[0].bias is not None,
            factory=factory,
            name=name,
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.hidden:
            x = layer(x, training=self.training)
            if self.gate:
                x1, x2 = torch.chunk(x, 2, dim=-1)
                x = x1 * F.silu(x2)
            if self.dropout is not None:
                x = self.dropout(x, training=self.training)
        return self.final(x, training=self.training)
