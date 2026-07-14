"""Keras twin of hepattn.models.dense.Dense with an identical layer structure."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hepattn.keras import keras
from hepattn.keras.factory import LayerFactory
from hepattn.models.activation import SwiGLU
from hepattn.models.dense import Dense


def assign_dense(layer, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    """Assign a torch-convention (out, in) weight matrix + bias to a keras Dense-like layer.

    Raises:
        RuntimeError: If the target layer has not been built yet.
    """
    if not layer.built:
        raise RuntimeError(
            f"keras layer '{layer.name}' is not built. Quantized (HGQ2) layers build lazily on their "
            "first forward pass — run the model once on a representative batch before porting weights."
        )
    layer.kernel.assign(weight.detach().cpu().numpy().T)
    if bias is not None:
        layer.bias.assign(bias.detach().cpu().numpy())


@torch.no_grad()
def port_linear(linear: nn.Linear, layer) -> None:
    assign_dense(layer, linear.weight, linear.bias)


@torch.no_grad()
def port_dense(dense: Dense, kdense: "KerasDense") -> None:
    linears = [m for m in dense.net if isinstance(m, nn.Linear)]
    klayers = [*kdense.hidden, kdense.final]
    assert len(linears) == len(klayers), f"structure mismatch: {len(linears)} torch Linears vs {len(klayers)} keras Dense layers"
    for lin, klayer in zip(linears, klayers, strict=True):
        port_linear(lin, klayer)


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

    Float sublayers are built eagerly at construction so their parameters exist before
    Lightning's configure_optimizers runs. Quantized (HGQ2) sublayers build lazily at
    the first forward (see _materialize) — quantized training drivers must run one
    forward before creating the optimizer.
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

        # Float layers are built eagerly (parameters must exist before the optimizer is
        # created). Quantized layers build LAZILY on the first forward: HGQ2 datalane
        # quantizers materialize per-element bitwidth variables and the EBOPs parallelism
        # count from the full static input shape, which is only known from real data
        # (hepattn pads CLIC events to fixed max_nodes/num_queries, so shapes ARE static).
        # Training drivers must therefore run one forward before creating the optimizer.
        hidden = []
        node_list = [input_size, *hidden_layers]
        for i in range(len(node_list) - 1):
            proj_dim = node_list[i + 1] * 2 if self.gate else node_list[i + 1]
            layer = factory.dense(proj_dim, activation=None if self.gate else activation, use_bias=bias, name=f"{name}_hidden{i}" if name else None)
            if not factory.quantize:
                layer.build((None, node_list[i]))
            hidden.append(layer)

        final = factory.dense(output_size, activation=final_activation, use_bias=bias, name=f"{name}_final" if name else None)
        if not factory.quantize:
            final.build((None, node_list[-1]))

        self.hidden = nn.ModuleList(hidden)
        self.final = final
        # torch Dense whose weights are ported into this module at lazy build (quantized
        # mode only — eagerly-built float layers are ported immediately by the caller)
        self._pending_port: Dense | None = None
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

    def _materialize(self, x: Tensor) -> None:
        """Build lazy (quantized) sublayers from the true static input shape and port pending weights.

        HGQ2 layers need the full static shape (batch excluded) to size their per-element
        bitwidth variables and EBOPs parallelism; it is only known at the first forward.
        """
        lead = (None, *(int(d) for d in x.shape[1:-1]))
        in_size = self.input_size
        for layer in self.hidden:
            if not layer.built:
                layer.build((*lead, in_size))
            in_size = layer.units // 2 if self.gate else layer.units
        if not self.final.built:
            self.final.build((*lead, in_size))
        if self._pending_port is not None:
            port_dense(self._pending_port, self)
            self._pending_port = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.final.built or self._pending_port is not None:
            self._materialize(x)
        for layer in self.hidden:
            x = layer(x, training=self.training)
            if self.gate:
                x1, x2 = torch.chunk(x, 2, dim=-1)
                x = x1 * F.silu(x2)
            if self.dropout is not None:
                x = self.dropout(x, training=self.training)
        return self.final(x, training=self.training)
