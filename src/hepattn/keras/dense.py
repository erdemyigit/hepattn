"""Keras twin of hepattn.models.dense.Dense with an identical layer structure.

SwiGLU leaf rule (docs/hgq/RETARGET_ORG_MAIN.md section 5.4, shared with keras_native): torch packs the gate as one
``Linear(in, 2h)`` followed by ``x1 * silu(x2)``, x1 being the first half. Here each half is its own factory dense
(``_x1`` linear, ``_x2`` with ``activation="silu"`` inside the layer, as for every other quantized activation), and the
product is a factory einsum ``bnc,bnc->bnc`` (``QEinsum`` when quantized). Float mode uses the same graph.
"""

import torch
from torch import Tensor, nn

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
    """Port a torch Dense into its KerasDense twin; a packed SwiGLU Linear is split into its x1 and x2 halves."""
    linears = [m for m in dense.net if isinstance(m, nn.Linear)]
    assert len(linears) == len(kdense.hidden) + 1, f"structure mismatch: {len(linears)} torch Linears vs {len(kdense.hidden) + 1} keras Dense layers"
    for i, lin in enumerate(linears[:-1]):
        if not kdense.gate:
            port_linear(lin, kdense.hidden[i])
            continue
        w1, w2 = lin.weight.chunk(2, dim=0)
        b1, b2 = lin.bias.chunk(2, dim=0) if lin.bias is not None else (None, None)
        assign_dense(kdense.hidden[i], w1, b1)
        assign_dense(kdense.gates[i], w2, b2)
    port_linear(linears[-1], kdense.final)


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

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        # Float layers are built eagerly (parameters must exist before the optimizer is
        # created). Quantized layers build LAZILY on the first forward: HGQ2 datalane
        # quantizers materialize per-element bitwidth variables and the EBOPs parallelism
        # count from the full static input shape, which is only known from real data
        # (hepattn pads CLIC events to fixed max_nodes/num_queries, so shapes ARE static).
        # Training drivers must therefore run one forward before creating the optimizer.
        # With SwiGLU, hidden[i] is the x1 half, gates[i] the silu(x2) half, products[i] their product.
        hidden, gates, products = [], [], []
        node_list = [input_size, *hidden_layers]
        for i in range(len(node_list) - 1):
            if self.gate:
                layers = [
                    factory.dense(node_list[i + 1], activation=None, use_bias=bias, name=sub(f"hidden{i}_x1")),
                    factory.dense(node_list[i + 1], activation="silu", use_bias=bias, name=sub(f"hidden{i}_x2")),
                ]
                products.append(factory.einsum("bnc,bnc->bnc", name=sub(f"hidden{i}_glu")))
            else:
                layers = [factory.dense(node_list[i + 1], activation=activation, use_bias=bias, name=sub(f"hidden{i}"))]
            if not factory.quantize:
                for layer in layers:
                    layer.build((None, node_list[i]))
            hidden.append(layers[0])
            gates.extend(layers[1:])

        final = factory.dense(output_size, activation=final_activation, use_bias=bias, name=sub("final"))
        if not factory.quantize:
            final.build((None, node_list[-1]))

        self.hidden = nn.ModuleList(hidden)
        self.gates = nn.ModuleList(gates)
        self.products = nn.ModuleList(products)
        self.final = final
        # torch Dense whose weights are ported into this module at lazy build (quantized
        # mode only — eagerly-built float layers are ported immediately by the caller).
        # Stored via __dict__ to bypass nn.Module.__setattr__: registering it as a
        # submodule would leak its parameters into state_dict/optimizer and expose it
        # to later kerasify passes.
        self.__dict__["_pending_port"] = None
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
        for i, layer in enumerate(self.hidden):
            for sublayer in (layer, self.gates[i]) if self.gate else (layer,):
                if not sublayer.built:
                    sublayer.build((*lead, in_size))
            in_size = layer.units
        if not self.final.built:
            self.final.build((*lead, in_size))
        if self._pending_port is not None:
            port_dense(self._pending_port, self)
            self.__dict__["_pending_port"] = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.final.built or self._pending_port is not None:
            self._materialize(x)
        for i, layer in enumerate(self.hidden):
            x = self._swiglu(i, x) if self.gate else layer(x, training=self.training)
            if self.dropout is not None:
                x = self.dropout(x, training=self.training)
        return self.final(x, training=self.training)

    def _swiglu(self, i: int, x: Tensor) -> Tensor:
        """x1 * silu(x2) through the product einsum, whose equation is rank 3 (other ranks are folded into it)."""
        x1 = self.hidden[i](x, training=self.training)
        x2 = self.gates[i](x, training=self.training)
        shape = x1.shape
        if x1.dim() != 3:
            x1, x2 = x1.reshape(shape[0], -1, shape[-1]), x2.reshape(shape[0], -1, shape[-1])
        return self.products[i]([x1, x2], training=self.training).reshape(shape)
