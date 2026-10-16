"""Keras-native twin of ``hepattn.models.dense.Dense``: [Dense + act (+ dropout)] * hidden + Dense (+ final act).

Same layer structure and names as the hybrid ``hepattn.keras.dense.KerasDense`` (``{name}_hidden{i}``,
``{name}_final``), so weights port one-to-one. Sublayers are built in ``build`` from the static input shape.

SwiGLU (``x1 * silu(x2)``, x1 = the first half of torch's packed ``Linear(in, 2h)``) uses one graph in float and
quantized mode (plan RETARGET_ORG_MAIN.md section 5.4): the packed projection is two factory denses,
``{name}_hidden{i}`` (x1, no activation) and ``{name}_hidden{i}_gate`` (x2, ``activation="silu"`` inside the leaf, as
every other quantized activation), and the product is the elementwise factory einsum ``{name}_hidden{i}_glu``
(``QEinsum`` when quantized). Porting splits the torch weight rows: x1 = rows [:h], x2 = rows [h:].

Initialisation (``init``): ``"keras"`` (glorot-uniform kernel, zero bias) is what the hybrid's encoder/decoder
Dense layers get, because the hybrid builds those from the config. ``"torch"`` is ``nn.Linear``'s default,
kernel and bias U(+-1/sqrt(fan_in)); the hybrid's input nets and task heads start from it, because
``kerasify_module`` ports them from torch ``Dense`` modules.
"""

import math

import keras

from hepattn.keras_native.factory import LayerFactory

# torch activation names (as written in YAML / by hepattn) -> keras activation names
ACTIVATION_ALIASES = {
    "SiLU": "silu",
    "ReLU": "relu",
    "GELU": "gelu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
}
KERAS_ACTIVATIONS = {"silu", "relu", "gelu", "sigmoid", "tanh"}


def resolve_activation(activation: str | None, default: str | None) -> str | None:
    """Map a torch or keras activation name to a keras name; ``SwiGLU`` is kept as is.

    Raises:
        TypeError: If ``activation`` is not a string.
        ValueError: If the name is not a known activation.
    """
    if activation is None:
        return default
    if not isinstance(activation, str):
        raise TypeError(f"activation must be a name, got {type(activation).__name__}")
    name = activation.removeprefix("torch.nn.")
    if name == "SwiGLU":
        return name
    name = ACTIVATION_ALIASES.get(name, name)
    if name not in KERAS_ACTIVATIONS:
        raise ValueError(f"No keras activation for '{activation}'")
    return name


class Dense(keras.layers.Layer):
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
        init: str = "keras",
        factory: LayerFactory | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Args as in hepattn Dense; activations are names (``silu``, ``SiLU``, ``torch.nn.SiLU``, ``SwiGLU``).

        ``init`` is ``"keras"`` or ``"torch"`` (see the module docstring).

        Raises:
            ValueError: For SwiGLU as the final activation or an unknown ``init``.
            NotImplementedError: For dropout in quantized mode (as in the hybrid).
        """
        super().__init__(name=name, **kwargs)
        factory = factory or LayerFactory()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]
        if isinstance(hidden_layers, int):
            hidden_layers = [input_size * hidden_dim_scale] * hidden_layers
        activation = resolve_activation(activation, "silu")
        final_activation = resolve_activation(final_activation, None)
        if final_activation == "SwiGLU":
            raise ValueError("SwiGLU is not supported as a final activation")

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = list(hidden_layers)
        self.gate = activation == "SwiGLU"
        if factory.quantize and dropout:
            raise NotImplementedError("dropout is not supported in quantized mode")

        if init not in {"keras", "torch"}:
            raise ValueError(f"init must be 'keras' or 'torch', got '{init}'")

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        def initializers(fan_in: int) -> dict:
            if init == "keras":
                return {}
            # nn.Linear: kaiming_uniform(a=sqrt(5)) kernel and bias, both U(+-1/sqrt(fan_in))
            bound = 1.0 / math.sqrt(fan_in)
            uniform = keras.initializers.VarianceScaling(scale=1.0 / 3.0, mode="fan_in", distribution="uniform")
            return {"kernel_initializer": uniform, "bias_initializer": keras.initializers.RandomUniform(-bound, bound)}

        fan_ins = [input_size, *self.hidden_sizes]  # both SwiGLU halves have the fan-in of the packed torch Linear
        self.hidden = [
            factory.dense(size, activation=None if self.gate else activation, use_bias=bias, name=sub(f"hidden{i}"), **initializers(fan_ins[i]))
            for i, size in enumerate(self.hidden_sizes)
        ]
        self.gates, self.products = [], []
        if self.gate:
            self.gates = [
                factory.dense(size, activation="silu", use_bias=bias, name=sub(f"hidden{i}_gate"), **initializers(fan_ins[i]))
                for i, size in enumerate(self.hidden_sizes)
            ]
            self.products = [factory.einsum("...d,...d->...d", name=sub(f"hidden{i}_glu")) for i in range(len(self.hidden_sizes))]
        self.final = factory.dense(output_size, activation=final_activation, use_bias=bias, name=sub("final"), **initializers(fan_ins[-1]))
        self.dropout = keras.layers.Dropout(dropout, name=sub("dropout")) if dropout else None

    def build(self, input_shape) -> None:
        if input_shape[-1] != self.input_size:
            raise ValueError(f"{self.name}: input feature size {input_shape[-1]} != input_size {self.input_size}")
        lead = (None, *input_shape[1:-1])
        in_size = self.input_size
        for i, (layer, size) in enumerate(zip(self.hidden, self.hidden_sizes, strict=True)):
            layer.build((*lead, in_size))
            if self.gate:
                self.gates[i].build((*lead, in_size))
                self.products[i].build([(*lead, size), (*lead, size)])
            in_size = size
        self.final.build((*lead, in_size))

    def call(self, x, training=None):
        for i, layer in enumerate(self.hidden):
            if self.gate:
                x = self.products[i]([layer(x, training=training), self.gates[i](x, training=training)], training=training)
            else:
                x = layer(x, training=training)
            if self.dropout is not None:
                x = self.dropout(x, training=training)
        return self.final(x, training=training)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_size)
