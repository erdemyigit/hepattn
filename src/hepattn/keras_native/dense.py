"""Keras-native twin of ``hepattn.models.dense.Dense``: [Dense + act (+ dropout)] * hidden + Dense (+ final act).

Same layer structure and names as the hybrid ``hepattn.keras.dense.KerasDense`` (``{name}_hidden{i}``,
``{name}_final``), so weights port one-to-one. Sublayers are built in ``build`` from the static input shape.
"""

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
        factory: LayerFactory | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Args as in hepattn Dense; activations are names (``silu``, ``SiLU``, ``torch.nn.SiLU``, ``SwiGLU``).

        Raises:
            ValueError: For SwiGLU as the final activation.
            NotImplementedError: For SwiGLU or dropout in quantized mode (as in the hybrid).
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
        if factory.quantize and self.gate:
            raise NotImplementedError("SwiGLU is not supported in quantized mode (not used by the CLIC config)")
        if factory.quantize and dropout:
            raise NotImplementedError("dropout is not supported in quantized mode")

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        self.hidden = [
            factory.dense(size * 2 if self.gate else size, activation=None if self.gate else activation, use_bias=bias, name=sub(f"hidden{i}"))
            for i, size in enumerate(self.hidden_sizes)
        ]
        self.final = factory.dense(output_size, activation=final_activation, use_bias=bias, name=sub("final"))
        self.dropout = keras.layers.Dropout(dropout, name=sub("dropout")) if dropout else None

    def build(self, input_shape) -> None:
        if input_shape[-1] != self.input_size:
            raise ValueError(f"{self.name}: input feature size {input_shape[-1]} != input_size {self.input_size}")
        lead = (None, *input_shape[1:-1])
        in_size = self.input_size
        for layer, size in zip(self.hidden, self.hidden_sizes, strict=True):
            layer.build((*lead, in_size))
            in_size = size
        self.final.build((*lead, in_size))

    def call(self, x, training=None):
        for layer in self.hidden:
            x = layer(x, training=training)
            if self.gate:
                x1, x2 = keras.ops.split(x, 2, axis=-1)
                x = x1 * keras.activations.silu(x2)
            if self.dropout is not None:
                x = self.dropout(x, training=training)
        return self.final(x, training=training)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_size)
