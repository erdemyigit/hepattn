"""Keras-native twins of ``hepattn.models.input.InputNet`` and ``hepattn.models.posenc.FourierPositionEncoder``.

Functional (no in-place ``x += posenc``). ``B`` is a non-trainable weight, so it is saved and restored with the
model; its value is ported from torch for parity. Names: the embedding Dense is ``{name}_net`` (hybrid:
``innet{i}_net``, leaves ``innet{i}_net_hidden0`` / ``_final``), the posenc layer is ``{name}_posenc``.
"""

import math

import keras

from hepattn.keras_native.dense import Dense
from hepattn.keras_native.factory import LayerFactory


class FourierPositionEncoder(keras.layers.Layer):
    """Gaussian Fourier features (https://arxiv.org/abs/2006.10739): [sin(2 pi x B), cos(2 pi x B)]."""

    def __init__(self, input_name: str, dim: int, fields: list[str], scale: float = 1, name: str | None = None, **kwargs) -> None:
        """Fourier features of ``{input_name}_{field}`` for each field, ``dim`` outputs (sin and cos halves).

        Raises:
            ValueError: If scale <= 0 or dim is odd.
        """
        super().__init__(name=name, **kwargs)
        if scale <= 0:
            raise ValueError("scale must be > 0")
        if dim % 2 != 0:
            raise ValueError("Dimension must be even")
        self.input_name = input_name
        self.dim = dim
        self.fields = list(fields)
        self.scale = scale

    def build(self, inputs_shape) -> None:
        # torch: Buffer(scale * torch.randn(len(fields), dim // 2))
        self.B = self.add_weight(
            name="B", shape=(len(self.fields), self.dim // 2), initializer=keras.initializers.RandomNormal(stddev=self.scale), trainable=False
        )

    def call(self, inputs: dict):
        xs = keras.ops.stack([inputs[f"{self.input_name}_{field}"] for field in self.fields], axis=-1)
        xs = (2 * math.pi) * xs  # torch: 2 * float32(pi) * xs, the same float32 constant
        xs = keras.ops.matmul(xs, keras.ops.cast(self.B, xs.dtype))
        return keras.ops.concatenate([keras.ops.sin(xs), keras.ops.cos(xs)], axis=-1)


def concat_fields(tensors: list):
    """Concatenate along the last axis, with (B, N) tensors expanded to (B, N, 1) (``hepattn.utils.tensor_utils.concat_tensors``)."""
    return keras.ops.concatenate([keras.ops.expand_dims(t, -1) if len(t.shape) == 2 else t for t in tensors], axis=-1)


class InputNet(keras.layers.Layer):
    def __init__(
        self,
        input_name: str,
        net: dict,
        fields: list[str],
        posenc: dict | None = None,
        input_dtype: str | None = None,
        factory: LayerFactory | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Embed ``{input_name}_{field}`` features with a Dense and add an optional Fourier positional encoding.

        Args:
            input_name: Constituent name (e.g. ``node``).
            net: ``Dense`` kwargs (``input_size``, ``output_size``, ...).
            fields: Feature fields concatenated along the last axis.
            posenc: ``FourierPositionEncoder`` kwargs (``dim``, ``fields``, ``scale``; ``input_name`` defaults to ours),
                or None.
            input_dtype: Must be None (a separate input precision is not ported).
            factory: Layer factory (float or HGQ2).
            name: Layer name and prefix of the sub-layer names.
            **kwargs: Passed to ``keras.layers.Layer``.

        Raises:
            NotImplementedError: For ``input_dtype`` or a posenc class other than FourierPositionEncoder.
        """
        super().__init__(name=name, **kwargs)
        if input_dtype is not None:
            raise NotImplementedError("input_dtype is not supported by the keras_native InputNet")
        factory = factory or LayerFactory()

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        self.input_name = input_name
        self.fields = list(fields)
        self.net = Dense(factory=factory, name=sub("net"), **net)
        self.posenc = None
        if posenc is not None:
            posenc = dict(posenc)
            class_path = posenc.pop("class_path", "FourierPositionEncoder")
            if class_path.rsplit(".", 1)[-1] != "FourierPositionEncoder":
                raise NotImplementedError(f"posenc '{class_path}' is not ported (only FourierPositionEncoder)")
            posenc.setdefault("input_name", input_name)
            self.posenc = FourierPositionEncoder(name=sub("posenc"), **posenc)

    def build(self, inputs_shape) -> None:
        shapes = [inputs_shape[f"{self.input_name}_{field}"] for field in self.fields]
        width = sum(1 if len(s) == 2 else s[-1] for s in shapes)
        self.net.build((None, *shapes[0][1:2], width))
        if self.posenc is not None:
            self.posenc.build(inputs_shape)

    def call(self, inputs: dict, training=None):
        x = self.net(concat_fields([inputs[f"{self.input_name}_{field}"] for field in self.fields]), training=training)
        if self.posenc is not None:
            x = x + self.posenc(inputs)
        return x
