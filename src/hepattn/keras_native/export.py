"""Export of trained keras_native layers: functional assembly and hls4ml conversion (plan section 3.12, S7).

The training model is a nested-layer graph with float glue (norms, selections, sorting, matching) that hls4ml cannot
consume as a whole (docs/hgq/FPGA_STATUS.md). The deployable pieces are assembled as fixed-shape keras functional
models FROM THE LIVE TRAINED LAYERS (weights and quantizer state shared, nothing copied), so a functional model gives
the trained layers' outputs bit for bit:

- ``build_functional_dense``: a native ``Dense`` (hidden layers + final; SwiGLU: ``x1``, ``x2 = silu(...)`` and the
  ``glu`` product einsum, folded to rank 3 as ``Dense._swiglu`` does);
- ``leaf_model``: one HGQ2 leaf on its build shapes (the per-leaf-kind hls4ml gate). A leaf without an input quantizer
  of its own (``QSoftmax``: in the model the previous layer's result type sets its input precision) gets a fixed
  ``INPUT_LANE`` quantizer in front, else hls4ml would read the float input with its default precision;
- export forms, for the leaf kinds hls4ml 1.3.0 does not convert, or not bit-exactly, as trained:
  - ``kept_export_form``: the renorm ``kept`` leaf, a ``QEinsum`` of the 1-bit mask with the computed (1, n, k) energy
    table, fails with "Batch dim mismatch for the inputs and output" (the table is a tensor with batch 1, not a
    kernel). Its export form is a ``QEinsumDense`` whose kernel is the frozen table ``W = E[:n]^2 / sum E^2``
    (``SeqProjection.energy`` in eval mode) with the leaf's own two input quantizers (1-bit mask lane, fixed table
    lane) as its input and kernel quantizers: the same products of the same quantized values, so the outputs are
    equal (``tests/keras_native/test_export.py``);
  - ``activation_export_form``: a ``QDense`` with an inline activation other than linear / relu. ``silu`` (task heads,
    SwiGLU ``x2``) does not convert: on a rank-3 input hls4ml raises in ``bit_exact.py`` (``np.ones(k)`` on an array,
    hls4ml 1.3.0 line 602), and on a rank-2 input it writes a call to ``nnet::silu``, which no hls4ml 1.3.0 template
    defines. ``sigmoid`` (the encoder value-residual mix) converts, but hls4ml's sigmoid is its own table, so C
    simulation differs from keras (measured, docs/hgq/port_log/S7.md). The export form is the same ``QDense`` without
    activation followed by a rank-2 ``QUnaryFunctionLUT`` of the activation with fixed signed input and output lanes:
    bit-exact in hls4ml, but NOT the trained graph: the table quantizes the pre-activation and its output, so it
    deviates from the trained layer by about half a step of those lanes (measured in the tests). A bit-exact
    deployable activation needs the table form at training time (a model change, docs/hgq/FPGA_STATUS.md).

``convert_to_hls`` runs hls4ml's HGQ2 flow (no precision configuration: HGQ2 models propagate their own precisions).
hls4ml reads values with ``keras.ops.convert_to_numpy``, so this works under any keras backend; no torch here.
"""

import math
from pathlib import Path

import keras
import numpy as np
from hgq.config import QuantizerConfig
from hgq.layers import QDense, QEinsumDense, QUnaryFunctionLUT

from hepattn.keras_native.dense import Dense

KEPT_EQUATIONS = {"bqn,znk->bqk": "bqn,nk->bqk", "bn,znk->bk": "bn,nk->bk"}
BIT_EXACT_ACTIVATIONS = {None, "linear", "relu"}  # inline QDense activations hls4ml reproduces bit for bit
INPUT_LANE = (8, 12)  # (integer, fractional) bits of the quantizer leaf_model puts before a leaf without one


def _fold(x, size: int):
    """Rank-3 view ``(B, prod(inner), size)`` of ``x`` (the shape ``Dense._swiglu`` gives the product einsum)."""
    inner = tuple(x.shape[1:-1])
    return keras.layers.Reshape((math.prod(inner), size))(x), inner


def build_functional_dense(ndense: Dense, input_shape: tuple[int, ...], name: str = "head", *, activations: str = "inline") -> keras.Model:
    """A keras functional model of a native ``Dense`` from its live sublayers.

    Args:
        ndense: The built (float or HGQ2) native ``Dense``, or the hybrid's ``KerasDense`` (the same ``hidden``,
            ``gates``, ``products``, ``final``, ``gate``, ``dropout`` layout; ``hepattn.keras.export`` delegates here).
        input_shape: Per-sample input shape the layers were built with, e.g. ``(num_queries, dim)`` for a per-query head:
            HGQ2 datalane quantizers are per element of that shape.
        name: Model name.
        activations: ``"inline"`` uses the trained layers (bit-identical to the native ``Dense``); ``"lut"`` replaces
            every ``QDense`` whose inline activation hls4ml does not reproduce (``needs_activation_export``) by its
            ``activation_export_form`` layers (hls4ml-convertible and bit-exact there, approximate to the trained layer).

    Returns:
        A keras.Model sharing the native layers' variables (``activations="lut"``: the replaced QDense layers are copies).

    Raises:
        ValueError: For an unknown ``activations`` mode or a dropout layer (training-only).
    """
    if activations not in {"inline", "lut"}:
        raise ValueError(f"activations must be 'inline' or 'lut', got {activations!r}")
    if ndense.dropout is not None:
        raise ValueError(f"{ndense.name}: dropout is training-only; export a model without dropout")

    def apply(layer, x):
        if activations == "lut" and needs_activation_export(layer):
            for part in activation_export_layers(layer):
                x = part(x)
            return x
        return layer(x)

    inputs = keras.Input(shape=input_shape, name=f"{name}_input")
    x = inputs
    for i, layer in enumerate(ndense.hidden):
        if not ndense.gate:
            x = apply(layer, x)
            continue
        x1, x2 = apply(layer, x), apply(ndense.gates[i], x)
        size = x1.shape[-1]
        if len(x1.shape) == 3:
            x = ndense.products[i]([x1, x2])
        else:
            (f1, inner), (f2, _) = _fold(x1, size), _fold(x2, size)
            x = keras.layers.Reshape((*inner, size))(ndense.products[i]([f1, f2]))
    return keras.Model(inputs, ndense.final(x), name=name)


def needs_activation_export(layer) -> bool:
    """A ``QDense`` whose inline activation hls4ml does not reproduce bit for bit (anything but linear / relu)."""
    return isinstance(layer, QDense) and _activation_name(layer) not in BIT_EXACT_ACTIVATIONS


def _activation_name(layer) -> str | None:
    act = layer.get_config().get("activation")
    return act if isinstance(act, str) else getattr(layer.activation, "__name__", None)


def _input_specs(layer) -> list[keras.KerasTensor]:
    shapes = layer._build_shapes_dict["input_shape"]  # noqa: SLF001  (keras records the build shapes here)
    multi = isinstance(shapes, list)
    specs = []
    for i, shape in enumerate(shapes if multi else [shapes]):
        # a leading 1 is a batch-1 table operand (the kept leaf's (1, n, k)), not the batch axis
        specs.append(keras.Input(batch_shape=tuple(shape)) if shape[0] == 1 else keras.Input(shape=tuple(shape[1:]), name=f"{layer.name}_in{i}"))
    return specs


def leaf_model(layer, name: str | None = None) -> keras.Model:
    """One trained leaf as a functional model on the shapes it was built with (weights and quantizers shared).

    A leaf without an input quantizer (``enable_iq`` False, e.g. ``QSoftmax``) reads its input through a fixed signed
    ``INPUT_LANE`` quantizer (see the module docstring).
    """
    from hgq.quantizer import Quantizer  # noqa: PLC0415

    specs = _input_specs(layer)
    xs = list(specs)
    if not getattr(layer, "enable_iq", True):
        xs = [Quantizer(signed_lane(*INPUT_LANE), name=f"{layer.name}_input_lane{i}")(x) for i, x in enumerate(xs)]
    out = layer(xs if len(xs) > 1 else xs[0])
    return keras.Model(specs if len(specs) > 1 else specs[0], out, name=name or f"{layer.name}_model")


def kept_export_form(kept, energy, name: str | None = None) -> QEinsumDense:
    """The renorm ``kept`` leaf as a ``QEinsumDense`` with the frozen table as kernel (built; see the module docstring).

    Args:
        kept: The trained ``QEinsum`` (``bqn,znk->bqk`` attend, ``bn,znk->bk`` project) of a masked-Linformer core.
        energy: The (n, k) table it multiplies the mask with: ``core.E.energy(training=False)`` (``kept_k``) or
            ``core.F.energy(training=False)`` (``kept_v``).
        name: Layer name (default ``<kept name>_export``).

    Raises:
        ValueError: For another equation or a table of another shape.
    """
    equation = KEPT_EQUATIONS.get(kept.get_config()["equation"])
    if equation is None:
        raise ValueError(f"{kept.name}: not a renorm kept leaf (equation {kept.get_config()['equation']!r})")
    table = np.asarray(keras.ops.convert_to_numpy(energy), dtype=np.float32)
    mask_shape = tuple(kept._build_shapes_dict["input_shape"][0])  # noqa: SLF001
    if table.shape != tuple(kept._build_shapes_dict["input_shape"][1][1:]):  # noqa: SLF001
        raise ValueError(f"{kept.name}: table {table.shape} vs the leaf's {kept._build_shapes_dict['input_shape'][1]}")  # noqa: SLF001
    mask_q, table_q = kept.iq.quantizers
    output_shape = (*mask_shape[1:-1], table.shape[-1])
    layer = QEinsumDense(
        equation,
        output_shape,
        bias_axes=None,
        iq_conf=mask_q.config,
        kq_conf=table_q.config,
        kernel_initializer=keras.initializers.Constant(table),
        enable_ebops=False,
        name=name or f"{kept.name}_export",
    )
    layer.build(mask_shape)
    layer.kernel.assign(table)
    return layer


def signed_lane(int_bits: int, frac_bits: int, place: str = "datalane") -> QuantizerConfig:
    """Fixed, non-trainable, signed saturating KIF lane with one bitwidth for the whole tensor."""
    return QuantizerConfig(
        "kif", place, k0=True, i0=int_bits, f0=frac_bits, overflow_mode="SAT_SYM", trainable=False, homogeneous_axis=None, heterogeneous_axis=()
    )


def activation_export_layers(qdense: QDense, in_bits: tuple[int, int] = (4, 6), out_bits: tuple[int, int] = (4, 8)) -> list:
    """The layers of ``activation_export_form``, to apply in sequence.

    A functional model must not nest a model: hls4ml parses a nested model through its da4ml fallback, which aborts
    the process in da4ml 0.6.0 ``cmvm``.

    Raises:
        ValueError: If ``qdense`` has no activation to replace (linear / relu are bit-exact as trained).
        RuntimeError: If the linear twin does not have the leaf's variables.
    """
    if not needs_activation_export(qdense):
        raise ValueError(f"{qdense.name}: activation {_activation_name(qdense)!r} needs no export form")
    fn = qdense.activation
    shape = tuple(qdense._build_shapes_dict["input_shape"])  # noqa: SLF001
    linear = QDense.from_config({**qdense.get_config(), "activation": None, "name": f"{qdense.name}_linear"})
    linear.build(shape)
    if len(linear.variables) != len(qdense.variables):
        raise RuntimeError(f"{qdense.name}: the linear twin has {len(linear.variables)} variables, the leaf {len(qdense.variables)}")
    for target, source in zip(linear.variables, qdense.variables, strict=True):
        target.assign(source.value)
    units = qdense.units
    inner = shape[1:-1]
    lut = QUnaryFunctionLUT(
        fn,
        iq_conf=signed_lane(*in_bits),
        oq_conf=signed_lane(*out_bits, place="table"),
        allow_heterogeneous_input=False,
        allow_heterogeneous_table=False,
        enable_ebops=False,
        name=f"{qdense.name}_{_activation_name(qdense)}_lut",
    )
    if not inner:
        return [linear, lut]
    return [linear, keras.layers.Reshape((math.prod(inner) * units,)), lut, keras.layers.Reshape((*inner, units))]


def activation_export_form(qdense: QDense, in_bits: tuple[int, int] = (4, 6), out_bits: tuple[int, int] = (4, 8)) -> keras.Model:
    """``QDense(activation=f)`` -> the same ``QDense`` without activation + a rank-2 ``QUnaryFunctionLUT(f)``.

    The linear part is a copy of ``qdense`` (kernel, bias, every quantizer); the table has ``2**(1 + sum(in_bits))``
    entries (hls4ml accepts at most 1e6). Approximate: see the module docstring.
    """
    shape = tuple(qdense._build_shapes_dict["input_shape"])  # noqa: SLF001
    x = inputs = keras.Input(shape=shape[1:], name=f"{qdense.name}_in")
    for layer in activation_export_layers(qdense, in_bits, out_bits):
        x = layer(x)
    return keras.Model(inputs, x, name=f"{qdense.name}_act_export")


def save_keras_model(model: keras.Model, path: str | Path) -> None:
    model.save(str(path))


def load_keras_model(path: str | Path) -> keras.Model:
    """``keras.models.load_model`` with the hepattn table functions (``rsqrt_floor`` is not a registered serializable)."""
    import hgq.layers  # noqa: F401, PLC0415  (registers the HGQ2 classes)

    from hepattn.keras_native.masked_linformer import rsqrt_floor  # noqa: PLC0415

    return keras.models.load_model(str(path), custom_objects={"rsqrt_floor": rsqrt_floor})


def convert_to_hls(model: keras.Model, output_dir: str | Path, backend: str = "Vitis", io_type: str = "io_parallel", compile_csim: bool = True):
    """Convert a functional HGQ2 model with hls4ml's bit-exact flow (no precision configuration is passed).

    Args:
        model: The functional HGQ2 keras model.
        output_dir: Where the HLS project is written.
        backend: hls4ml backend.
        io_type: hls4ml io type.
        compile_csim: Also compile the C-simulation library (``hls_model.predict``). Needs a toolchain that builds
            hls4ml's ap_types headers: Apple clang / libc++ does not, so run the bit-exactness gates on Linux.

    Returns:
        The hls4ml ModelGraph, written (and compiled with ``compile_csim``).
    """
    import hls4ml  # noqa: PLC0415  (heavy, optional)

    hls_model = hls4ml.converters.convert_from_keras_model(model, output_dir=str(output_dir), backend=backend, io_type=io_type)
    if compile_csim:
        hls_model.compile()
    else:
        hls_model.write()
    return hls_model
