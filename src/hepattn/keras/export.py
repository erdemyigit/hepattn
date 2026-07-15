"""Functional-model assembly and hls4ml conversion for the deployable subgraphs.

The training model is an orchestrated hybrid (torch glue + keras leaves), which hls4ml
cannot consume directly. For deployment, fixed-shape keras functional models are
assembled FROM THE LIVE LAYER INSTANCES (weights shared, no copying) for the
convertible subgraphs — task-head MLPs and the attention core between the float
norms. The float-boundary pieces (LayerNorms, residual glue, mask thresholding) are
documented in docs/hgq/FPGA_STATUS.md.
"""

from pathlib import Path

from hepattn.keras import keras
from hepattn.keras.dense import KerasDense


def build_functional_dense(kdense: KerasDense, input_shape: tuple[int, ...], name: str = "head") -> keras.Model:
    """Assemble a keras functional model from a KerasDense's live sublayers.

    Args:
        kdense: The (float or HGQ2) KerasDense whose layers to assemble.
        input_shape: Per-sample input shape, e.g. (150, 32) for per-query heads —
            must match the rank/shape the layers were built with (HGQ2 quantizers
            materialize per-element bitwidths for a specific static shape).
        name: Model name.

    Returns:
        A keras.Model sharing the KerasDense's weights.

    Raises:
        NotImplementedError: For gated (SwiGLU) nets, whose gating is torch glue.
    """
    if kdense.gate:
        raise NotImplementedError("SwiGLU-gated KerasDense cannot be exported as a functional model")
    inputs = keras.Input(shape=input_shape, name=f"{name}_input")
    x = inputs
    for layer in [*kdense.hidden, kdense.final]:
        x = layer(x)
    return keras.Model(inputs, x, name=name)


def save_keras_model(model: keras.Model, path: str | Path) -> None:
    model.save(str(path))


def load_keras_model(path: str | Path) -> keras.Model:
    return keras.models.load_model(str(path))


def convert_to_hls(model: keras.Model, output_dir: str | Path, backend: str = "Vitis", io_type: str = "io_parallel", compile_csim: bool = True):
    """Convert a functional HGQ2 model with hls4ml's bit-exact flow.

    Per the hls4ml HGQ2 documentation, NO precision configuration is passed:
    HGQ2 models trigger model-wise precision propagation that keeps the HLS
    bit-exact with keras.

    Args:
        model: The functional HGQ2 keras model.
        output_dir: Where the HLS project is written.
        backend: hls4ml backend.
        io_type: hls4ml io type.
        compile_csim: Also compile the C-simulation library so hls_model.predict
            works. Requires a toolchain compatible with hls4ml's ap_types headers
            (Apple clang/libc++ is NOT — run the bit-exactness gate on linux).

    Returns:
        The hls4ml ModelGraph (with a runnable .predict if compile_csim succeeded).
    """
    import hls4ml  # noqa: PLC0415  (heavy import, keep optional at module level)
    import torch  # noqa: PLC0415

    # no_grad: under the torch backend, the converter materializes LUT tables by
    # calling activations on domain grids and then .numpy()s the result, which fails
    # on grad-tracking tensors
    with torch.no_grad():
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            output_dir=str(output_dir),
            backend=backend,
            io_type=io_type,
        )
        if compile_csim:
            hls_model.compile()
        else:
            hls_model.write()
    return hls_model
