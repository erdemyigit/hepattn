"""hls4ml 1.3.0 codegen (no C-sim: macOS) for each new leaf kind of the quantized masked-linformer design."""
import sys
import traceback

import hls4ml
import keras
import numpy as np
import torch
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.layers import QEinsum, QEinsumDense, QSoftmax, QUnaryFunctionLUT
from hgq.quantizer import QuantizerConfig

OUT = sys.argv[1]
H, NQ, N, K = 2, 4, 6, 3


def rsqrt_floor(x):
    return keras.ops.rsqrt(keras.ops.maximum(x, 1e-16))


def scopes():
    s = [QuantizerConfigScope(place="weight", default_q_type="kbi", b0=8, i0=2),
         QuantizerConfigScope(place="datalane", default_q_type="kif", i0=4, f0=8),
         QuantizerConfigScope(place="table", default_q_type="kif", i0=2, f0=10),
         LayerConfigScope(enable_ebops=True, beta0=1e-5)]
    return s


def build(name):
    for sc in (stack := scopes()):
        sc.__enter__()
    try:
        if name == "E_proj  QEinsumDense bhqn,nk->bhqk":
            x = keras.Input((H, NQ, N)); y = QEinsumDense("bhqn,nk->bhqk", (H, NQ, K))(x); ins = [x]
        elif name == "F_proj  QEinsumDense bhqk,nk->bhqn":
            x = keras.Input((H, NQ, K)); y = QEinsumDense("bhqk,nk->bhqn", (H, NQ, N))(x); ins = [x]
        elif name == "mul_k   QEinsum bhqk,bqk->bhqk":
            a, b = keras.Input((H, NQ, K)), keras.Input((NQ, K)); y = QEinsum("bhqk,bqk->bhqk")([a, b]); ins = [a, b]
        elif name == "kept    QEinsumDense bqn,nk->bqk (1-bit mask lane, W kernel)":
            m = keras.Input((NQ, N))
            iq = QuantizerConfig("kif", "datalane", k0=False, i0=1, f0=0, overflow_mode="SAT", trainable=False, homogeneous_axis=None, heterogeneous_axis=())
            y = QEinsumDense("bqn,nk->bqk", (NQ, K), iq_conf=iq)(m); ins = [m]
        elif name == "rsqrt   QUnaryFunctionLUT rank-2, homogeneous":
            x = keras.Input((NQ * K,))
            iq = QuantizerConfig("kif", "datalane", k0=False, i0=1, f0=12, overflow_mode="SAT", homogeneous_axis=None, heterogeneous_axis=())
            oq = QuantizerConfig("kif", "table", k0=False, i0=7, f0=8)
            y = QUnaryFunctionLUT(rsqrt_floor, iq_conf=iq, oq_conf=oq, allow_heterogeneous_input=False, allow_heterogeneous_table=False)(x); ins = [x]
        elif name == "rsqrt   QUnaryFunctionLUT rank-3 (b, nq, k)":
            x = keras.Input((NQ, K))
            iq = QuantizerConfig("kif", "datalane", k0=False, i0=1, f0=12, overflow_mode="SAT", homogeneous_axis=None, heterogeneous_axis=())
            oq = QuantizerConfig("kif", "table", k0=False, i0=7, f0=8)
            y = QUnaryFunctionLUT(rsqrt_floor, iq_conf=iq, oq_conf=oq, allow_heterogeneous_input=False, allow_heterogeneous_table=False)(x); ins = [x]
        elif name == "softmax QSoftmax over k, no mask":
            x = keras.Input((H, NQ, K)); y = QSoftmax(axis=-1)(x); ins = [x]
        return keras.Model(ins, y)
    finally:
        for sc in reversed(stack):
            sc.__exit__(None, None, None)


names = ["E_proj  QEinsumDense bhqn,nk->bhqk", "F_proj  QEinsumDense bhqk,nk->bhqn", "mul_k   QEinsum bhqk,bqk->bhqk",
         "kept    QEinsumDense bqn,nk->bqk (1-bit mask lane, W kernel)", "rsqrt   QUnaryFunctionLUT rank-2, homogeneous",
         "softmax QSoftmax over k, no mask", "rsqrt   QUnaryFunctionLUT rank-3 (b, nq, k)"]
for i, name in enumerate(names):
    model = build(name)
    rng = np.random.default_rng(0)
    xs = [rng.random((8, *t.shape[1:])).astype("float32") for t in model.inputs]
    model(xs if len(xs) > 1 else xs[0], training=True)  # one train-mode call so quantizer ranges settle
    try:
        with torch.no_grad():
            hm = hls4ml.converters.convert_from_keras_model(model, output_dir=f"{OUT}/hls_{i}", backend="Vitis", io_type="io_parallel")
            hm.write()
        kinds = [layer.class_name for layer in hm.get_layers()]
        print(f"OK    {name:62s} -> {kinds}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name:62s} -> {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(limit=2, file=sys.stderr)
