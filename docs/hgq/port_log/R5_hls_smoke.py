r"""R5 gate: hls4ml code generation for each new HGQ2 leaf kind of the retarget (no C simulation: macOS).

Every leaf is built by the factory the models use (``hepattn.keras_native.factory.LayerFactory``, production spec of
polaris/03_make_config.py, renorm lanes from ``masked_linformer.renorm_bits``), wrapped in a functional keras model at
small shapes, called once in training mode (so the quantizer ranges settle), converted with hls4ml (Vitis,
io_parallel) and written. Rows: the masked-Linformer core leaves (plan section 5.6 table) and the quantized-SwiGLU
product (section 5.4). Control: the rsqrt LUT on a rank-3 input, which the design reshapes away because hls4ml
1.3.0 fails on it; it must FAIL here, so a passing row means something.

Known gap (R5 finding, plan section 7 risk 5): the renorm ``kept`` leaf as built, a ``QEinsum`` of the 1-bit mask
with the (1, n, k) energy table, does not convert ("Batch dim mismatch"): the table is a computed tensor with batch 1,
not a kernel. Its rows are marked ``GAP`` (expected to fail; a pass would mean the gap is gone, and is reported).
The S7 export form, a ``QEinsumDense`` with the frozen table ``W`` as its kernel and the same 1-bit mask lane
(``kept ... S7 form`` rows), must convert.

Run in the hybrid env (hls4ml is installed there, not in .venv-jax):

    KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu PYTHONPATH=src ../hepattn/.venv/bin/python \\
        docs/hgq/port_log/R5_hls_smoke.py <out_dir>

Prints one line per row and exits 1 unless every row is as expected.
"""

import sys
import traceback
from pathlib import Path

import hls4ml
import keras
import numpy as np
import torch
from hgq.layers import QEinsumDense

from hepattn.keras_native.factory import LayerFactory, _fixed_config  # noqa: PLC2701  (the factory's own fixed-lane config)
from hepattn.keras_native.masked_linformer import renorm_bits, rsqrt_floor

PROD = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0e-12},
}
H, D, NQ, N, K, C = 2, 4, 4, 6, 3, 5


def build(row: str, factory: LayerFactory) -> keras.Model:
    cap = factory.quant.seq_proj_frac_cap
    bits = renorm_bits(cap)
    with factory.scopes():
        if row == "E[:n] attend  QEinsumDense bhqn,nk->bhqk":
            x = keras.Input((H, NQ, N))
            return keras.Model(x, factory.einsum_dense("bhqn,nk->bhqk", (H, NQ, K), max_frac_bits=cap)(x))
        if row == "F[:n] attend  QEinsumDense bhqk,nk->bhqn":
            x = keras.Input((H, NQ, K))
            return keras.Model(x, factory.einsum_dense("bhqk,nk->bhqn", (H, NQ, N), max_frac_bits=cap)(x))
        if row == "E project     QEinsumDense bhnd,nk->bhkd":
            x = keras.Input((H, N, D))
            return keras.Model(x, factory.einsum_dense("bhnd,nk->bhkd", (H, K, D), max_frac_bits=cap)(x))
        if row == "GAP kept attend   QEinsum bqn,znk->bqk (1-bit mask, table lane)":
            m, w = keras.Input((NQ, N)), keras.Input(batch_shape=(1, N, K))  # the core feeds E^2 / sum E^2 as (1, n, k)
            return keras.Model([m, w], factory.mask_einsum("bqn,znk->bqk", table_bits=bits["lane"])([m, w]))
        if row == "GAP kept project  QEinsum bn,znk->bk (1-bit mask, table lane)":
            m, w = keras.Input((N,)), keras.Input(batch_shape=(1, N, K))
            return keras.Model([m, w], factory.mask_einsum("bn,znk->bk", table_bits=bits["lane"])([m, w]))
        if row.startswith("kept attend   S7 form"):
            m = keras.Input((NQ, N))
            mask_conf = _fixed_config("datalane", 1, 0)
            return keras.Model(m, QEinsumDense("bqn,nk->bqk", (NQ, K), iq_conf=mask_conf, bias_axes=None)(m))
        if row.startswith("kept project  S7 form"):
            m = keras.Input((N,))
            mask_conf = _fixed_config("datalane", 1, 0)
            return keras.Model(m, QEinsumDense("bn,nk->bk", (K,), iq_conf=mask_conf, bias_axes=None)(m))
        if row == "rsqrt         QUnaryFunctionLUT rank 2, production lanes":
            x = keras.Input((NQ * K,))
            return keras.Model(x, factory.lut(rsqrt_floor, in_bits=bits["lane"], out_bits=bits["lut_out"])(x))
        if row == "mul_k         QEinsum bhqk,bqk->bhqk":
            a, b = keras.Input((H, NQ, K)), keras.Input((NQ, K))
            return keras.Model([a, b], factory.einsum("bhqk,bqk->bhqk")([a, b]))
        if row == "qk            QEinsum bhqd,bhnd->bhqn":
            a, b = keras.Input((H, NQ, D)), keras.Input((H, N, D))
            return keras.Model([a, b], factory.einsum("bhqd,bhnd->bhqn")([a, b]))
        if row == "pv            QEinsum bhqn,bhnd->bhqd":
            a, b = keras.Input((H, NQ, N)), keras.Input((H, N, D))
            return keras.Model([a, b], factory.einsum("bhqn,bhnd->bhqd")([a, b]))
        if row == "softmax       QSoftmax over the k slots, no mask":
            x = keras.Input((H, NQ, K))
            return keras.Model(x, factory.softmax(axis=-1)(x))
        if row == "SwiGLU        QEinsum bnc,bnc->bnc":
            a, b = keras.Input((N, C)), keras.Input((N, C))
            return keras.Model([a, b], factory.einsum("bnc,bnc->bnc")([a, b]))
        if row == "CONTROL rsqrt QUnaryFunctionLUT rank 3 (must fail)":
            x = keras.Input((NQ, K))
            return keras.Model(x, factory.lut(rsqrt_floor, in_bits=bits["lane"], out_bits=bits["lut_out"])(x))
    raise ValueError(row)


ROWS = [
    "E[:n] attend  QEinsumDense bhqn,nk->bhqk",
    "F[:n] attend  QEinsumDense bhqk,nk->bhqn",
    "E project     QEinsumDense bhnd,nk->bhkd",
    "GAP kept attend   QEinsum bqn,znk->bqk (1-bit mask, table lane)",
    "GAP kept project  QEinsum bn,znk->bk (1-bit mask, table lane)",
    "kept attend   S7 form QEinsumDense bqn,nk->bqk (1-bit mask, W kernel)",
    "kept project  S7 form QEinsumDense bn,nk->bk (1-bit mask, W kernel)",
    "rsqrt         QUnaryFunctionLUT rank 2, production lanes",
    "mul_k         QEinsum bhqk,bqk->bhqk",
    "qk            QEinsum bhqd,bhnd->bhqn",
    "pv            QEinsum bhqn,bhnd->bhqd",
    "softmax       QSoftmax over the k slots, no mask",
    "SwiGLU        QEinsum bnc,bnc->bnc",
    "CONTROL rsqrt QUnaryFunctionLUT rank 3 (must fail)",
]


def main(out: Path) -> int:
    factory = LayerFactory(PROD)
    rng = np.random.default_rng(0)
    bad = 0
    for i, row in enumerate(ROWS):
        expect_ok = not row.startswith(("CONTROL", "GAP"))
        try:
            model = build(row, factory)
            xs = [rng.random((t.shape[0] or 8, *t.shape[1:])).astype("float32") for t in model.inputs]
            model(xs if len(xs) > 1 else xs[0], training=True)
            with torch.no_grad():
                hm = hls4ml.converters.convert_from_keras_model(model, output_dir=str(out / f"hls_{i:02d}"), backend="Vitis", io_type="io_parallel")
                hm.write()
            written = sorted(p.name for p in (out / f"hls_{i:02d}/firmware").glob("*.cpp"))
            kinds = [layer.class_name for layer in hm.get_layers()]
            ok, detail = True, f"{kinds} firmware {written}"
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            traceback.print_exc(limit=2, file=sys.stderr)
        bad += ok != expect_ok
        print(f"{'OK  ' if ok else 'FAIL'} {'(expected)' if ok == expect_ok else '(UNEXPECTED)'} {row:62s} -> {detail}", flush=True)
    print(f"hls4ml {hls4ml.__version__}, keras {keras.__version__}; {bad} unexpected")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
