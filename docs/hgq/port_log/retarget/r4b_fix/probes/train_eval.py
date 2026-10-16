"""Train vs eval core output inside the hybrid attention, by input scale (does training=False change anything?)."""
import sys, torch
import hepattn.keras; hepattn.keras.set_keras_default_device("cpu")
sys.path.insert(0, sys.argv[1] + "/tests/keras")
import test_linformer_parity as T
C = T.C
for shape in ("q_ca", "enc"):
    for spec_name in C.SPECS:
        for scale in (1, 4, 16):
            x = T.make_inputs(shape, 41, sort=shape == "q_ca")
            x = {k: (v * scale if k in ("q", "kv") and v is not None else v) for k, v in x.items()}
            if T.make_inputs(shape, 41)["kv"] is T.make_inputs(shape, 41)["q"]:
                pass
            if shape in T.SELF:
                x["kv"] = x["q"]
            k = T.hybrid_attention(shape, T.torch_attention(shape, 40), x, spec=C.SPECS[spec_name])
            k.train(); tr = T.call(k, x); k.eval(); ev = T.call(k, x)
            print(f"{shape} {spec_name} x{scale}: attention train-eval {float((tr-ev).abs().max()):.4g} out std {float(ev.std()):.3g}")
