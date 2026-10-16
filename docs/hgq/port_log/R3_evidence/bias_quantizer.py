import sys

sys.path.insert(0, ".")
import numpy as np
from tests.keras_native.test_pr8_layers import QUANT_SPECS, swiglu_input, swiglu_pair

t, n = swiglu_pair("ffn_64", QUANT_SPECS["high"])
l = n.hidden[0]
for q in ("_iq", "_kq", "_bq"):
    c = getattr(l, q).config
    print(q, getattr(l, q).quantizer.__class__.__name__, c.config if hasattr(c, "config") else vars(c))
x = swiglu_input("ffn_64")
k = np.asarray(l.kernel)
qk = np.asarray(l.qkernel)
print("kernel err", np.abs(k - qk).max())
b = np.asarray(l.bias)
qb = np.asarray(l.qbias)
print("bias err", np.abs(b - qb).max(), b[:4], qb[:4])
xi = np.asarray(l.iq(x))
print("input err", np.abs(xi - x).max())
