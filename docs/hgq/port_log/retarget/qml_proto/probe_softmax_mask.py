"""Why a masked QSoftmax costs ~1e11 EBOPs: the -1e9 fill reaches the exp-table input quantizer."""
import numpy as np
from hgq.layers import QSoftmax
from hepattn.keras_native.factory import LayerFactory

f = LayerFactory({"weight": {"default_q_type": "kbi", "b0": 8, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
                  "table": {"default_q_type": "kif", "i0": 2, "f0": 10}, "ebops": {"beta0": 1.0}})
rng = np.random.default_rng(0)
x = rng.standard_normal((2, 2, 12, 16)).astype("float32")
mask = rng.random((2, 1, 12, 16)) < 0.5
for label, m in (("no mask", None), ("mask", mask)):
    with f.scopes():
        sm = QSoftmax(axis=-1, name="sm")
    sm.build(x.shape)
    sm(x, training=True, mask=m)
    k, i, f_ = (np.asarray(t) for t in sm.exp_table.iq.quantizer.kif)
    print(f"{label:8s}: exp-table input quantizer i bits max {i.max():.0f}, bits max {np.asarray(sm.exp_table.iq.bits).max():.0f}; "
          f"exp_table ebops (float loss) {float(sum(float(l) for l in sm.exp_table.losses)):.4e}, uint32 var {int(sm.exp_table._ebops.value)}")
