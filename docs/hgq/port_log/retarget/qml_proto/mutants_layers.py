"""Non-vacuity of run_quant.core_parity: each deliberate bug must make the float-leaf parity FAIL."""
import sys

import run_quant as rq
import qml_layers as ql
from keras import ops

orig_rescale, orig_energy = ql.rescale, ql.SeqProjection.energy


def no_renorm(kept, lut, mask, energy, training=None):
    return ops.ones_like(orig_rescale(kept, lut, mask, energy, training))


def no_tail(self, training=None):  # renorm numerator over E[:n] only, not all seq_len rows
    head, _ = self.rows(training)
    return ops.square(head) / ops.sum(ops.square(head), 0)


for name, patch in (("renorm dropped", lambda: setattr(ql, "rescale", no_renorm)),
                    ("renorm numerator without tail rows", lambda: setattr(ql.SeqProjection, "energy", no_tail))):
    ql.rescale, ql.SeqProjection.energy = orig_rescale, orig_energy
    patch()
    print(f"== mutant: {name}")
    print("   core_parity(float) ->", "PASS (mutant NOT caught)" if rq.core_parity(None, 2e-5) else "FAIL (mutant caught)")
