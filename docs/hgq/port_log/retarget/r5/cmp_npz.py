"""Array-by-array comparison of two npz files: same keys, dtypes, shapes and bytes (bit-identical content)."""
import sys
import numpy as np

def cmp(a, b):
    with np.load(a) as x, np.load(b) as y:
        kx, ky = set(x.files), set(y.files)
        if kx != ky:
            return f"keys differ: only-a {sorted(kx-ky)[:5]} only-b {sorted(ky-kx)[:5]}"
        bad = [k for k in sorted(kx) if x[k].dtype != y[k].dtype or x[k].shape != y[k].shape or x[k].tobytes() != y[k].tobytes()]
        return f"IDENTICAL ({len(kx)} arrays)" if not bad else f"{len(bad)} arrays differ, e.g. {bad[:5]}"

for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    print(a.rsplit('/',1)[-1], cmp(a, b))
