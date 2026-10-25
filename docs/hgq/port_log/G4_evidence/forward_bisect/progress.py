# ruff: noqa  (one-off diagnostic, kept as run for docs/hgq/port_log/G4.md "Forward check")
import collections
import sys

import numpy as np

h, n = np.load(sys.argv[1]), np.load(sys.argv[2])


def index(d):
    out = collections.OrderedDict()
    seen = collections.Counter()
    for k in d.files:
        i, path, cls, slot = k.split("|")
        path = path.rsplit("/", 1)[-1]
        seen[path, slot] += 1
        out[path, slot, seen[path, slot]] = (int(i), cls, k)
    return out


hi, ni = index(h), index(n)
rows = []
for k in sorted((k for k in hi if k in ni and k[1] == "out0"), key=lambda k: hi[k][0]):
    a, b = h[hi[k][2]], n[ni[k][2]]
    if a.shape != b.shape or a.dtype == bool or not a.size:
        continue
    d = np.abs(a.astype(np.float64) - b)
    sc = max(float(np.abs(a).max()), 1e-12)
    rows.append((hi[k][0], k[0], hi[k][1], float(d.max()) / sc, float((d > 0.01 * sc).mean())))
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
filt = sys.argv[4] if len(sys.argv) > 4 else ""
for r in rows:
    if filt and filt not in r[1]:
        continue
    if r[0] % step == 0 or r[3] > 0.05:
        print(f"#{r[0]:5d} {r[1]:55s} {r[2]:20s} maxrel {r[3]:.2e} frac>1% {r[4]:.4f}")
