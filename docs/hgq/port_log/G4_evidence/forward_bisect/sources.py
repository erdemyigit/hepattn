# ruff: noqa  (one-off diagnostic, kept as run for docs/hgq/port_log/G4.md "Forward check")
import collections
import sys

import numpy as np

h, n = np.load(sys.argv[1]), np.load(sys.argv[2])


def index(d):
    out = collections.defaultdict(dict)
    seen = collections.Counter()
    for k in d.files:
        i, path, cls, slot = k.split("|")
        path = path.rsplit("/", 1)[-1]
        out[(int(i))][slot] = (path, cls, k)
    return out


def by_name(d):
    res = collections.OrderedDict()
    seen = collections.Counter()
    for i, slots in sorted(index(d).items()):
        path, cls, _ = next(iter(slots.values()))
        seen[path] += 1
        res[path, seen[path]] = (i, cls, {s: v[2] for s, v in slots.items()})
    return res


H, N = by_name(h), by_name(n)
lim = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
for key, (i, cls, hs) in H.items():
    if key not in N:
        continue
    ns = N[key][2]

    def cnt(slot):
        a, b = h[hs[slot]], n[ns[slot]]
        if a.shape != b.shape:
            return -1, a.size
        if a.dtype == bool:
            return int((a != b).sum()), a.size
        m = np.abs(a) < 1e7
        sc = float(np.percentile(np.abs(a[m]), 99)) if m.any() else 1.0
        return int(((np.abs(a.astype(np.float64) - b) > lim * max(1e-3, sc)) & m).sum() + (a[~m] != b[~m]).sum()), a.size

    ins = [cnt(s) for s in hs if s.startswith("in") and s in ns]
    outs = [cnt(s) for s in hs if s.startswith("out") and s in ns]
    if not outs:
        continue
    in_bad = sum(c for c, _ in ins)
    out_bad, out_n = outs[0]
    if in_bad == 0 and out_bad > 0:
        print(f"SOURCE #{i} {key[0]} ({cls}) inputs equal, out0 differs in {out_bad}/{out_n}")
