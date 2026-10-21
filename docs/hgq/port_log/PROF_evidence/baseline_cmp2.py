"""Run dirs vs s7: every npz array bitwise (state); __meta__ JSON equal after dropping wall-clock keys; CSVs w/o time/."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
ref = root / "s7"


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not (k.startswith("time/") or k.endswith("_s") or k in {"saved_at", "wall"})}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def rows(d):
    out = {}
    for p in sorted(d.rglob("*.csv")):
        with p.open(newline="") as f:
            out[p.relative_to(d).as_posix()] = [{k: v for k, v in r.items() if not k.startswith("time/")} for r in csv.DictReader(f)]
    return out


R = sorted(p.relative_to(ref) for p in ref.rglob("*.npz"))
C = rows(ref)
print("ref:", len(R), "npz files", {k: len(v) for k, v in C.items()}, "csv rows")
for name in sys.argv[2:]:
    d = root / name
    N = sorted(p.relative_to(d) for p in d.rglob("*.npz"))
    ok, narr, metadiff = N == R, 0, set()
    for rel in R:
        a, b = np.load(ref / rel), np.load(d / rel)
        ok &= set(a.files) == set(b.files)
        for k in a.files:
            if k == "__meta__":
                ma, mb = json.loads(str(a[k])), json.loads(str(b[k]))
                if strip(ma) != strip(mb):
                    ok = False
                    metadiff |= {x for x in set(ma) | set(mb) if strip(ma.get(x)) != strip(mb.get(x))}
                continue
            narr += 1
            ok &= a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes()
    csv_ok = rows(d) == C
    verdict = "EQUAL" if ok and csv_ok else "DIFFERENT"
    print(
        f"{name}: {len(N)} npz, {narr} state arrays bitwise-compared, meta (no wall-clock) diffs {sorted(metadiff)}, csv equal {csv_ok} -> {verdict}"
    )
