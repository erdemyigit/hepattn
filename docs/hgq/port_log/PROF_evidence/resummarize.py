"""Re-derive the summary numbers (loop window included) from a recorded ``phases.csv`` and its ``.json`` (wall, mode).

PYTHONPATH=src KERAS_BACKEND=jax python docs/hgq/port_log/PROF_evidence/resummarize.py <run_dir> [<run_dir> ...]
"""

import csv
import json
import sys
from pathlib import Path

from hepattn.keras_native import profile

KEYS = (
    "wall_s",
    "steady_wall_s",
    "outside_loop_s",
    "loop_wall_s",
    "s_per_micro_batch_steady",
    "s_per_micro_batch_loop",
    "device_idle_frac",
    "device_idle_frac_loop",
)

for d in map(Path, sys.argv[1:]):
    old = json.loads((d / "phases.csv.json").read_text())
    p = profile.PhaseProfiler(sync=old["mode"] == "sync")
    with (d / "phases.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            floats = (float(r[k]) for k in ("start_s", "dur_s", "excl_s"))
            p.events.append((int(r["seq"]), r["path"], *floats, int(r["micro_step"]), int(r["global_step"]), int(r["epoch"])))
    p.t0, p.t1 = 0.0, old["wall_s"]
    s = p.summary()
    assert abs(s["steady_wall_s"] - old["steady_wall_s"]) < 1e-9  # the recorded run's own steady window
    print(d.name, {k: round(s[k], 4) for k in KEYS if k in s})
