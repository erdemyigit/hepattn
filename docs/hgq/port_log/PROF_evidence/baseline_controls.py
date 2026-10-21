"""Controls of baseline_cmp2.py: copies of the ``off`` run dir with one known change each (must compare DIFFERENT).

    cd <runs> && cp -R off ctl_bit && cp -R off ctl_meta && python baseline_controls.py

ctl_bit: the lowest mantissa bit of one element of one float32 ``tv/`` array in the last step checkpoint is flipped.
ctl_meta: a numeric ``__meta__`` field (the first one: ``beta``) of the first checkpoint is increased by 1.
"""

import json
from pathlib import Path

import numpy as np


def rewrite(path: Path, change) -> None:
    arrays = dict(np.load(path))
    change(arrays)
    np.savez(path, **arrays)


def flip(arrays: dict) -> None:
    key = next(k for k in arrays if k.startswith("tv/") and arrays[k].dtype == np.float32 and arrays[k].size)
    a = arrays[key].copy()
    a.view(np.uint32).reshape(-1)[0] ^= 1
    arrays[key] = a
    print("ctl_bit:", key)


def bump_meta(arrays: dict) -> None:
    meta = json.loads(str(arrays["__meta__"]))
    key = next(k for k, v in meta.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    meta[key] += 1
    arrays["__meta__"] = np.array(json.dumps(meta))
    print("ctl_meta:", key)


rewrite(max(Path("ctl_bit").rglob("*.npz")), flip)
rewrite(min(Path("ctl_meta").rglob("*.npz")), bump_meta)
