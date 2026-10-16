"""Native HGQ2 penalty on the generated Polaris org model at beta0 1e-12 (as polaris/15_g2_forward.py), CLIC dummy events.

Compare with the hybrid's deduplicated ``quant_losses()`` on the same events (tests/keras/test_quant_losses_dedupe.py:
3.13701, old double count 3.41763). Run from the worktree root in the jax env, PYTHONPATH=src:
    python docs/hgq/port_log/S5_native_penalty.py <polaris_hgq.yaml> <batches.npz from S5_make_batches.py> <scale yaml>
"""
# ruff: noqa: PLC2701

import json
import sys

import jax
import numpy as np
import yaml

from hepattn.keras_native.ebops import EbopsLeaves
from hepattn.keras_native.maskformer import _modules, from_config

cfg = yaml.safe_load(open(sys.argv[1]))  # noqa: PTH123, SIM115
init = cfg["model"]["model"]["init_args"]
init["quant"] = {**init["quant"], "ebops": {"beta0": 1.0e-12}}
for task in _modules(init["tasks"]):
    if "scale_dict_path" in task.get("init_args", {}):
        task["init_args"]["scale_dict_path"] = sys.argv[3]
data = np.load(sys.argv[2])
out = []
for seed in (0, 1):
    import keras  # noqa: PLC0415

    keras.utils.set_random_seed(seed)
    model = from_config(cfg["model"])
    for i in range(int(data["n"])):
        inputs = {k.split("::", 2)[2]: data[k] for k in data.files if k.startswith(f"{i}::in::")}
        inputs = {k: v.astype(np.int32) if v.dtype == np.int64 else v for k, v in inputs.items()}
        if not model.built:
            model(inputs, training=False)
            el = EbopsLeaves(model)
            fwd = jax.jit(lambda tv, ntv, x: model.stateless_call(tv, ntv, x, training=True, return_losses=True)[1:])  # noqa: B023
        tv = [v.value for v in model.trainable_variables]
        ntv = [v.value for v in model.non_trainable_variables]
        _, losses = fwd(tv, ntv, inputs)
        pen, reg = el.split_losses(losses)
        total = float(sum(float(x) for x in losses))
        row = {"seed": seed, "batch": i, "penalty_total": total, "ebops": float(sum(float(x) for x in pen)) / 1e-12, "reg": float(sum(float(x) for x in reg))}
        out.append(row)
        print(json.dumps(row), flush=True)
