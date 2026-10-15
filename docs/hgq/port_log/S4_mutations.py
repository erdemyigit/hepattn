"""S4 mutation runner: apply one mutation at a time, run the named tests, record the result, restore.

Usage (from anywhere): ``python docs/hgq/port_log/S4_mutations.py [M1 R1 ...]``. The runner copies the
tracked files of this worktree (their working-tree contents) into a private temporary directory and
mutates only that copy, so it never touches the shared lane worktree (other agents may edit it
concurrently). With no arguments it runs all mutations and writes ``S4_mutations.json`` next to this file.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = str(ROOT.parent / "hepattn-keras-jax/.venv-jax/bin/python")
GIT = "/usr/bin/git"
LOSSES = "src/hepattn/keras_native/losses.py"
MATCH = "src/hepattn/keras_native/matching.py"
OBJ = "src/hepattn/keras_native/objective.py"
TP = "tests/keras_native/test_objective_parity.py"
TM = "tests/keras_native/test_matching.py"
TE = "tests/keras_native/test_objective_e2e.py"
# R* were found by the S4 reviewers; the S4 fixes added the tests that kill them.
M = [
    (
        "M1 mask_bce/dice per-event (diagonal b==q only, mean over Nv)",
        LOSSES,
        "    total = ops.sum(ops.where(valid, term, 0.0))\n    return total / ops.maximum(batch * _count(object_valid_mask), 1.0)",
        "    term = term * ops.eye(batch)[:, :, None]\n"
        "    total = ops.sum(ops.where(valid, term, 0.0))\n"
        "    return total / ops.maximum(_count(object_valid_mask), 1.0)",
        [TP, TE],
    ),
    (
        "M2 naive BCE -[t log s + (1-t) log(1-s)]",
        LOSSES,
        "    return (1.0 - targets) * logits + ops.softplus(-logits)",
        "    s = ops.sigmoid(logits)\n    return -(targets * ops.log(s) + (1.0 - targets) * ops.log(1.0 - s))",
        [TP, TE],
    ),
    (
        "M3 object_ce_cost without the second softmax",
        LOSSES,
        "    probs = ops.softmax(f32(class_probs), axis=-1)",
        "    probs = f32(class_probs)",
        [TP, TE],
    ),
    ("M4 weighted CE normalised as mean(w*l)", LOSSES, "    return ops.sum(w * nll) / ops.sum(w)", "    return ops.mean(w * nll)", [TP, TE]),
    (
        "M5 kl_div_cost divided by T instead of N",
        LOSSES,
        '    return -ops.einsum("bpn,btn->bpt", log_pred, f32(true)) / pred.shape[-1]',
        '    return -ops.einsum("bpn,btn->bpt", log_pred, f32(true)) / true.shape[1]',
        [TP, TE],
    ),
    (
        "M6 regression l1 normalised per object, not per object*field",
        LOSSES,
        "    return ops.sum(ops.where(mask, diff, 0.0)) / ops.maximum(_count(mask), 1.0)\n",
        "    return ops.sum(ops.where(mask, diff, 0.0)) / ops.maximum(_count(object_valid_mask), 1.0)\n",
        [TP, TE],
    ),
    ("M7 dice target sum masked by the node pad", LOSSES, "+ ops.sum(targets, axis=-1)[None]", '+ ops.einsum("qpn,bn->bqp", targets, pad)', [TP, TE]),
    ("M8 matcher solves the untransposed cost", MATCH, "    costs_t = costs.swapaxes(1, 2)", "    costs_t = costs", [TM, TE]),
    (
        "M9 unmatched tail in descending order",
        MATCH,
        "np.arange(num_pred, dtype=np.int32)[remaining]",
        "np.arange(num_pred, dtype=np.int32)[remaining][::-1]",
        [TM],
    ),
    (
        "M10 permute with the inverse permutation",
        MATCH,
        '    idx = ops.cast(idx, "int32")\n',
        '    idx = ops.argsort(ops.cast(idx, "int32"), axis=1)\n',
        [TM, TE],
    ),
    (
        "M11 one host call per layer instead of one per step",
        OBJ,
        "    idx = match(stacked, valid, n_jobs=n_jobs, host_fn=host_fn)",
        "    idx = ops.concatenate([match(costs[layer], target_valid, n_jobs=n_jobs, host_fn=host_fn) for layer in layers], axis=0)",
        [TM],
    ),
    (
        "M12 no stop_gradient on the costs",
        MATCH,
        '    costs = jax.lax.stop_gradient(ops.cast(costs, "float32"))',
        '    costs = ops.cast(costs, "float32")',
        [TM],
    ),
    ("M13 nan_to_num removed", OBJ, "    return ops.nan_to_num(x, nan=0.0)", "    return x", [TM]),
    (
        "M14 mask task cost skipped",
        OBJ,
        "        return {name: w * L.mask_dice_cost(logits, target, pad) for name, w in self.costs.items()}",
        "        return {name: 0.0 * w * L.mask_dice_cost(logits, target, pad) for name, w in self.costs.items()}",
        [TP, TE],
    ),
    (
        "M15 permute_outputs skips the regression outputs",
        OBJ,
        "            if not spec.permute_loss or spec.name not in layer_out:",
        '            if not spec.permute_loss or spec.name not in layer_out or spec.name == "regression":',
        [TM, TE],
    ),
    ("R1 total_loss drops the first leaf", OBJ, "    return sum(leaves[1:], leaves[0])", "    return sum(leaves[1:], 0.0 * leaves[0])", [TP, TE]),
    ("R4 mask_bce ignores sample_weight", LOSSES, "        loss *= f32(sample_weight)", "        loss = loss", [TP, TE]),
    (
        "R14 kl_div_loss without eps",
        LOSSES,
        "    loss = -f32(true) * ops.log(f32(pred) + eps)",
        "    loss = -f32(true) * ops.log(f32(pred))",
        [TP, TM, TE],
    ),
    (
        "R14c kl_div_cost without eps",
        LOSSES,
        "    log_pred = ops.log(f32(pred) + eps)",
        "    log_pred = ops.log(f32(pred))",
        [TP, TE],
    ),
    (
        "R15 mask_dice_cost without the pad multiply",
        LOSSES,
        "    probs = ops.sigmoid(f32(logits)) * ops.expand_dims(f32(input_pad_mask), 1)",
        "    probs = ops.sigmoid(f32(logits))",
        [TP, TE],
    ),
    ("R17 encoder outputs ignored again", OBJ, "    if encoder:\n", "    if False:\n", [TP]),
    (
        "R18 matcher n_jobs ignored",
        OBJ,
        '    return int(args.get("n_jobs", 8)) if args.get("parallel_solver", False) else 1',
        "    return 1",
        [TP],
    ),
]


def _private_copy() -> Path:
    files = subprocess.run([GIT, "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\0")
    dst = Path(tempfile.mkdtemp(prefix="s4_mutations_"))
    for rel in filter(None, files):
        src = ROOT / rel
        if src.is_file():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / rel)
    return dst


def main(selected: list[str]) -> None:
    results = []
    copy = _private_copy()
    print(f"private copy: {copy}", flush=True)
    try:
        for name, path, old, new, tests in M:
            if selected and name.split()[0] not in selected:
                continue
            f = copy / path
            src = f.read_text()
            assert src.count(old) == 1, (name, src.count(old))
            f.write_text(src.replace(old, new))
            try:
                env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(copy / "src"), "HOME": os.environ["HOME"]}
                cmd = [PY, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-W", "ignore", *tests]
                r = subprocess.run(cmd, cwd=copy, env=env, capture_output=True, text=True, check=False)
                lines = r.stdout.splitlines()
                failed = [line.split(" - ")[0].replace("FAILED ", "") for line in lines if line.startswith(("FAILED", "ERROR"))]
                summary = lines[-1] if lines else r.stderr[-300:]
            finally:
                f.write_text(src)
            results.append({"mutation": name, "summary": summary, "failed": failed})
            print(json.dumps(results[-1]), flush=True)
    finally:
        shutil.rmtree(copy, ignore_errors=True)
    if not selected:
        Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
