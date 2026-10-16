"""R1 mutation check: apply each mutant to a scratch copy of a commit and list the R1-owned tests that fail.

Usage (from the repository root, with the JAX venv):

    KERAS_BACKEND=jax .venv-jax/bin/python docs/hgq/port_log/R1_mutants.py [--rev HEAD] [--out DIR] [NAME ...]

Each mutant is applied to ``git archive <rev>`` in its own directory under ``--out`` (default: a new
temporary directory), never to the working tree. ``m00_none`` is the unmutated control and must give 0
failures. Results are in ``docs/hgq/port_log/R1.md``.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET  # noqa: S405  (parses our own pytest junit output)
from concurrent.futures import ThreadPoolExecutor

LOSSES = "src/hepattn/keras_native/losses.py"
OBJECTIVE = "src/hepattn/keras_native/objective.py"
MATCHING = "src/hepattn/keras_native/matching.py"
TESTS = [
    "tests/keras_native/test_org_loss_fixes.py",
    "tests/keras_native/test_objective_parity.py",
    "tests/keras_native/test_objective_e2e.py",
    "tests/keras_native/test_matching.py",
]
PRE_R1 = "5db27a8"  # R0 head: native losses with the cross-event broadcast

# name -> [(file, old, new)]; old must occur exactly once; old=None replaces the file with new.
MUTANTS = {
    "m00_none": [],
    "m01_old_losses_module": [(LOSSES, None, f"git:{PRE_R1}:{LOSSES}")],
    "m02_bce_global_count": [
        (
            LOSSES,
            "counts = ops.maximum(ops.sum(pad, axis=-1, keepdims=True), 1.0)  # [B, 1]",
            "counts = ops.maximum(ops.sum(pad), 1.0) / pad.shape[0]",
        )
    ],
    "m03_dice_unmasked_probs": [
        (
            LOSSES,
            "probs = ops.sigmoid(f32(logits)) * ops.expand_dims(f32(input_pad_mask), 1)\n    targets = f32(targets)\n    numerator = 2.0 * ops.sum(",
            "probs = ops.sigmoid(f32(logits))\n    targets = f32(targets)\n    numerator = 2.0 * ops.sum(",
        )
    ],
    "m04_mean_over_all_objects": [(LOSSES, "return ops.sum(per_object * weight) / ops.maximum(ops.sum(weight), 1.0)", "return ops.mean(per_object)")],
    "m05_no_clamp_min": [
        (LOSSES, "return ops.sum(per_object * weight) / ops.maximum(ops.sum(weight), 1.0)", "return ops.sum(per_object * weight) / ops.sum(weight)")
    ],
    "m06_bce_pad_not_applied": [
        (LOSSES, "per_object = ops.sum(loss * ops.expand_dims(pad, 1), axis=-1) / counts", "per_object = ops.sum(loss, axis=-1) / counts")
    ],
    "m07_class_cost_from_probs": [
        (
            OBJECTIVE,
            'not *_class_prob\n        logits = outputs[self.output_object + "_logit"]',
            'not *_class_prob\n        logits = outputs[self.output_object + "_class_prob"]',
        )
    ],
    "m08_no_nan_to_num": [
        (
            MATCHING,
            "costs = np.nan_to_num(np.asarray(costs, dtype=np.float32), nan=_BIG, posinf=_BIG, neginf=-_BIG)",
            "costs = np.asarray(costs, dtype=np.float32)",
        )
    ],
    "m09_neginf_to_plus_big": [(MATCHING, "neginf=-_BIG)", "neginf=_BIG)")],
    "m10_nan_to_minus_big": [(MATCHING, "nan=_BIG, posinf", "nan=-_BIG, posinf")],
    "m11_lap1015_late_rejected": [(OBJECTIVE, '_HOST_SOLVERS = {"scipy", "lap1015_late"}', '_HOST_SOLVERS = {"scipy"}')],
    "m12_no_fallback_log": [(OBJECTIVE, '    if solver == "lap1015_late":\n        _log_lap1015_fallback()\n', "")],
    "m13_device_solver_accepted": [(OBJECTIVE, '    if args.get("device_solver") is not None:\n', "    if False:\n")],
    "m14_any_solver_accepted": [(OBJECTIVE, "if solver not in _HOST_SOLVERS or args.get(", "if args.get(")],
    "m15_log_every_call": [(OBJECTIVE, "@cache\ndef _log_lap1015_fallback", "def _log_lap1015_fallback")],
    # fix round 1 (reviewer mutants that survived a37b04a)
    "m16_bce_count_unclamped": [
        (
            LOSSES,
            "counts = ops.maximum(ops.sum(pad, axis=-1, keepdims=True), 1.0)  # [B, 1]",
            "counts = ops.sum(pad, axis=-1, keepdims=True)  # [B, 1]",
        )
    ],
    "m17_dice_numerator_unmasked": [
        (LOSSES, "numerator = 2.0 * ops.sum(probs * targets, axis=-1)", "numerator = 2.0 * ops.sum(ops.sigmoid(f32(logits)) * targets, axis=-1)")
    ],
    "m18_posinf_to_minus_big": [(MATCHING, "posinf=_BIG", "posinf=-_BIG")],
    "m19_scipy_rejected": [(OBJECTIVE, '_HOST_SOLVERS = {"scipy", "lap1015_late"}', '_HOST_SOLVERS = {"lap1015_late"}')],
    # fix round 2 (reviewer mutant r08 / C_big_f32max that survived e812fa9; equivalent for scipy)
    "m20_big_is_f32max": [(MATCHING, "_BIG = float(np.finfo(np.float32).max / 10)", "_BIG = float(np.finfo(np.float32).max)")],
}


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout  # noqa: S607  (git from PATH)


def run(name: str, rev: str, out: pathlib.Path) -> tuple[str, int, list[str]]:
    root = out / name
    root.mkdir(parents=True)
    archive = _git("archive", rev, "src", "tests", "pyproject.toml")
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)  # noqa: S607  (tar from PATH)
    for rel, old, new in MUTANTS[name]:
        path = root / rel
        if old is None:
            _, src_rev, src_path = new.split(":", 2)
            path.write_bytes(_git("show", f"{src_rev}:{src_path}"))
            continue
        text = path.read_text()
        assert text.count(old) == 1, (name, rel, text.count(old))
        path.write_text(text.replace(old, new))
    xml = out / f"{name}.xml"
    env = {**os.environ, "KERAS_BACKEND": "jax", "PYTHONPATH": str(root / "src")}
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-W", "ignore", f"--junitxml={xml}", *TESTS]
    subprocess.run(cmd, cwd=root, env=env, capture_output=True, check=False)
    cases = list(ET.parse(xml).getroot().iter("testcase"))  # noqa: S314  (our own pytest output)
    failed = [
        f"{c.get('classname').rsplit('.', 1)[-1]}::{c.get('name')}" for c in cases if c.find("failure") is not None or c.find("error") is not None
    ]
    return name, len(cases), failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(MUTANTS))
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    out = args.out or pathlib.Path(tempfile.mkdtemp(prefix="r1_mutants_"))
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda n: run(n, args.rev, out), args.names))
    for name, total, failed in results:
        print(f"{name}: {len(failed)}/{total} failed")
        for f in failed:
            print(f"    {f}")
    survivors = [n for n, _, f in results if n != "m00_none" and not f]
    control_failed = any(n == "m00_none" and f for n, _, f in results)
    print(f"survivors: {survivors}; control failures: {control_failed}; scratch trees in {out}")
    sys.exit(1 if survivors or control_failed else 0)


if __name__ == "__main__":
    main()
