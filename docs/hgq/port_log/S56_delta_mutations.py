"""Mutation run for the S5 x S6a delta fixes (S56-merge.md, "Delta fix").

Each mutant is applied to a fresh copy of ``src`` and ``tests`` (never to the worktree) and runs its own tests there.
``*_oldtest`` mutants also put back test files from ``OLD_REV`` (the merge HEAD before the delta fix) to show that the
old tests let the mutant survive. Usage, from the worktree root::

    KERAS_BACKEND=jax .venv-jax/bin/python docs/hgq/port_log/S56_delta_mutations.py <scratch dir> [name ...]
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD_REV = "8de6720"
CONFIG = "src/hepattn/keras_native/config.py"
CFG_TESTS = ["tests/keras_native/test_config.py", "-k", "overlay or merge or replaced_whole or lrs_config or short_class or explicit_class or posenc"]
OPTIM_TESTS = ["tests/keras_native/test_optim.py", "-k", "onecycle or learning_rate"]
GUARD_TESTS = ["tests/keras_native/test_train_step.py", "-k", "another_model"]
CK_TESTS = ["tests/keras_native/test_checkpoint.py", "-k", "resume_checkpoint or keep_last_k"]
SPIKE = "    return ops.where(n <= self.phase1_end, rising, falling)\n"
GUARD = "any(a is not b for a, b in zip(tracked, model.trainable_variables, strict=True))"
PROTECT = "        if self.resume_path is not None and path.resolve() == self.resume_path:\n            return\n"

# name: (edits [(file, old, new)], test args, test files restored from OLD_REV)
MUTANTS = {
    "D0_none": ([], CFG_TESTS, []),
    # keys next to class_path accepted (no check at all)
    "D1_short_form_accepted": ([(CONFIG, '        if "class_path" in node and stray:\n', "        if False:\n")], CFG_TESTS, []),
    # an overlay list item without class_path replaces the list (the loader before the fix)
    "D2_bare_list_item_accepted": ([(CONFIG, "    _check_list_overlay(base, overlay, key)\n", "")], CFG_TESTS, []),
    # the torch decoder's posenc merged as a class (review mutant R3, inverted)
    "D3_decoder_posenc_merged": (
        [
            (
                CONFIG,
                '_DICT_TYPED = {"KerasMaskFormer": {"encoder", "decoder"}, "MaskFormerDecoder": {"posenc"}}',
                '_DICT_TYPED = {"KerasMaskFormer": {"encoder", "decoder"}}',
            )
        ],
        CFG_TESTS,
        [],
    ),
    # model_description ignores the owning class: the decoder posenc is described as a FourierPositionEncoder
    "D4_canonical_without_parent": (
        [(CONFIG, "    args = {k: _canonical(v, k, class_name) for k, v", "    args = {k: _canonical(v, k) for k, v")],
        CFG_TESTS,
        [],
    ),
    # review mutant R7: the optimizer guard compares paths
    "D5_guard_by_path": (
        [("src/hepattn/keras_native/train_step.py", GUARD, "any(a.path != b.path for a, b in zip(tracked, model.trainable_variables, strict=True))")],
        GUARD_TESTS,
        [],
    ),
    "D5_guard_by_path_oldtest": (
        [("src/hepattn/keras_native/train_step.py", GUARD, "any(a.path != b.path for a, b in zip(tracked, model.trainable_variables, strict=True))")],
        GUARD_TESTS,
        ["tests/keras_native/test_train_step.py"],
    ),
    # the resume checkpoint is pruned like any other
    "D6_resume_not_protected": ([("src/hepattn/keras_native/checkpoint.py", PROTECT, "")], CK_TESTS, []),
    # the schedule is off by 2e-6 relative at one step that the old samples do not contain (128, the measured worst)
    "D7_lr_spike_at_128": (
        [("src/hepattn/keras_native/optim.py", SPIKE, SPIKE.rstrip("\n") + " * ops.where(n == 128.0, 1.0 + 2e-6, 1.0)\n")],
        OPTIM_TESTS,
        [],
    ),
    "D7_lr_spike_at_128_oldtest": (
        [("src/hepattn/keras_native/optim.py", SPIKE, SPIKE.rstrip("\n") + " * ops.where(n == 128.0, 1.0 + 2e-6, 1.0)\n")],
        OPTIM_TESTS,
        ["tests/keras_native/test_optim.py"],
    ),
}


def apply(copy: Path, edits) -> None:
    for rel, old, new in edits:
        path = copy / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: anchor found {text.count(old)} times")
        path.write_text(text.replace(old, new))


def main(scratch: str, names: list[str]) -> int:
    for name in names or list(MUTANTS):
        edits, tests, old_files = MUTANTS[name]
        copy = Path(scratch) / name
        shutil.rmtree(copy, ignore_errors=True)
        for part in ("src", "tests"):
            shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
        shutil.copy(ROOT / "pyproject.toml", copy / "pyproject.toml")
        for rel in old_files:
            old = subprocess.run([shutil.which("git"), "-C", str(ROOT), "show", f"{OLD_REV}:{rel}"], capture_output=True, text=True, check=True)
            (copy / rel).write_text(old.stdout)
        apply(copy, edits)
        env = {**os.environ, "KERAS_BACKEND": "jax", "PYTHONPATH": str(copy / "src")}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider", "-rf"],
            cwd=copy,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (copy / "pytest.out").write_text(proc.stdout + proc.stderr)
        summary = [line for line in proc.stdout.splitlines() if line.startswith(("FAILED", "ERROR")) or " passed" in line or " failed" in line]
        print(name, *summary, sep="\n  ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
