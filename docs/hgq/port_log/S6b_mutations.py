"""Mutation run for S6b (the driver ``keras_native.loop``): each mutant must make its named tests fail.

Each mutant is applied to a fresh copy of ``src`` and ``tests`` (never to the worktree) and runs its tests there.
Usage, from the worktree root::

    KERAS_BACKEND=jax .venv-jax/bin/python docs/hgq/port_log/S6b_mutations.py <scratch dir> [name ...]
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOOP = "src/hepattn/keras_native/loop.py"
OPTIM = "src/hepattn/keras_native/optim.py"
T = "tests/keras_native/test_loop.py"


def tests(*names: str) -> list[str]:
    return [T, "-k", " or ".join(names)]


RESUME = "continues_bit_for_bit"
MUTANTS = {
    "K0_none": ([], tests("non_vacuous", "periodic", "val_loss", "pid", "stepping_past", "optimizer_reader")),
    "K1_ebops_period_by_micro_step": (
        [
            (
                LOOP,
                "return self.has_ebops and self.global_step % self.ebops_every == 0",
                "return self.has_ebops and self.micro_step % self.ebops_every == 0",
            )
        ],
        tests("periodic"),
    ),
    "K2_val_loss_task_plus_regularizers": (
        [(LOOP, '"val/loss": task, ', '"val/loss": task + res["regularization"], ')],
        tests("val_loss"),
    ),
    "K3_no_tail_recompute": ([(LOOP, "            if n < self.batch_size:\n", "            if False:\n")], tests("val_loss")),
    "K4_val_weighted_by_batch_size": (
        [(LOOP, "                sums[key] += float(value) * n\n", "                sums[key] += float(value) * self.batch_size\n")],
        tests("val_loss"),
    ),
    "K5_pid_state_not_restored": ([(LOOP, '            self.pid.load_state_dict(meta["beta_controller"])\n', "            pass\n")], tests(RESUME)),
    "K6_data_position_off_by_one": (
        [
            (
                LOOP,
                '"data_position": {"epoch": self.epoch, "batch": self.batch},',
                '"data_position": {"epoch": self.epoch, "batch": max(0, self.batch - 1)},',
            )
        ],
        tests(RESUME),
    ),
    "K7_host_beta_from_float32_state": ([(LOOP, '            self.beta = float(meta["beta"])\n', "            self.beta = stored\n")], tests(RESUME)),
    "K8_no_total_steps_guard": (
        [(LOOP, "        if self.global_step >= self.total_steps:  # torch", "        if False:  # torch")],
        tests("stepping_past"),
    ),
    "K9_ramp_after_the_step": (
        [(LOOP, "ramp_beta(self.global_step, **self.ramp)", "ramp_beta(self.global_step + 1, **self.ramp)")],
        tests("beta_scheduler"),
    ),
    "K10_resume_checkpoint_not_protected": (
        [(LOOP, "cfg.step, resume_path=self.resumed_from)", "cfg.step, resume_path=None)")],
        tests("survives_pruning"),
    ),
    "K11_keras_default_epsilon": ([(LOOP, "        adam_epsilon=o.epsilon,\n", "        adam_epsilon=1e-7,\n")], tests("optimizer_reader")),
    "K12_null_clip_algorithm_rejected": (
        [(OPTIM, 'trainer_section.get("gradient_clip_algorithm") or "norm"', 'trainer_section.get("gradient_clip_algorithm", "norm")')],
        tests("optimizer_reader"),
    ),
}


def apply(copy: Path, edits) -> None:
    for rel, old, new in edits:
        path = copy / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: anchor found {text.count(old)} times: {old!r}")
        path.write_text(text.replace(old, new))


def main(scratch: str, names: list[str]) -> int:
    for name in names or list(MUTANTS):
        edits, args = MUTANTS[name]
        copy = Path(scratch) / name
        shutil.rmtree(copy, ignore_errors=True)
        for part in ("src", "tests"):
            shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
        shutil.copy(ROOT / "pyproject.toml", copy / "pyproject.toml")
        apply(copy, edits)
        env = {**os.environ, "KERAS_BACKEND": "jax", "PYTHONPATH": str(copy / "src")}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", "-rf"],
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
