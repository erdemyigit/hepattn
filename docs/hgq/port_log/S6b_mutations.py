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
CLI = "src/hepattn/keras_native/cli.py"
T = "tests/keras_native/test_loop.py"
TC = "tests/keras_native/test_cli.py"


def tests(*names: str) -> list[str]:
    return [T, "-k", " or ".join(names)]


def cli_tests(*names: str) -> list[str]:
    return [TC, "-k", " or ".join(names)]


CADENCE = "log_cadence"
SAME_DIR = "same_run_directory"


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
    # fix round (reviewer mutants R1/R6, R11, R2, R3, R5, R8, R9 and the fixes of this round)
    "K0b_none_fix_round": ([], tests("val_metrics", CADENCE, SAME_DIR, "csv_log", "refuses_another", RESUME)),
    "K13_val_metrics_count_padding": (
        [
            (
                LOOP,
                '            events += n\n            metrics.update(res["preds"], batch.targets, event_valid=batch.event_valid)\n',
                '            events += n\n            metrics.update(res["preds"], batch.targets, event_valid=None)\n',
            )
        ],
        tests("val_metrics"),
    ),
    "K14_val_metrics_per_layer": (
        [(LOOP, 'ClicMetrics("val", pool_layers=self.s.pool_layers)', 'ClicMetrics("val", pool_layers=False)')],
        tests("val_metrics"),
    ),
    "K15_log_cadence_off_by_one": (
        [(LOOP, "if self.global_step % self.s.log_every_n_steps == 0:", "if (self.global_step - 1) % self.s.log_every_n_steps == 0:")],
        tests(CADENCE),
    ),
    "K16_lr_of_the_next_update": (
        [(LOOP, "float(self.schedule((self.global_step - 1) * self.k))", "float(self.schedule(self.global_step * self.k))")],
        tests(CADENCE),
    ),
    "K17_bits_trainable_only": (
        [(LOOP, "[i for i, v in enumerate(model.non_trainable_variables) if last(v) == s]", "[]")],
        tests(CADENCE),
    ),
    "K18_check_val_interval_ignored": (
        [(LOOP, "if val is not None and (epoch + 1) % self.s.check_val_every_n_epoch == 0:", "if val is not None:")],
        tests(CADENCE),
    ),
    "K19_no_matmul_precision": (
        [(LOOP, "matmul_precision=settings.matmul_precision, donate", "matmul_precision=None, donate")],
        tests(CADENCE),
    ),
    "K20_logs_not_restored": (
        [
            (
                LOOP,
                '        self.epoch_logs = dict(meta.get("epoch_logs") or {})\n        self.train_logs = dict(meta.get("train_logs") or {})\n',
                "",
            )
        ],
        tests(RESUME),
    ),
    "K21_csv_rows_not_dropped_on_resume": (
        [(LOOP, "        if self.resumed_from is not None:  # same run directory", "        if False:  # same run directory")],
        tests(SAME_DIR),
    ),
    "K22_stale_temporaries_kept": ([(LOOP, "            stale.unlink(missing_ok=True)\n", "            pass\n")], tests(SAME_DIR)),
    "K23_seed_not_checked": ([(LOOP, '*(("seed",) if check_seed else ())', "*()")], tests("refuses_another")),
    "K24_csv_header_fixed_at_first_row": (
        [(LOOP, "        elif new:\n            self._rewrite(", "        elif False:\n            self._rewrite(")],
        tests("csv_log", RESUME),
    ),
    "K25_saved_config_without_overrides": (
        [(CLI, "yaml.safe_dump(effective_raw(rc, args), sort_keys=False)", "yaml.safe_dump(rc.raw, sort_keys=False)")],
        cli_tests("saved_config"),
    ),
    "K26_run_json_overwritten_on_resume": (
        [(CLI, '_revision(run_dir / "run.json")', 'run_dir / "run.json"')],
        cli_tests("keeps_the_first_run_files"),
    ),
    # second fix round (review of 8380d8a)
    "K0c_none_fix_round2": ([], tests(CADENCE, "max_steps", "limit_val", "seed_sets")),
    "K27_epoch_checkpoint_without_validation": (
        [(LOOP, "(validated or self.s.check_val_every_n_epoch == 1)", "True")],
        tests(CADENCE),
    ),
    "K28_bits_min_is_max": (
        [
            (
                LOOP,
                "jnp.min(jnp.stack([jnp.min(a.astype(jnp.float32)) for a in arrays]))",
                "jnp.max(jnp.stack([jnp.max(a.astype(jnp.float32)) for a in arrays]))",
            )
        ],
        tests(CADENCE),
    ),
    "K29_max_steps_ignored": ([(LOOP, "        if settings.max_steps > 0:\n", "        if False:\n")], tests("max_steps")),
    "K30_no_stop_after_max_steps_inside_an_epoch": (
        [(LOOP, "            if not ended:  # max_steps inside an epoch: Lightning runs no validation there\n                break\n", "")],
        tests("max_steps"),
        900,  # the mutant loops forever: a timeout is a kill
    ),
    "K31_limit_val_batches_ignored": (
        [(LOOP, "            if self.s.limit_val_batches is not None and i >= self.s.limit_val_batches:\n", "            if False:\n")],
        tests("limit_val"),
    ),
    "K32_no_set_random_seed": (
        [(LOOP, "        keras.utils.set_random_seed(settings.seed)", "        pass")],
        tests("seed_sets"),
    ),
    "K33_source_seed_fixed": (
        [("src/hepattn/experiments/clic/main_jax.py", "dataset=self.dataset, seed=seed)", "dataset=self.dataset, seed=42)")],
        tests("seed_sets"),
    ),
    "K34_cli_data_seed_fixed": (
        [(CLI, 'train = data_factory(rc, "train", seed=settings.seed', 'train = data_factory(rc, "train", seed=42')],
        cli_tests("reach_the_data"),
    ),
    "K35_cli_limit_val_not_passed": (
        [(CLI, "        limit_val_batches=args.limit_val_batches,\n", "        limit_val_batches=None,\n")],
        cli_tests("reach_the_data"),
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
        edits, args, *limit = MUTANTS[name]
        copy = Path(scratch) / name
        shutil.rmtree(copy, ignore_errors=True)
        for part in ("src", "tests"):
            shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
        shutil.copy(ROOT / "pyproject.toml", copy / "pyproject.toml")
        apply(copy, edits)
        env = {**os.environ, "KERAS_BACKEND": "jax", "PYTHONPATH": str(copy / "src")}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", "-rf"],
                cwd=copy,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=limit[0] if limit else None,
            )
        except subprocess.TimeoutExpired:
            print(name, f"TIMEOUT after {limit[0]} s: killed (the run does not end)", sep="\n  ", flush=True)
            continue
        (copy / "pytest.out").write_text(proc.stdout + proc.stderr)
        summary = [line for line in proc.stdout.splitlines() if line.startswith(("FAILED", "ERROR")) or " passed" in line or " failed" in line]
        print(name, *summary, sep="\n  ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
