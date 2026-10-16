"""S5 mutation runner: apply one mutation at a time to a private copy of the tracked files, run the named tests, restore.

Usage: ``python docs/hgq/port_log/S5_mutations.py [M1 M2 ...]``. The copy holds the working-tree contents of the files
``git ls-files`` lists (commit new files first), so the shared worktree is never mutated. ``jax`` targets run in
``hepattn-keras-jax/.venv-jax``; ``torch`` targets in the hybrid env ``hepattn/.venv`` with ``KERAS_BACKEND=torch``.
With no arguments every mutation runs and the results go to ``S5_mutations.out`` (JSON) next to this file.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = {"jax": str(ROOT.parent / "hepattn-keras-jax/.venv-jax/bin/python"), "torch": str(ROOT.parent / "hepattn/.venv/bin/python")}
GIT = "/usr/bin/git"
TS = "src/hepattn/keras_native/train_step.py"
OPT = "src/hepattn/keras_native/optim.py"
EB = "src/hepattn/keras_native/ebops.py"
BC = "src/hepattn/keras_native/beta_control.py"
MATCH = "src/hepattn/keras_native/matching.py"
HMF = "src/hepattn/keras/maskformer.py"
CFG = "polaris/03_make_config.py"
T_TS = "tests/keras_native/test_train_step.py"
T_OPT = "tests/keras_native/test_optim.py"
T_EB = "tests/keras_native/test_ebops.py"
T_BC = "tests/keras_native/test_beta_control.py"
T_PID = "tests/keras/test_beta_pid.py"
T_DEDUP = "tests/keras/test_quant_losses_dedupe.py"
T_SET = "tests/polaris/test_beta_setpoints.py"

# (name, file, old, new, env, pytest targets)
M = [
    ("M1 train_step does not donate by default", TS, "donate: bool = True", "donate: bool = False", "jax", [f"{T_TS}::test_state_donation_and_model_variables"]),
    (
        "M2 objective without the EBOPs penalty",
        TS,
        "        return task + penalty + regularization, (ntv, task, losses, penalty, regularization)",
        "        return task + regularization, (ntv, task, losses, penalty, regularization)",
        "jax",
        [f"{T_TS}::test_penalty_check_against_an_eager_reference"],
    ),
    (
        "M3 matmul precision context dropped",
        TS,
        "        return jax.default_matmul_precision(self.matmul_precision)",
        "        return contextlib.nullcontext()",
        "jax",
        [f"{T_TS}::test_steps_request_highest_matmul_precision"],
    ),
    (
        "M4 EBOPs switch is a no-op",
        EB,
        "                leaf._enable_ebops = bool(on)  # noqa: SLF001",
        "                pass",
        "jax",
        [f"{T_TS}::test_periodic_ebops_skipped_step_has_no_penalty", f"{T_EB}::test_loss_list_is_ebops_terms_then_regularizers"],
    ),
    (
        "M5 EBOPs switch does not restore the flags",
        EB,
        "            for leaf, flag in zip(self.leaves, before, strict=True):\n                leaf._enable_ebops = flag  # noqa: SLF001",
        "            pass",
        "jax",
        [f"{T_EB}::test_loss_list_is_ebops_terms_then_regularizers"],
    ),
    (
        "M6 set_beta writes the first leaf only",
        EB,
        "        for i in self.beta_index:\n            ntv[i]",
        "        for i in self.beta_index[:1]:\n            ntv[i]",
        "jax",
        [f"{T_TS}::test_beta_change_keeps_the_executable_and_scales_the_penalty", f"{T_EB}::test_beta_helpers_refuse_invalid_states"],
    ),
    (
        "M7 read-out divides all loss terms (regularizers included)",
        EB,
        "        n_pen = len(losses) - self.n_regularizers",
        "        n_pen = len(losses)",
        "jax",
        [f"{T_EB}::test_readout_equals_the_torch_backend_value", f"{T_EB}::test_loss_list_is_ebops_terms_then_regularizers"],
    ),
    (
        "M8 weight decay excluded by var_names (name, not path)",
        OPT,
        "    opt.exclude_from_weight_decay(var_list=quantizer_variables(model))",
        '    opt.exclude_from_weight_decay(var_names=["quantizer"])',
        "jax",
        [f"{T_OPT}::test_weight_decay_excludes_exactly_the_quantizer_paths"],
    ),
    (
        "M9 OneCycle reads micro-steps (no division by k)",
        OPT,
        "        n = ops.floor(ops.cast(step, \"float32\") / self.steps_per_update)",
        "        n = ops.floor(ops.cast(step, \"float32\"))",
        "jax",
        [f"{T_OPT}::test_onecycle_equals_torch_at_20_steps", f"{T_OPT}::test_optimizer_reads_the_schedule_at_the_optimizer_step"],
    ),
    (
        "M10 OneCycle linear anneal instead of cos",
        OPT,
        "        return end + (start - end) / 2.0 * (ops.cos(math.pi * pct) + 1.0)",
        "        return (end - start) * pct + start",
        "jax",
        [f"{T_OPT}::test_onecycle_equals_torch_at_20_steps"],
    ),
    (
        "M11 accumulation ignored by the factory",
        OPT,
        '        "gradient_accumulation_steps": accumulate if accumulate > 1 else None,',
        '        "gradient_accumulation_steps": None,',
        "jax",
        [f"{T_OPT}::test_accumulation_averages_k_micro_batches_like_one_k_times_batch", f"{T_TS}::test_accumulation_makes_one_host_call_per_micro_batch_and_one_update_per_k"],
    ),
    ("M12 Keras default Adam epsilon", OPT, "    adam_epsilon: float = 1e-8,", "    adam_epsilon: float = 1e-7,", "jax", [f"{T_OPT}::test_factory_reads_the_generated_polaris_config_and_org_base"]),
    (
        "M13 controller seeds without HGQ2's 1e-9",
        BC,
        "            err = math.log10(ebops / self.target_ebops + 1e-9)",
        "            err = math.log10(ebops / self.target_ebops)",
        "jax",
        [T_BC],
    ),
    (
        "M14 controller steps on a missing reading",
        BC,
        "        if ebops <= 0.0:\n            return self.beta",
        "        if ebops < 0.0:\n            return self.beta",
        "jax",
        [T_BC],
    ),
    (
        "M15 ramp one step ahead",
        BC,
        "    fraction = min(1.0, step / max(1, warmup_steps))",
        "    fraction = min(1.0, (step + 1) / max(1, warmup_steps))",
        "torch",
        [T_PID],
    ),
    ("M16 hybrid quant_losses walks duplicates again", HMF, "for layer in self.unique_keras_layers() for loss in layer.losses]", "for layer in self.keras_layers() for loss in layer.losses]", "torch", [T_DEDUP]),
    (
        "M17 hybrid BetaPID integral gain ignored",
        BC,
        "        beta = 10.0 ** (self.p * error + self.i * self.integral + self.d * derivative)",
        "        beta = 10.0 ** (self.p * error + 0.5 * self.integral + self.d * derivative)",
        "torch",
        [T_PID],
    ),
    ("M18 generator drops min_beta", CFG, '            "min_beta": MIN_BETA_EFFECTIVE,\n', "", "torch", [T_SET]),
    ("M19 generator ramp starts at 0", CFG, '"init_args": {"beta_start": BETA_START_EFFECTIVE,', '"init_args": {"beta_start": 0.0,', "torch", [T_SET]),
    (
        "M20 matching without stop_gradient on the costs (S4 code, gradient gate on the full model)",
        MATCH,
        '    costs = jax.lax.stop_gradient(ops.cast(costs, "float32"))',
        '    costs = ops.cast(costs, "float32")',
        "jax",
        [f"{T_TS}::test_gradients_through_matching_equal_the_fixed_index_reference"],
    ),
]


def _private_copy() -> Path:
    files = subprocess.run([GIT, "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\0")
    dst = Path(tempfile.mkdtemp(prefix="s5_mutations_"))
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
        for name, path, old, new, backend, tests in M:
            if selected and name.split()[0] not in selected:
                continue
            f = copy / path
            src = f.read_text()
            assert src.count(old) == 1, (name, src.count(old))
            f.write_text(src.replace(old, new))
            try:
                env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(copy / "src"), "HOME": os.environ["HOME"], "KERAS_BACKEND": backend}
                cmd = [PY[backend], "-m", "pytest", "-p", "no:cacheprovider", "-q", "-W", "ignore", *tests]
                r = subprocess.run(cmd, cwd=copy, env=env, capture_output=True, text=True, check=False)
                lines = r.stdout.splitlines()
                failed = [line.split(" - ")[0].replace("FAILED ", "") for line in lines if line.startswith(("FAILED", "ERROR"))]
                summary = lines[-1] if lines else r.stderr[-300:]
            finally:
                f.write_text(src)
            results.append({"mutation": name, "summary": summary, "failed": failed, "killed": bool(failed)})
            print(json.dumps(results[-1]), flush=True)
    finally:
        shutil.rmtree(copy, ignore_errors=True)
    if not selected:
        Path(__file__).with_suffix(".out").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
