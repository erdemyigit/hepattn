"""Completeness critique of S5 + S6: the critic's own mutants, one per gate, each on a scratch copy of the tree.

Each mutant is applied to a fresh copy of ``src``, ``tests`` and ``polaris`` (never to the worktree) and runs the gate's
test there; the ``X0_*`` rows run the same tests unmutated. ``jax`` rows use ``.venv-jax``, ``torch`` rows the hybrid env.
Usage, from the worktree root::

    python docs/hgq/port_log/S56_critique_mutations.py <scratch dir> <hybrid python> [name ...]
"""

import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JAX_PY = ROOT / ".venv-jax/bin/python"
KN = "src/hepattn/keras_native"
TN = "tests/keras_native"


def sel(path: str, expr: str) -> list[str]:
    return [path, "-k", expr]


MUTANTS = {
    "X0_jax": (
        "jax",
        [],
        [
            f"{TN}/test_train_step.py",
            f"{TN}/test_ebops.py",
            f"{TN}/test_beta_control.py",
            f"{TN}/test_optim.py",
            f"{TN}/test_predict_writer.py",
            f"{TN}/test_metrics.py",
            f"{TN}/test_config.py",
            "-k",
            "training_port or every_trainable or penalty_check or beta_change or readout_equals or controller_equals_hgq2"
            " or onecycle_equals_torch or accumulation_averages or weight_decay_excludes or h5_layout or epoch_reduction"
            " or weighted_mean or loader_reads_every_form or periodic_ebops_reproduces or periodic_ebops_skipped",
        ],
    ),
    "X0_loop": ("jax", [], sel(f"{TN}/test_loop.py", "periodic_ebops_follows or continues_bit_for_bit or trace_counts")),
    "X0_torch": ("torch", [], ["tests/keras/test_quant_losses_dedupe.py", "tests/polaris/test_beta_setpoints.py"]),
    # S5: loss falls and quantizers move -> the quantizer gradients zeroed before the update
    "X1_quantizer_grads_zeroed": (
        "jax",
        [
            (
                f"{KN}/train_step.py",
                "        grads, ntv, out = self._grads_impl(state, batch, ebops_on)\n",
                "        grads, ntv, out = self._grads_impl(state, batch, ebops_on)\n"
                "        grads = [jnp.zeros_like(g) if 'quantizer' in v.path else g for g, v in zip(grads, self.model.trainable_variables)]\n",
            )
        ],
        sel(f"{TN}/test_train_step.py", "training_port"),
    ),
    # S5: every trainable variable gets a gradient -> the register tokens detached
    "X2_register_tokens_detached": (
        "jax",
        [
            (
                f"{KN}/encoder.py",
                "keras.ops.cast(self.register_tokens, x.dtype)",
                "keras.ops.stop_gradient(keras.ops.cast(self.register_tokens, x.dtype))",
            )
        ],
        sel(f"{TN}/test_train_step.py", "every_trainable"),
    ),
    # S5: penalty check rel <= 1e-5 -> the EBOPs penalty off by 1e-4
    "X3_penalty_scaled": (
        "jax",
        [
            (
                f"{KN}/train_step.py",
                "penalty = sum(penalty_terms, jnp.zeros((), jnp.float32))",
                "penalty = 0.9999 * sum(penalty_terms, jnp.zeros((), jnp.float32))",
            )
        ],
        sel(f"{TN}/test_train_step.py", "penalty_check"),
    ),
    # S5: EBOPs read-out = torch value at rel 1.2e-6 -> read-out off by 2e-6
    "X4_readout_biased": (
        "jax",
        [(f"{KN}/ebops.py", "    return float(penalty) / float(beta)\n", "    return float(penalty) / float(beta) * (1 + 2e-6)\n")],
        sel(f"{TN}/test_ebops.py", "readout_equals"),
    ),
    # S5: beta change keeps the executable -> set_beta returns a tuple (another pytree: a retrace)
    "X5_set_beta_tuple": (
        "jax",
        [
            (
                f"{KN}/ebops.py",
                "            ntv[i] = jnp.full(np.shape(ntv[i]), beta, dtype=ntv[i].dtype)\n        return ntv\n",
                "            ntv[i] = jnp.full(np.shape(ntv[i]), beta, dtype=ntv[i].dtype)\n        return tuple(ntv)\n",
            )
        ],
        sel(f"{TN}/test_train_step.py", "beta_change"),
    ),
    # S5: PeriodicEBOPs -> the EBOPs step shifted by one optimizer step
    "X6_periodic_shifted": (
        "jax",
        [(f"{KN}/loop.py", "self.global_step % self.ebops_every == 0", "(self.global_step + 1) % self.ebops_every == 0")],
        sel(f"{TN}/test_loop.py", "periodic_ebops_follows"),
    ),
    # S5: BetaPID control law = HGQ2 -> half the damping below target
    "X7_pid_half_damping": (
        "jax",
        [(f"{KN}/beta_control.py", "beta *= 1.0 - self.damp_beta_on_target", "beta *= 1.0 - 0.5 * self.damp_beta_on_target")],
        sel(f"{TN}/test_beta_control.py", "controller_equals_hgq2"),
    ),
    # S5: OneCycle = torch at 20 steps -> no floor on the micro-step / k
    "X8_onecycle_no_floor": (
        "jax",
        [
            (
                f"{KN}/optim.py",
                'n = ops.floor(ops.cast(step, "float32") / self.steps_per_update)',
                'n = ops.cast(step, "float32") / self.steps_per_update',
            )
        ],
        sel(f"{TN}/test_optim.py", "onecycle_equals_torch"),
    ),
    # S5: k micro-batches = one k-times batch -> accumulation dropped (one update per micro-batch)
    "X9_no_accumulation": (
        "jax",
        [(f"{KN}/optim.py", '"gradient_accumulation_steps": accumulate if accumulate > 1 else None,', '"gradient_accumulation_steps": None,')],
        sel(f"{TN}/test_optim.py", "accumulation_averages"),
    ),
    # S5: weight-decay exclusion covers every quantizer path -> the integer bits decayed
    "X10_wd_decays_integer_bits": (
        "jax",
        [(f"{KN}/optim.py", 'if "quantizer" in v.path]', 'if "quantizer" in v.path and not v.path.endswith("/i")]')],
        sel(f"{TN}/test_optim.py", "weight_decay_excludes"),
    ),
    # S6: resume bitwise -> the last EBOPs reading is not restored (the PID gets no reading at the next epoch start)
    "X11_resume_drops_ebops_reading": (
        "jax",
        [(f"{KN}/loop.py", 'self._reading, self._ebops_value = None, meta["last_ebops"]', "self._reading, self._ebops_value = None, None")],
        sel(f"{TN}/test_loop.py", "continues_bit_for_bit"),
    ),
    # S6: trace counts -> every val batch goes through the tail program (one more tail trace)
    "X12_tail_for_every_batch": (
        "jax",
        [(f"{KN}/loop.py", "            if n < self.batch_size:\n", "            if n <= self.batch_size:\n")],
        sel(f"{TN}/test_loop.py", "trace_counts"),
    ),
    # S6: prediction h5 = PflowPredictionWriter -> half precision
    "X13_h5_half_precision": (
        "jax",
        [(f"{KN}/predict_writer.py", 'precision="full",', 'precision="half",')],
        sel(f"{TN}/test_predict_writer.py", "h5_layout"),
    ),
    # S6: metrics = torchmetrics / Lightning -> epoch mean weighted per batch, not per event
    "X14_mean_per_batch": (
        "jax",
        [(f"{KN}/metrics.py", "        self.weight += weight\n", "        self.weight += 1.0\n")],
        sel(f"{TN}/test_metrics.py", "epoch_reduction or weighted_mean"),
    ),
    # S6: YAML loader reads every form -> gradient_clip_val dropped
    "X15_clip_dropped": (
        "jax",
        [(f"{KN}/config.py", "clip_norm=None if clip is None else float(clip),", "clip_norm=None,")],
        sel(f"{TN}/test_config.py", "loader_reads_every_form"),
    ),
    # S5: PeriodicEBOPs active vs skipped -> the switch ignored (EBOPs always on)
    "X18_ebops_switch_ignored": (
        "jax",
        [
            (
                f"{KN}/ebops.py",
                "                leaf._enable_ebops = bool(on)  # noqa: SLF001\n",
                "                leaf._enable_ebops = True  # noqa: SLF001\n",
            )
        ],
        [f"{TN}/test_ebops.py", f"{TN}/test_train_step.py", "-k", "periodic_ebops_reproduces or periodic_ebops_skipped"],
    ),
    # S5: hybrid quant_losses dedupe -> the task heads counted twice again
    "X16_hybrid_double_count": (
        "torch",
        [
            (
                "src/hepattn/keras/maskformer.py",
                "return list({id(layer): layer for layer in self.keras_layers()}.values())",
                "return list(self.keras_layers())",
            )
        ],
        ["tests/keras/test_quant_losses_dedupe.py"],
    ),
    # S5: provisional setpoints -> init beta off by 10%
    "X17_init_beta_off": (
        "torch",
        [("polaris/03_make_config.py", 'os.environ.get("HGQ_INIT_BETA", "8.7e-11")', 'os.environ.get("HGQ_INIT_BETA", "9.6e-11")')],
        ["tests/polaris/test_beta_setpoints.py"],
    ),
}


def apply(copy: Path, edits) -> None:
    for rel, old, new in edits:
        path = copy / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: anchor found {text.count(old)} times: {old!r}")
        path.write_text(text.replace(old, new))


def run(scratch: Path, torch_py: str, name: str) -> str:
    env_kind, edits, args = MUTANTS[name]
    copy = scratch / name
    shutil.rmtree(copy, ignore_errors=True)
    for part in ("src", "tests", "polaris"):
        shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
    shutil.copy(ROOT / "pyproject.toml", copy / "pyproject.toml")
    apply(copy, edits)
    py = str(JAX_PY) if env_kind == "jax" else torch_py
    env = {**os.environ, "KERAS_BACKEND": env_kind, "PYTHONPATH": str(copy / "src")}
    proc = subprocess.run(
        [py, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", "-rf"], cwd=copy, env=env, capture_output=True, text=True, check=False
    )
    (copy / "pytest.out").write_text(proc.stdout + proc.stderr)
    # the -q final line ("1 failed, 20 deselected ... in 3.2s"); warning text such as "was passed an input" must not match
    final = re.compile(r"^\d+ (passed|failed|error)")
    summary = [line for line in proc.stdout.splitlines() if line.startswith(("FAILED", "ERROR")) or final.match(line)]
    return "\n  ".join([name, *summary])


def main(scratch: str, torch_py: str, names: list[str]) -> int:
    with ThreadPoolExecutor(max_workers=int(os.environ.get("MUT_WORKERS", "3"))) as pool:
        for line in pool.map(lambda n: run(Path(scratch), torch_py, n), names or list(MUTANTS)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
