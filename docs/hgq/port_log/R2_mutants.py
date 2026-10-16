# ruff: noqa: E501  (the mutants are verbatim source lines)
"""R2 mutation check: each mutant is applied to a scratch copy of the tree, never to the tracked files.

Usage (from the repo root; KERAS_BACKEND=jax env with torch, e.g. hepattn-keras-jax/.venv-jax):
    python docs/hgq/port_log/R2_mutants.py <scratch dir> [mutant ...]
Copies src/, tests/ and pyproject.toml into <scratch dir>/<mutant>, applies the edit (each old string must occur
exactly once), runs that mutant's target tests there and prints the pytest summary. A mutant is killed when at least
one target fails. Results of the R2 run: docs/hgq/port_log/R2.md, "Mutants".
"""

import shutil
import subprocess
import sys
from pathlib import Path

ML = "src/hepattn/keras_native/masked_linformer.py"
AT = "src/hepattn/keras_native/attention.py"
FA = "src/hepattn/keras_native/factory.py"
T = "tests/keras_native/test_masked_linformer.py"
MUTANTS = {
    "m01_mask_by_multiply": [
        (
            ML,
            "        k, v = ops.where(seen, k, 0.0), ops.where(seen, v, 0.0)\n        s = ops.where(m, self.qk([q * self.d**-0.5, k], training=training), 0.0)",
            "        fs = ops.cast(seen, k.dtype)\n        k, v = k * fs, v * fs\n        s = self.qk([q * self.d**-0.5, k], training=training) * ops.cast(m, k.dtype)",
        ),
        (
            ML,
            "            k, v = ops.where(km, k, 0.0), ops.where(km, v, 0.0)",
            "            k, v = k * ops.cast(km, k.dtype), v * ops.cast(km, v.dtype)",
        ),
    ],
    "m02_numerator_head_only": [(ML, "        if tail is not None:\n            full2 = full2 + ops.sum(ops.square(tail), 0)", "        pass")],
    "m03_no_final_where": [(ML, "        p = ops.where(m, self.F(a, training=training), 0.0)", "        p = self.F(a, training=training)")],
    "m04_renorm_dropped": [
        (ML, "        if self.renorm:\n            s = self.mul_k(", "        if False:\n            s = self.mul_k("),
        (ML, "        if self.renorm:\n            a = self.mul_v(", "        if False:\n            a = self.mul_v("),
        (ML, "        if kv_mask is not None and self.renorm:", "        if False:"),
    ],
    "m05_tail_dropped": [(ML, "        self.has_tail = renorm and seq_len > n", "        self.has_tail = False")],
    "m06_no_sort_gather": [
        (AT, "        if kv_sort_idx is not None:\n            idx = keras.ops.cast", "        if False:\n            idx = keras.ops.cast")
    ],
    "m07_helen_rank_k_mask": [
        (
            AT,
            "    unknown = sorted(set(attn_kwargs) - ATTENTION_KEYS)",
            '    if attn_type == "linformer":\n        from hepattn.keras_native.linformer import LinformerAttention  # noqa: PLC0415\n\n'
            '        seq_len, k = attn_kwargs.pop("linformer_seq_len"), attn_kwargs.pop("linformer_k")\n'
            "        return LinformerAttention(dim, seq_len=seq_len, k=k, factory=factory, name=name, **attn_kwargs)\n"
            "    unknown = sorted(set(attn_kwargs) - ATTENTION_KEYS)",
        )
    ],
    "m08_linformer_k_ignored": [
        (AT, "        self.linformer_k = linformer_k\n", "        self.linformer_k = None if linformer_k is None else 256\n")
    ],
    "m09_renorm_lanes_12": [(ML, "    lane = 2 * frac_cap + 4", "    lane = 12")],
    "m10_no_frac_cap": [
        (
            FA,
            '        return QuantizerConfig("kif", "weight", i0=base["i0"], f0=min(base["f0"], max_frac_bits), fc=MinMax(-24, max_frac_bits))',
            '        return QuantizerConfig("kif", "weight", i0=base["i0"], f0=base["f0"])',
        )
    ],
    "m11_core_imports_jax": [
        (ML, "import keras\nfrom hgq.quantizer import Quantizer", "import jax  # noqa: F401\nimport keras\nfrom hgq.quantizer import Quantizer")
    ],
    "m12_glorot_in_proj": [
        (
            AT,
            "            return factory.dense(dim, use_bias=bias, name=sub(part), kernel_initializer=keras.initializers.RandomUniform(-bound, bound))",
            "            return factory.dense(dim, use_bias=bias, name=sub(part))",
        )
    ],
    "m13_value_residual_ignored": [(AT, "                vp = vp * mix + initial_v * (1.0 - mix)\n", "                pass\n")],
    "m14_rescale_kv_mask_not_query_mask": [
        (
            ML,
            "            s = self.mul_k([s, rescale(self.kept_k, self.lut_k, attn_mask, self.E.energy(training), training)], training=training)",
            "            s = self.mul_k([s, rescale(self.kept_k, self.lut_k, ops.broadcast_to(ops.any(attn_mask, axis=1, keepdims=True), ops.shape(attn_mask)), self.E.energy(training), training)], training=training)",
        )
    ],
}
TARGETS = {
    "m01_mask_by_multiply": [
        f"{T}::test_attend_core_forward_and_gradients_equal_masked_linformer",
        f"{T}::test_project_core_equals_project_then_unmasked_attention",
        f"{T}::test_padded_tokens_do_not_reach_valid_outputs",
    ],
    "m02_numerator_head_only": [
        f"{T}::test_attend_core_forward_and_gradients_equal_masked_linformer",
        f"{T}::test_project_core_equals_project_then_unmasked_attention",
        f"{T}::test_linformer_attention_equals_org",
    ],
    "m03_no_final_where": [
        f"{T}::test_attend_core_forward_and_gradients_equal_masked_linformer",
        "tests/keras_native/test_masked_query.py::test_fully_masked_query_is_key_independent",
        f"{T}::test_linformer_attention_equals_org",
    ],
    "m04_renorm_dropped": [
        f"{T}::test_attend_core_forward_and_gradients_equal_masked_linformer",
        f"{T}::test_project_core_equals_project_then_unmasked_attention",
        f"{T}::test_linformer_attention_equals_org",
        f"{T}::test_batch_of_one_all_valid_mask_differs_from_no_mask_below_seq_len",
    ],
    "m05_tail_dropped": [
        f"{T}::test_attend_core_forward_and_gradients_equal_masked_linformer",
        f"{T}::test_project_core_equals_project_then_unmasked_attention",
        f"{T}::test_linformer_attention_equals_org",
        f"{T}::test_in_projection_and_sequence_projection_init_match_org",
    ],
    "m06_no_sort_gather": [f"{T}::test_linformer_attention_equals_org", f"{T}::test_kv_sort_idx_gathers_keys_masks_and_attn_mask_columns"],
    "m07_helen_rank_k_mask": [
        f"{T}::test_build_attention_honours_linformer_seq_len_and_k",
        f"{T}::test_build_attention_rejects_unknown_or_missing_options",
        "tests/keras_native/test_float_parity.py::test_clic_linformer_encoder_parity",
    ],
    "m08_linformer_k_ignored": [f"{T}::test_build_attention_honours_linformer_seq_len_and_k", f"{T}::test_linformer_attention_equals_org"],
    "m09_renorm_lanes_12": [
        f"{T}::test_renorm_precision_bound_at_one_valid_key",
        f"{T}::test_projection_fractional_bits_are_capped_and_the_table_fits_hls4ml",
    ],
    "m10_no_frac_cap": [f"{T}::test_projection_fractional_bits_are_capped_and_the_table_fits_hls4ml"],
    "m11_core_imports_jax": ["tests/keras_native/test_import_hygiene.py"],
    "m12_glorot_in_proj": [f"{T}::test_in_projection_and_sequence_projection_init_match_org"],
    "m13_value_residual_ignored": [f"{T}::test_linformer_value_residual_two_layers_equal_org"],
    "m14_rescale_kv_mask_not_query_mask": [
        f"{T}::test_attend_core_forward_and_gradients_equal_masked_linformer",
        f"{T}::test_linformer_attention_equals_org",
    ],
}
MUTANTS["m15_q_sa_masked_project_core"] = [
    (
        T,
        '    if kind == "project_unmasked":\n        kwargs = {"linformer_core": kind, **kwargs}',
        '    if False:\n        kwargs = {"linformer_core": kind, **kwargs}',
    )
]
TARGETS["m15_q_sa_masked_project_core"] = [f"{T}::test_trains_under_jit_at_the_production_spec", f"{T}::test_leaf_inventory_and_every_leaf_is_live"]
MUTANTS["m16_mul_k_bypassed"] = [
    (
        ML,
        "            s = self.mul_k([s, rescale(self.kept_k, self.lut_k, attn_mask, self.E.energy(training), training)], training=training)",
        "            s = s * rescale(self.kept_k, self.lut_k, attn_mask, self.E.energy(training), training)[:, None]",
    )
]
TARGETS["m16_mul_k_bypassed"] = [
    f"{T}::test_trains_under_jit_at_the_production_spec",
    f"{T}::test_leaf_inventory_and_every_leaf_is_live",
    f"{T}::test_attend_core_forward_and_gradients_equal_masked_linformer",
]


def run(scratch: Path, mutant: str) -> str:
    root = Path(__file__).resolve().parents[3]
    tree = scratch / mutant
    shutil.rmtree(tree, ignore_errors=True)
    tree.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", ".parity_ledger.jsonl")
    for part in ("src", "tests"):
        shutil.copytree(root / part, tree / part, ignore=ignore)
    shutil.copy(root / "pyproject.toml", tree)
    for rel, old, new in MUTANTS[mutant]:
        path = tree / rel
        text = path.read_text()
        assert text.count(old) == 1, (mutant, rel, text.count(old))
        path.write_text(text.replace(old, new))
    env = {"KERAS_BACKEND": "jax", "PYTHONPATH": str(tree / "src"), "HEPATTN_PARITY_LEDGER": "0", "PATH": "/usr/bin:/bin"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf", *TARGETS[mutant]]
    proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, check=False)
    return proc.stdout.strip().splitlines()[-1]


if __name__ == "__main__":
    scratch_dir = Path(sys.argv[1])
    for name in sys.argv[2:] or MUTANTS:
        print(f"{name}: {run(scratch_dir, name)}", flush=True)
