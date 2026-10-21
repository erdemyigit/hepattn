# ruff: noqa: E501  (one mutation per line: a table)
"""S7 mutation runner: each mutant must make at least one of its S7 tests fail (docs/hgq/port_log/S7.md, "Mutations").

The worktree is copied (rsync, without .git and the venvs) to ``<scratch>/s7_mut``; each mutation replaces one snippet
(which must occur exactly once) in the copy, runs the named tests there with ``-x`` (``PYTHONPATH=<copy>/src``, so the
test conftest checks that hepattn resolves to the copy), and restores the file. The worktree is never written.

    python docs/hgq/port_log/S7_mutations.py <scratch> [OUT.json] [--only m01,m02]

Native tests run in ``.venv-jax`` (JAX backend), hybrid tests in ``../hepattn/.venv`` (``KERAS_BACKEND=torch``).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JAX = REPO / ".venv-jax/bin/python"
HYBRID = REPO.parent / "hepattn/.venv/bin/python"
CONV = "src/hepattn/experiments/clic/convert_to_native.py"
EXPORT = "src/hepattn/keras_native/export.py"
T = "tests/keras_native/"
EXPECTED_SURVIVORS = {"m28f"}  # the forward gate cannot see the same-shape quantizer swap (review round 1)

MUTATIONS = [
    # baselines: the unmutated copy passes every target (killed must be False)
    ("m00a", None, "", "", "jax", [T + "test_convert.py", T + "test_export.py"]),
    ("m00b", None, "", "", "torch", ["tests/keras/test_clic_clis.py", "tests/keras/test_export.py"]),
    # converter: torch modules -> native
    (
        "m01",
        CONV,
        "w_q, w_k, w_v = np.split(_np(tattn.in_proj_weight), 3, axis=0)",
        "w_k, w_q, w_v = np.split(_np(tattn.in_proj_weight), 3, axis=0)",
        "jax",
        [T + "test_convert.py", "-k", "org_float_forward"],
    ),
    (
        "m02",
        CONV,
        "            w1, w2 = np.split(weight, 2, axis=0)",
        "            w2, w1 = np.split(weight, 2, axis=0)",
        "jax",
        [T + "test_convert.py", "-k", "org_float_forward"],
    ),
    (
        "m03",
        CONV,
        "        _assign(seq_proj.tail, full[seq_proj.n :])",
        "        _assign(seq_proj.tail, full[: seq_proj.seq_len - seq_proj.n])",  # the head rows as the tail (row order alone is an equivalent mutant: the tail enters as a column sum)
        "jax",
        [T + "test_convert.py", "-k", "org_float_forward"],
    ),
    (
        "m04",
        CONV,
        "    _assign(layer.kernel, weight.T)",
        "    _assign(layer.kernel, weight.T if weight.shape[0] != weight.shape[1] else weight)",
        "jax",
        [T + "test_convert.py", "-k", "org_float_forward"],
    ),
    (
        "m05",
        CONV,
        "    if unread:\n        raise ConversionError",
        "    if False:\n        raise ConversionError",
        "jax",
        [T + "test_convert.py", "-k", "unread_tensor"],
    ),
    (
        "m06",
        CONV,
        "    weights_unwritten = [v.path for v in unwritten if id(v) not in qstate]",
        "    weights_unwritten = []",
        "jax",
        [T + "test_convert.py", "-k", "unwritten_variable"],
    ),
    # converter: hybrid state -> native
    (
        "m07",
        CONV,
        "for canon, items in found.items()}",
        "for canon, items in found.items()} if False else {canon: [var for _, var in sorted(items, key=operator.itemgetter(0), reverse=True)] for canon, items in found.items()}",
        "jax",
        [T + "test_convert.py", "-k", "hybrid_forward or bit_for_bit"],
    ),
    (
        "m08",
        CONV,
        "            put(nmodel.input_nets[int(posenc.group(1))].posenc.B, value, key)",
        "            put(nmodel.input_nets[int(posenc.group(1))].posenc.B, value * 0, key)",
        "jax",
        [T + "test_convert.py", "-k", "hybrid_forward"],
    ),
    (
        "m09",
        CONV,
        '    if unwritten:\n        raise ConversionError(f"{len(unwritten)} native variables have no hybrid source',
        '    if False:\n        raise ConversionError(f"{len(unwritten)} native variables have no hybrid source',
        "jax",
        [T + "test_convert.py", "-k", "missing_quantizer"],
    ),
    # warm start
    (
        "m10",
        "src/hepattn/keras_native/loop.py",
        "        self.state = TrainState([jnp.asarray(x) for x in tv], [jnp.asarray(x) for x in ntv], self.state.ov)",
        "        pass",
        "jax",
        [T + "test_convert_fit.py"],
    ),
    # export
    (
        "m11",
        EXPORT,
        "        kq_conf=table_q.config,",
        "        kq_conf=mask_q.config,",
        "jax",
        [T + "test_export.py", "-k", "kept_export_form_equals"],
    ),
    ("m12", EXPORT, "            x = ndense.products[i]([x1, x2])", "            x = x1", "jax", [T + "test_export.py", "-k", "swiglu_net_matches"]),
    (
        "m13",
        EXPORT,
        "    return isinstance(layer, QDense) and _activation_name(layer) not in BIT_EXACT_ACTIVATIONS",
        "    return False",
        "jax",
        [T + "test_export.py", "-k", "every_org_leaf_kind"],
    ),
    (
        "m14",
        EXPORT,
        "    table = np.asarray(keras.ops.convert_to_numpy(energy), dtype=np.float32)",
        "    table = np.asarray(keras.ops.convert_to_numpy(energy), dtype=np.float32) ** 0.5",
        "jax",
        [T + "test_export.py", "-k", "kept_export_form_equals"],
    ),
    # hybrid CLIs
    (
        "m15",
        "src/hepattn/experiments/clic/port_to_keras.py",
        "    portable_attention(model)\n",
        "    __import__('hepattn.models.encoder', fromlist=['x']).change_attn_backends(model, 'torch')\n",
        "torch",
        ["tests/keras/test_clic_clis.py", "-k", "keeps_the_linformer or complete_and_reproduces"],
    ),
    (
        "m16",
        "src/hepattn/experiments/clic/port_to_keras.py",
        "    model.register_keras_parameters()\n    model.load_state_dict",
        "    model.load_state_dict",
        "torch",
        ["tests/keras/test_clic_clis.py", "-k", "bit_for_bit"],
    ),
    (
        "m17",
        "src/hepattn/experiments/clic/evaluate_hgq.py",
        '    cfg.get("init_args", {}).pop("quant", None)\n',
        "",
        "torch",
        ["tests/keras/test_clic_clis.py", "-k", "float_baseline"],
    ),
    (
        "m18",
        "src/hepattn/experiments/clic/evaluate_hgq.py",
        'name="clf_head", activations="lut")',
        'name="clf_head", activations="inline")',
        "torch",
        ["tests/keras/test_clic_clis.py", "-k", "hls_report"],
    ),
    (
        "m19",
        "src/hepattn/keras/export.py",
        "    return assemble(kdense, input_shape, name, activations=activations)",
        "    raise NotImplementedError('SwiGLU')",
        "torch",
        ["tests/keras/test_clic_clis.py", "-k", "swiglu_nets"],
    ),
    # review round 1 (S7.md "Review round 1"): each mutant reverts one fix or disables one refusal
    ("m00c", None, "", "", "jax", [T + "test_convert_fit.py", "-k", "beta or fresh_trainer or exclusive"]),
    (
        "m20",
        CONV,
        "    if _native_attn_type(tattn.attn_type) != nattn.attn_type:",
        "    if tattn.attn_type != nattn.attn_type:",
        "jax",
        [T + "test_convert.py", "-k", "flash_attention"],
    ),
    (
        "m21",
        CONV,
        "        portable_attention(tmodel)  # for the verify forward on the CPU\n",
        "",
        "jax",
        [T + "test_convert.py", "-k", "flash_attention"],
    ),
    (
        "m22",
        CONV,
        "        if tnorm.bias is not None:  # LayerNorm(bias=False): native center=False, checked above\n            _assign(nnorm.beta, _np(tnorm.bias))",
        "        _assign(nnorm.beta, _np(tnorm.bias))",
        "jax",
        [T + "test_convert.py", "-k", "norm_without_bias"],
    ),
    ("m23", CONV, "    if missing:\n", "    if False:\n", "jax", [T + "test_convert.py", "-k", "native_twin"]),
    (
        "m24",
        CONV,
        "    expected = np.asarray(_loss_weight_owner(name, nmodel).loss_weights, dtype=np.float32)\n",
        "    return\n",
        "jax",
        [T + "test_convert.py", "-k", "class_weights"],
    ),
    (
        "m25",
        CONV,
        "            if not (np.issubdtype(array.dtype, np.integer) and np.issubdtype(dtype, np.integer)):",
        "            if False:",
        "jax",
        [T + "test_convert.py", "-k", "dtype_mismatch"],
    ),
    (
        "m26",
        CONV,
        "                if not torch.equal(keras_vars[path], value):",
        "                if False:",
        "jax",
        [T + "test_convert.py", "-k", "two_values"],
    ),
    (
        "m27",
        CONV,
        '    raise ConversionError(f"model.model.class_path',
        '    return TORCH_SOURCE\n    raise ConversionError(f"model.model.class_path',
        "jax",
        [T + "test_convert.py", "-k", "source_kind"],
    ),
    (
        "m28",
        CONV,
        "        for segs, var in zip(sorted(members, key=_suffix_order), targets, strict=True):",
        "        for segs, var in zip(sorted(members, key=_suffix_order, reverse=len({tuple(keras_vars['/'.join(m)].shape) for m in members}) == 1), targets, strict=True):",
        "jax",
        [T + "test_convert.py", "-k", "einsum_inputs_pair"],
    ),
    (  # the forward gate alone does not see the same-shape swap (expected to survive: why m28's test exists)
        "m28f",
        CONV,
        "        for segs, var in zip(sorted(members, key=_suffix_order), targets, strict=True):",
        "        for segs, var in zip(sorted(members, key=_suffix_order, reverse=len({tuple(keras_vars['/'.join(m)].shape) for m in members}) == 1), targets, strict=True):",
        "jax",
        [T + "test_convert.py", "-k", "hybrid_forward_equals"],
    ),
    (
        "m29",
        "src/hepattn/keras_native/loop.py",
        "            self.beta = self.steps.read_beta(self.state)\n        self.warm_started_from",
        "            pass\n        self.warm_started_from",
        "jax",
        [T + "test_convert_fit.py", "-k", "checkpoint_beta"],
    ),
    (
        "m30",
        "src/hepattn/keras_native/loop.py",
        "        if self.resumed_from is not None or self.global_step or self.micro_step:",
        "        if self.resumed_from is not None:",
        "jax",
        [T + "test_convert_fit.py", "-k", "fresh_trainer"],
    ),
    (
        "m31",
        "src/hepattn/keras_native/cli.py",
        "    start = fit.add_mutually_exclusive_group()",
        "    start = fit",
        "jax",
        [T + "test_convert_fit.py", "-k", "exclusive"],
    ),
    (
        "m32",
        EXPORT,
        "            x = keras.layers.Reshape((*inner, size))(ndense.products[i]([f1, f2]))",
        "            x = keras.layers.Reshape((*inner, size))(ndense.products[i]([f1, f1]))",
        "jax",
        [T + "test_export.py", "-k", "of_another_rank"],
    ),
    ("m33", EXPORT, "if shape[0] == 1 else", "if False else", "jax", [T + "test_export.py", "-k", "controls_kept"]),
    (
        "m34",
        T + "test_export.py",
        '        hls = convert_to_hls(model, tmp_path / f"k{i:02d}", compile_csim=False)',
        '        model(random_inputs(model, np.random.default_rng(4)), training=True)\n        hls = convert_to_hls(model, tmp_path / f"k{i:02d}", compile_csim=False)',
        "jax",
        [T + "test_export.py", "-k", "every_org_leaf_kind"],
    ),
    (
        "m35",
        "src/hepattn/keras_native/masked_linformer.py",
        "        p = ops.where(m, self.F(a, training=training), 0.0)",
        "        p = self.F(a, training=training)",
        "jax",
        [T + "test_export.py", "-k", "rsqrt_floor"],
    ),
    (
        "m36",
        "src/hepattn/experiments/clic/eval_keras.py",
        "    model = load_keras_state(build_keras_maskformer_from_config(model_cfg, quant=quant).eval(), opts.state, first_inputs)",
        "    model = build_keras_maskformer_from_config(model_cfg, quant=quant).eval()\n    model.materialize(first_inputs)\n"
        "    model.load_state_dict(__import__('hepattn.experiments.clic.port_to_keras', fromlist=['x']).read_model_state(opts.state))",
        "torch",
        ["tests/keras/test_clic_clis.py", "-k", "eval_keras_main"],
    ),
    (
        "m37",
        "tests/keras_native/s7_fixtures/make_s7_fixtures.py",
        "for q in layer.iq.quantizers]",
        "for q in layer.iq.quantizers[::-1]]",
        "torch",
        ["tests/keras/test_clic_clis.py", "-k", "einsum_input_order"],
    ),
]


def run(copy: Path, env_name: str, tests: list[str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"}
    env["PYTHONPATH"] = str(copy / "src")
    python = JAX
    if env_name == "torch":
        env.update(KERAS_BACKEND="torch", KERAS_TORCH_DEVICE="cpu")
        python = HYBRID
    cmd = [str(python), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning", *tests]
    return subprocess.run(cmd, cwd=copy, env=env, capture_output=True, text=True, check=False)


def main() -> None:
    scratch = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    copy = scratch / "s7_mut"
    copy.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            shutil.which("rsync") or "rsync",
            "-a",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            ".venv-jax",
            "--exclude",
            "__pycache__",
            f"{REPO}/",
            f"{copy}/",
        ],
        check=True,
    )
    results = []
    for name, rel, old, new, env_name, tests in MUTATIONS:
        if only and name not in only:
            continue
        start = time.time()
        if rel is None:
            proc = run(copy, env_name, tests)
        else:
            path = copy / rel
            original = path.read_text()
            assert original.count(old) == 1, f"{name}: snippet found {original.count(old)} times in {rel}"
            path.write_text(original.replace(old, new))
            try:
                proc = run(copy, env_name, tests)
            finally:
                path.write_text(original)
        tail = [line for line in proc.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line][-1:]
        failed = proc.returncode != 0
        results.append({
            "mutant": name,
            "file": rel,
            "tests": tests,
            "killed": failed,
            "summary": tail[0] if tail else proc.stdout[-300:],
            "seconds": round(time.time() - start),
        })
        print(json.dumps(results[-1]), flush=True)
    if out_path:
        out_path.write_text(json.dumps(results, indent=1))
    # a baseline must pass; an EXPECTED_SURVIVORS mutant documents what a weaker test misses
    survivors = [r["mutant"] for r in results if r["killed"] == (r["file"] is None or r["mutant"] in EXPECTED_SURVIVORS)]
    print(f"{len(results)} mutants, survivors: {survivors}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
