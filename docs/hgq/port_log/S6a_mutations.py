"""S6a mutation runner: each mutant must make at least one S6a test fail (docs/hgq/port_log/S6a.md, "Mutations").

Each mutation replaces one snippet (which must occur exactly once) in a committed source file, runs the named tests
with ``-x``, and restores the file from memory; at the end ``git diff --quiet`` on the mutated files must hold.

    PYTHONPATH=src .venv-jax/bin/python docs/hgq/port_log/S6a_mutations.py [OUT.json] [--only M1,M2]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
N = "src/hepattn/keras_native/"
T = "tests/keras_native/"

FAST = [T + "test_config.py", "-k", "not build_the_same_model and not control_forward and not control_shape and not quant_spec"]
CK_FAST = [T + "test_checkpoint.py", "-k", "longest or durable"]

MUTATIONS = [
    # config.py
    ("M1", N + "config.py", "        return _activation_name(node)", "        return node", T + "test_config.py"),
    ("M2", N + "config.py", 'if key == "decoder" and "class_path" in node:', "if False:", T + "test_config.py"),
    (
        "M3",
        N + "config.py",
        "out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)",
        "out[key] = copy.deepcopy(value)",
        T + "test_config.py",
    ),
    (
        "M4",
        N + "config.py",
        "    packaged = CLIC_CONFIGS / candidate.name\n",
        "    packaged = CLIC_CONFIGS / 'missing' / candidate.name\n",
        T + "test_config.py",
    ),
    ("M5", N + "config.py", '.get("every_n_steps", 1))', '.get("every_n_step", 1))', T + "test_config.py"),
    # checkpoint.py
    (
        "M6",
        N + "checkpoint.py",
        '    return "/".join(_AUTO_SUFFIX.sub("", segment) for segment in path.split("/"))',
        "    return path",
        T + "test_checkpoint.py",
    ),
    ("M7", N + "checkpoint.py", 'keys.append(name if total[name] == 1 else f"{name}#{seen[name]}")', "keys.append(name)", T + "test_checkpoint.py"),
    ("M8", N + "checkpoint.py", '    if digest != sidecar.get("sha256"):', "    if False:", T + "test_checkpoint.py"),
    (
        "M9",
        N + "checkpoint.py",
        '        ref = next((r for r in by_length if name.startswith(r + "_")), None)',
        "        ref = None",
        T + "test_checkpoint.py",
    ),
    (
        "M10",
        N + "checkpoint.py",
        '    if expect_config_hash is not None and meta.get("config_hash") != expect_config_hash:',
        "    if False:",
        T + "test_checkpoint.py",
    ),
    (
        "M11",
        N + "checkpoint.py",
        "    with _atomic(path) as f:\n        np.savez(f, **arrays)",
        "    with path.open('wb') as f:\n        np.savez(f, **arrays)",
        T + "test_checkpoint.py",
    ),
    (
        "M12",
        N + "checkpoint.py",
        "        return self.every_n_steps is not None and step > 0 and step % self.every_n_steps == 0",
        "        return self.every_n_steps is not None and step % self.every_n_steps == 0",
        T + "test_checkpoint.py",
    ),
    # data.py / native_data.py
    ("M13", N + "data.py", "        return array.astype(np.int32)", "        return array", T + "test_data.py"),
    (
        "M14",
        N + "data.py",
        '    if key == f"{target_object}_class":\n        return null_class',
        '    if key == f"{target_object}_class":\n        return 0',
        T + "test_data.py",
    ),
    (
        "M15",
        N + "data.py",
        "np.concatenate([array, fill]) if missing else array",
        "np.concatenate([fill, array]) if missing else array",
        T + "test_data.py",
    ),
    (
        "M16",
        N + "data.py",
        "        event_valid = np.arange(self.batch_size) < size",
        "        event_valid = np.ones(self.batch_size, bool)",
        T + "test_data.py",
    ),
    ("M17", "src/hepattn/experiments/clic/native_data.py", '    train = split == "train"', "    train = False", T + "test_data.py"),
    ("M18", "src/hepattn/experiments/clic/native_data.py", "manual_seed(self.seed + self.epoch)", "manual_seed(self.seed)", T + "test_data.py"),
    # metrics.py
    (
        "M19",
        N + "metrics.py",
        "        weights = ((self.tp + self.fp + self.fn) != 0).astype(np.float32)",
        "        weights = np.ones(self.num_classes, np.float32)",
        T + "test_metrics.py",
    ),
    ("M20", N + "metrics.py", '        other = self.fp if self.stat == "precision" else self.fn', "        other = self.fp", T + "test_metrics.py"),
    ("M21", N + "metrics.py", "/ tgt.sum(-1).astype(np.float32))", "/ pred.sum(-1).astype(np.float32))", T + "test_metrics.py"),
    (
        "M22",
        N + "metrics.py",
        "        truth_valid = class_labels < self.num_classes - 1",
        '        truth_valid = np.asarray(targets["particle_valid"]) | True',
        T + "test_metrics.py",
    ),
    ("M23", N + "metrics.py", "abs_err / np.abs(target) + np.float32(1e-8)", "abs_err / target + np.float32(1e-8)", T + "test_metrics.py"),
    # predict_writer.py
    (
        "M24",
        N + "predict_writer.py",
        'f"{obj}_valid": classes < task.num_classes}',
        'f"{obj}_valid": classes <= task.num_classes}',
        T + "test_predict_writer.py",
    ),
    (
        "M25",
        N + "predict_writer.py",
        '"targets": _field(targets["particle_node_valid"], "truth_masks", "i8"),',
        '"targets": _field(targets["particle_node_valid"], "truth_mask", "i8"),',
        T + "test_predict_writer.py",
    ),
    ("M26", N + "predict_writer.py", "        if name in scaler.transforms:", "        if False:", T + "test_predict_writer.py"),
    (
        "M27",
        N + "predict_writer.py",
        "keep = slice(None) if event_valid is None else np.asarray(event_valid, bool)",
        "keep = slice(None)",
        T + "test_predict_writer.py",
    ),
    (
        "M28",
        N + "predict_writer.py",
        '        probs = ops.sigmoid(out[f"{hit}_logit"])',
        '        probs = ops.sigmoid(out[f"{hit}_logit"] + 0.01)',
        T + "test_predict_writer.py",
    ),
    # fix round (reviewer findings R1-R14 and the hash / durability fixes); FAST runs only the config tests that
    # do not build models, CK_FAST the checkpoint tests that need no compiled step
    ("M29", N + "config.py", 'lr_max=float(lrs["max"]),', 'lr_max=float(lrs["initial"]),', FAST),
    ("M30", N + "config.py", 'every_n_train_steps=step.get("every_n_train_steps"),', "every_n_train_steps=None,", FAST),
    ("M31", N + "config.py", 'save_last=bool(step.get("save_last") or best.get("save_last")),', "save_last=False,", FAST),
    ("M32", N + "config.py", 'weight_decay=float(lrs.get("weight_decay", 0.0)),', "weight_decay=0.0,", FAST),
    ("M33", N + "config.py", 'seed=cfg.get("seed_everything"),', "seed=None,", FAST),
    ("M34", N + "config.py", 'num_val=int(data.get("num_val", -1)),', "num_val=-1,", FAST),
    ("M35", N + "config.py", 'pct_start=float(lrs["pct_start"]),', "pct_start=0.3,", FAST),
    ("M36", N + "config.py", 'dirpath=step.get("dirpath") or best.get("dirpath"),', "dirpath=None,", FAST),
    ("M37", N + "config.py", 'mode=best.get("mode", "min"),', 'mode="max",', FAST),
    ("M38", N + "config.py", 'batch_size=int(data["batch_size"]),', "batch_size=32,", FAST),
    ("M39", N + "config.py", "if k not in defaults or v != _canonical(defaults[k], k)}}", "}}", FAST),
    ("M40", N + "config.py", "        return _file_digest(node)", "        return node", FAST),
    ("M41", N + "config.py", "return config_hash(model_description(self.model_section))", "return config_hash(self.model_section)", FAST),
    ("M42", N + "config.py", "        return float(node)  # jsonargparse", "        return node  # jsonargparse", FAST),
    (
        "M43",
        N + "config.py",
        'beta_pid=_with_defaults("BetaPID", callbacks.get("BetaPID"), BETA_PID_ARGS),',
        'beta_pid=callbacks.get("BetaPID"),',
        FAST,
    ),
    ("M44", N + "config.py", '    "p": (float, 1.0),', '    "p": (float, 2.0),', FAST),
    (
        "M45",
        N + "metrics.py",
        '    ).items():\n                        self.means[f"{prefix}{layer}_{task_name}_{key}"].update(value, batch_size)',
        '    ).items():\n                        self.means[f"{prefix}{layer}_{task_name}_{key}"].update(value, 1.0)',
        T + "test_metrics.py",
    ),
    (
        "M46",
        N + "metrics.py",
        'self.means[f"{prefix}{layer}_{key}"].update(value, 1.0)',
        'self.means[f"{prefix}{layer}_{key}"].update(value, batch_size)',
        T + "test_metrics.py",
    ),
    (
        "M47",
        N + "metrics.py",
        'self.fields).items():\n                        self.means[f"{prefix}{layer}_{task_name}_{key}"].update(value, batch_size)',
        'self.fields).items():\n                        self.means[f"{prefix}{layer}_{task_name}_{key}"].update(value, 1.0)',
        T + "test_metrics.py",
    ),
    ("M48", N + "metrics.py", 'custom = self.stage != "train"', "custom = True", T + "test_metrics.py"),
    ("M49", N + "metrics.py", 'key = f"{self.stage}/*_{metric}" if self.pool_layers else name', "key = name", T + "test_metrics.py"),
    (
        "M50",
        N + "checkpoint.py",
        "if tuple(array.shape) != tuple(var.shape) or np.dtype(array.dtype) != np.dtype(var.dtype):",
        "if tuple(array.shape) != tuple(var.shape):",
        T + "test_checkpoint.py",
    ),
    ("M51", N + "checkpoint.py", '    if meta != sidecar["meta"]:', "    if False:", T + "test_checkpoint.py"),
    ("M52", N + "checkpoint.py", "by_length = sorted(refs, key=len, reverse=True)", "by_length = sorted(refs, key=len)", CK_FAST),
    ("M53", N + "checkpoint.py", "    _fsync_directory(path.parent)\n", "", CK_FAST),
    (
        "M54",
        N + "checkpoint.py",
        "    if not overwrite and (path.exists() or sidecar_path(path).exists()):",
        "    if False:",
        T + "test_checkpoint.py",
    ),
    (
        "M55",
        N + "checkpoint.py",
        'path = save_checkpoint(self._new_path("step", step),',
        'path = save_checkpoint(self.directory / f"step-{step:09d}.npz",',
        T + "test_checkpoint.py",
    ),
    ("M56", N + "data.py", "    if array.dtype == np.float64:\n        return array.astype(np.float32)\n", "", T + "test_data.py"),
]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    results = []
    touched = set()
    for name, rel, old, new, tests in MUTATIONS:
        if only and name not in only:
            continue
        path = REPO / rel
        original = path.read_text()
        assert original.count(old) == 1, f"{name}: snippet found {original.count(old)} times in {rel}"
        touched.add(rel)
        path.write_text(original.replace(old, new))
        start = time.time()
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-x",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-W",
                    "ignore::DeprecationWarning",
                    *([tests] if isinstance(tests, str) else tests),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_text(original)
        failed = [line for line in proc.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
        result = {
            "mutation": name,
            "file": rel,
            "killed": proc.returncode != 0,
            "first_failure": failed[:1],
            "seconds": round(time.time() - start, 1),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    diff = subprocess.run(["git", "diff", "--quiet", "--", *sorted(touched)], cwd=REPO, check=False)  # noqa: S607  (git from PATH)
    assert diff.returncode == 0, "a mutated file differs from git after the run"
    survivors = [r["mutation"] for r in results if not r["killed"]]
    print(f"{len(results) - len(survivors)}/{len(results)} killed; survivors: {survivors}")
    if out_path:
        out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
