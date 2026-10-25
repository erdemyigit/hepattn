r"""Mutation check of the G4 tests: each mutant edits one line of a scratch copy; its target tests must fail.

    python docs/hgq/port_log/G4_mutations.py --jax-python .venv-jax/bin/python --hybrid-python ../hepattn/.venv/bin/python \\
        --out docs/hgq/port_log/G4_evidence/mutations.json

The copy holds ``polaris/``, ``src/`` and the two test files; the G4 tests run in the JAX env (all of
``tests/polaris/test_g4_compare.py`` runs there, the data-order test included), the BetaPID test in the hybrid env.
m00 is the unmutated baseline (must pass).
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
G4 = "polaris/19_g4_compare.py"
HY = "polaris/19_g4_hybrid.py"
CB = "src/hepattn/keras/callbacks.py"
T_G4 = "tests/polaris/test_g4_compare.py"
T_PID = "tests/keras/test_beta_pid.py::test_controller_state_survives_a_checkpoint"

# id: (file, old, new, test target, env)
MUTANTS = {
    "m00": (None, None, None, T_G4, "jax"),
    "m00b": (None, None, None, T_PID, "hybrid"),
    "m01": (G4, "math.atan(c / math.sqrt(0.75))", "math.atan(c)", T_G4, "jax"),
    "m02": (G4, '"inside": (not degenerate) and abs(x - m) <= half,', '"inside": (not degenerate) and abs(x - m) <= half / 2,', T_G4, "jax"),
    "m03": (G4, '"inside": (not degenerate) and abs(x - m) <= half,', '"inside": abs(x - m) <= half,', T_G4, "jax"),
    "m04": (G4, 'pre["lr_ok"] = bool(lr) and max(lr) <= LR_RTOL', 'pre["lr_ok"] = True', T_G4, "jax"),
    "m05": (G4, 'pre["forward_ok"] = forward_ok(forward)', 'pre["forward_ok"] = True', T_G4, "jax"),
    "m06": (G4, 'ok=at_k["inside"] or closer)', 'ok=at_k["inside"])', T_G4, "jax"),
    "m07": (G4, "closer = abs(e_j) <= max(abs(e_a), abs(e_b))", "closer = True", T_G4, "jax"),
    "m08": (
        G4,
        'out["status"] = "PASS" if v["inside"] and ebops.get("ok") else "FAIL"',
        'out["status"] = "PASS" if v["inside"] else "FAIL"',
        T_G4,
        "jax",
    ),
    "m09": (G4, 'return {int(r["step"]): r for r in rows', 'return {int(r["step"]): r for r in reversed(rows)', T_G4, "jax"),
    "m10": (G4, "for s in range(ckpt_every, k + 1, ckpt_every)", "for s in range(ckpt_every, k, ckpt_every)", T_G4, "jax"),
    "m11": (G4, 't["limit_train_batches"] = s["epoch_steps"] * s["accum"]', 't["limit_train_batches"] = s["epoch_steps"]', T_G4, "jax"),
    "m12": (G4, "    return 2.0 / math.pi * math.atan", "    return 1.0 / math.pi * math.atan", T_G4, "jax"),
    "m13": (G4, "return bool(floats) and all(", "return all(", T_G4, "jax"),
    "m23": (G4, ' and all(v["flips"] <= flip_frac * v["n"] for v in flips)', "", T_G4, "jax"),
    "m24": (
        G4,
        'return bool(platforms) and all(float_ok(p["float"]) for p in platforms.values())',
        'return bool(platforms) and any(float_ok(p["float"]) for p in platforms.values())',
        T_G4,
        "jax",
    ),
    "m25": (
        G4,
        'platforms = (check or {}).get("platforms") or {}',
        'platforms = (check or {}).get("platforms") or {"x": {"float": (check or {}).get("errors", {})}}',
        T_G4,
        "jax",
    ),
    "m26": (G4, "        keep = np.abs(value) < 1e30\n", "        keep = np.abs(value) < np.inf\n", T_G4, "jax"),
    "m14": (G4, '    if missing:\n        out["status"] = "INCOMPLETE"', '    if False:\n        out["status"] = "INCOMPLETE"', T_G4, "jax"),
    "m15": (G4, '    cfg["data"]["data_seed"] = seed\n', "", T_G4, "jax"),
    "m16": (HY, "start = global_step * accum - epoch * micro_per_epoch", "start = global_step * accum - epoch * micro_per_epoch + 1", T_G4, "jax"),
    "m17": (HY, "generator=torch.Generator().manual_seed(seed + epoch)", "generator=torch.Generator().manual_seed(seed)", T_G4, "jax"),
    "m18": (HY, "    order = order[: len(order) - len(order) % batch_size]\n", "", T_G4, "jax"),
    "m19": (
        HY,
        "        if pos is not None:\n            self.epoch, self.start_batch = pos",
        "        if False:\n            self.epoch, self.start_batch = pos",
        T_G4,
        "jax",
    ),
    "m22": (G4, "at_k = band(e_j, e_a, e_b, c, floor=EBOPS_FLOOR_DEX)", "at_k = band(e_j, e_a, e_b, c)", T_G4, "jax"),
    "m27": (G4, 'if not (args.ignore_forward or forward_ok(manifest.get("forward_check"))):', "if False:", T_G4, "jax"),
    "m20": (
        CB,
        "    def state_dict(self) -> dict:\n        return self.controller.state_dict()",
        "    def state_dict(self) -> dict:\n        return {}",
        T_PID,
        "hybrid",
    ),
    "m21": (CB, "        self.controller.load_state_dict(state_dict)", "        pass", T_PID, "hybrid"),
}


def make_copy(root: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "polaris", root / "polaris", ignore=ignore)
    shutil.copytree(REPO / "src", root / "src", ignore=ignore)
    for test in (T_G4, T_PID.split("::", maxsplit=1)[0]):
        (root / test).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / test, root / test)


def run(mid: str, spec, pythons: dict) -> dict:
    path, old, new, target, env_name = spec
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_copy(root)
        if path is not None:
            text = (root / path).read_text()
            if text.count(old) != 1:
                return {"id": mid, "error": f"pattern found {text.count(old)} times in {path}"}
            (root / path).write_text(text.replace(old, new))
        env = {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"} | {"PYTHONPATH": str(root / "src")}
        if env_name == "hybrid":
            env["KERAS_BACKEND"] = "torch"
        cp = subprocess.run(
            [pythons[env_name], "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", target],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    tail = [ln for ln in cp.stdout.splitlines() if ln.strip()][-1:]
    failed = next((ln for ln in cp.stdout.splitlines() if ln.startswith("FAILED")), "")
    outcome = "pass" if cp.returncode == 0 else "killed"
    return {"id": mid, "file": path, "env": env_name, "outcome": outcome, "summary": tail[0] if tail else "", "first_failure": failed}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jax-python", required=True)
    p.add_argument("--hybrid-python", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    args = p.parse_args()
    pythons = {
        "jax": str(Path(args.jax_python).absolute()),
        "hybrid": str(Path(args.hybrid_python).absolute()),
    }  # not resolve(): a venv python is a symlink
    ids = args.only.split(",") if args.only else list(MUTANTS)
    results = []
    for mid in ids:
        res = run(mid, MUTANTS[mid], pythons)
        results.append(res)
        print(json.dumps(res), flush=True)
    ok = all("outcome" in r and (r["outcome"] == "pass") == r["id"].startswith("m00") for r in results)  # a stale pattern is a failure
    Path(args.out).write_text(json.dumps({"results": results, "all_as_expected": ok}, indent=1) + "\n")
    print("ALL-AS-EXPECTED" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
