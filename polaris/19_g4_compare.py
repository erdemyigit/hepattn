r"""Gate G4 (docs/hgq/KERAS_JAX_PLAN.md section 4, "S8 / G4"): JAX vs hybrid training from the same converted initial
weights and the same data order, K = 5000 optimizer steps, fp32, one A100 per run.

Stages (each a command; ``19_g4_compare.pbs`` runs them on Polaris, ``G4_STAGE``):

``init``    The G4 config (``03_make_config.py`` with ``HGQ_PRECISION=32 HGQ_DEVICES=1``, plus ``g4_overrides``), the
            hybrid's initial state (``19_g4_hybrid.py init``: Lightning checkpoint at step 0 and the eval outputs on
            the first validation batch), its conversion to the native format (``convert_to_native``), the native
            forward on the same batch from the converted weights (``forward_check.json``), and ``manifest.json``
            (settings, seeds, sha256 of every file). Refuses to overwrite a complete init.
``run``     One arm (``--arm``) until it reaches K or the process dies; resumes from the arm's own last checkpoint,
            otherwise starts from the init (hybrid: ``G4_WARM_START`` = the init checkpoint; JAX: ``--init-from`` the
            converted one). Appends a segment (job, steps, wall time) to ``<arm>/segments.jsonl``. Does nothing for a
            finished arm, so a chained job that finds every arm done costs a few seconds.
``status``  Steps done per arm.
``report``  The verdict JSON (``--out``), from the arms' ``metrics.csv`` / ``steps.csv`` / ``segments.jsonl``.

Arms. The two hybrid seeds differ ONLY in the train order: ``data_seed`` A or B in the native ``EpochSampler(seed +
epoch)``; same initial weights, config, beta controller, LR schedule. Their spread is the data-order spread at a fixed
initial state. The JAX arm has the order of seed A. ``jax-b`` (order B) is an optional diagnostic that uses the
node's fourth GPU; it does not enter the verdict.

    hybrid-a   hybrid, order A        hybrid-b   hybrid, order B        jax   JAX, order A        jax-b   JAX, order B

G4 config (``g4_overrides``). The production config's epoch is 7769 optimizer steps and BetaPID engages after 2
epochs, so within K = 5000 steps the controller would never act and "EBOPs track the setpoint" could not be tested.
G4 therefore uses epochs of ``--epoch-steps`` (500) optimizer steps (``limit_train_batches`` = 500 x accumulation, each
epoch a fresh random subset of the train set), ``max_epochs`` = K / 500 = 10, BetaPID warm-up 2 epochs (8 controller
updates), validation (``num_val`` 5000) at every epoch end, a step checkpoint every ``--ckpt-every`` (96) steps in the
hybrid and ``--jax-ckpt-every`` (200) in the JAX driver (the newest kept) and a train row every ``--log-every`` (10)
steps. Everything else is the production config: batch 32 x accumulation 4, LR max 5e-5 / clip 0.1 with no
rescaling (the 2026-10-14 fix), PeriodicEBOPs every 8, the provisional BetaPID setpoints. Both frameworks read the same YAML.

Hybrid checkpoint steps must not fall on an epoch's last step (``check_ckpt_steps``; 96 and 500 first coincide at 12000).
Measured locally (Lightning 2.5.2): a resume from a step checkpoint written on the last batch of an epoch marks that
epoch complete without running its validation (``FitLoop.reset``: ``restarted_on_train_batch_end and
restarted_mid_epoch and is_last_batch``), and the restart stage, still set, skips ``on_train_epoch_start`` of the next
epoch: BetaPID did not step (beta stayed at init_beta, steps.csv after the resume). A mid-epoch checkpoint resumes
bit for bit (loss, LR, beta). The native driver has no such case (it resumed bit for bit across an epoch boundary).

Verdict (``verdict``). Preconditions (else INVALID): every gate arm has its validation row at K; the LR of every
matched logged update agrees between JAX and hybrid A (``LR_RTOL``); the converted weights reproduce the hybrid's mask
and class logits on the init batch (``FORWARD_RTOL`` of max |logit|). Gate 1, validation loss at K: the plan asks for the
JAX value "within the spread of two hybrid seeds". With two exchangeable runs X1, X2 and a third X3 from the same
normal distribution, P(X3 in [min, max]) = 1/3: the literal rule fails a correct port two times in three. So the band
is M +- c |D| (M, D the mean and difference of the hybrids); (X3 - M) / |D| = sqrt(3/4) T with T standard Cauchy, so
P(inside) = (2 / pi) atan(c / sqrt(3/4)); c = 5.47 gives 0.90 (``coverage``). The strict [min, max] result is reported
next to it. Gate 2, EBOPs: the JAX log10(EBOPs / setpoint) at K is inside the same band of the hybrids' values (half-width at least
``EBOPS_FLOOR_DEX``), or no farther from the setpoint than the farther hybrid. Reported: every matched validation step, beta and EBOPs
trajectories, early train-loss agreement (JAX vs hybrid A against hybrid A vs B: same weights and order give a small
ratio), and throughput (median wall ms per micro-batch of the logged windows, and per segment).

Local check (CPU, tiny sizes, dummy events; both envs):
    python polaris/19_g4_compare.py init --tiny --root /tmp/g4 --tag t --hybrid-python ../hepattn/.venv/bin/python \\
        --jax-python .venv-jax/bin/python
    python polaris/19_g4_compare.py run --root /tmp/g4 --tag t --arm hybrid-a   (... each arm)
    python polaris/19_g4_compare.py report --root /tmp/g4 --tag t --out /tmp/g4_t.json
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jax_common import REPO, generate_config, shrink_config

ARMS = {"hybrid-a": ("hybrid", "a"), "hybrid-b": ("hybrid", "b"), "jax": ("jax", "a"), "jax-b": ("jax", "b")}
GATE_ARMS = ("hybrid-a", "hybrid-b", "jax")
TINY = {"dim": 16, "heads": 2, "enc_layers": 1, "dec_layers": 1, "linformer_k": 8, "nodes": 64, "queries": 16}
C_DEFAULT = 5.47  # band half-width in |D|: 90% coverage of an exchangeable third run (module docstring)
LR_RTOL = 1e-4  # native schedule in float32 (~6e-8 relative), torch OneCycleLR in float64
# Converted weights vs the hybrid on the init batch, |diff| / max|logit| per mask / class logit output. A quantized
# network turns float ulps into whole LSB steps: measured 0.0115 / 2.24 = 5e-3 (final mask logits, tiny CPU init, one
# datalane step 2^-8 x 3), 0 for the class logits; a mis-mapped tensor gives O(1): S7 controls 1.20 (one weight bit
# less) and 10.6 (unconverted model) on mask logits of max ~2. 0.05 sits 10x from both.
FORWARD_RTOL = 0.05
DEGENERATE = 1e-6  # |D| / |M| below this: the two hybrids coincide and the band is empty -> inconclusive
# EBOPs band floor, in log10(EBOPs / setpoint): 0.01 dex (2.3%), 1/22 of the initial distance to the setpoint
# (log10(2.511e11 / 1.5e11) = 0.224, 03_make_config.py). EBOPs move in discrete bit steps, so two hybrids can coincide
# (tiny CPU run: spread 2.8e-6 dex, JAX 2.6e-6 dex from their mean): without a floor the band is empty.
EBOPS_FLOOR_DEX = 0.01
KILL_EXIT = 75
HYBRID_SCRIPT = Path(__file__).resolve().parent / "19_g4_hybrid.py"
JAX_KILL_CHILD = (
    "import os, sys\n"
    "from hepattn.keras_native import checkpoint as C\n"
    "at = int(os.environ['G4_KILL_AT_STEP'])\n"
    "orig = C.CheckpointManager.save\n"
    "def save(self, step, *a, **k):\n"
    "    out = orig(self, step, *a, **k)\n"
    "    if step >= at:\n"
    "        print(f'[G4] simulated kill at step {step}', flush=True)\n"
    "        os._exit(75)\n"
    "    return out\n"
    "C.CheckpointManager.save = save\n"
    "from hepattn.experiments.clic.main_jax import main\n"
    "sys.exit(main(sys.argv[1:]))\n"
)


# --------------------------------------------------------------------------------------------- small helpers
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, float]]:
    """A CSV as floats (empty -> NaN); missing file -> []."""
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return [{k: float(v) if v not in {"", None} else math.nan for k, v in r.items()} for r in csv.DictReader(f)]


def by_step(rows: list[dict[str, float]]) -> dict[int, dict[str, float]]:
    """Rows keyed by step; a later row of the same step wins (a resumed run rewrites the rows after its checkpoint)."""
    return {int(r["step"]): r for r in rows if "step" in r and not math.isnan(r["step"])}


def env_with_src(**extra: str) -> dict:
    return {**os.environ, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", "PYTHONNOUSERSITE": "1", **extra}


def run_logged(cmd: list[str], env: dict, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        f.flush()
        return subprocess.run([str(c) for c in cmd], env=env, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=False).returncode


# --------------------------------------------------------------------------------------------- config
def g4_overrides(cfg: dict, s: dict, ckpt_dir: Path) -> dict:
    """The G4 settings ``s`` written into a generated config (module docstring)."""
    t, d = cfg["trainer"], cfg["data"]
    t["max_epochs"] = s["k"] // s["epoch_steps"]
    t["limit_train_batches"] = s["epoch_steps"] * s["accum"]
    t["log_every_n_steps"] = s["log_every"]
    t["check_val_every_n_epoch"] = 1
    t["num_sanity_val_steps"] = 0
    d["num_workers"] = s["workers"]
    d["num_train"] = s["num_train"]
    d["num_val"] = s["num_val"]
    for cb in t["callbacks"]:
        name = cb.get("class_path", "")
        if name == "lightning.pytorch.callbacks.ModelCheckpoint":
            cb["init_args"].update(dirpath=str(ckpt_dir), every_n_train_steps=s["ckpt_every"], save_top_k=1, save_last=True)
        if name.endswith("BetaPID"):
            cb["init_args"]["warmup_epochs"] = s["pid_warmup_epochs"]
    return cfg


def make_config(s: dict, init_dir: Path) -> Path:
    import yaml  # noqa: PLC0415

    env = {
        "HGQ_PRECISION": "32",
        "HGQ_DEVICES": "1",
        "HGQ_BATCH": str(s["batch"]),
        "HGQ_ACCUM": str(s["accum"]),
        "HGQ_PID_WARMUP_EPOCHS": str(s["pid_warmup_epochs"]),
        "CKPT_DIR": str(init_dir / "unused_ckpts"),
    }
    if s["data_dir"]:
        env["DATA_DIR"] = s["data_dir"]
    path = generate_config(init_dir / "generated.yaml", **env)
    cfg = g4_overrides(yaml.safe_load(path.read_text()), s, init_dir / "unused_ckpts")
    if s["tiny"]:
        cfg = shrink_config(cfg, **TINY)
        cfg["trainer"]["accelerator"] = "cpu"
        cfg["data"]["dummy_data"] = True
        # a larger LR, so that a few dummy-data steps move the loss and the beta controller (as 16_jax_smoke --tiny)
        cfg["model"]["lrs_config"].update(initial=2e-4, max=2e-3, end=2e-5, pct_start=0.3)
    out = init_dir / "config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def arm_hybrid_config(manifest: dict, arm_dir: Path, seed: int) -> Path:
    """The G4 config with this arm's checkpoint / metrics paths and data seed (the hybrid reads paths from the YAML)."""
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(Path(manifest["config"]).read_text())
    for cb in cfg["trainer"]["callbacks"]:
        name = cb.get("class_path", "")
        if name == "lightning.pytorch.callbacks.ModelCheckpoint":
            cb["init_args"]["dirpath"] = str(arm_dir / "ckpts")
        if name.endswith("MetricsCsvWriter"):
            cb["init_args"]["path"] = str(arm_dir / "metrics.csv")
    cfg["data"]["data_seed"] = seed
    out = arm_dir / "hybrid_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


# --------------------------------------------------------------------------------------------- init
def check_ckpt_steps(k: int, epoch_steps: int, ckpt_every: int) -> list[int]:
    """Hybrid step-checkpoint steps up to K that are an epoch's last step (must be none; module docstring)."""
    return [s for s in range(ckpt_every, k + 1, ckpt_every) if s % epoch_steps == 0]


def settings_from(args) -> dict:
    if args.k % args.epoch_steps:
        raise SystemExit(f"--k {args.k} is not a multiple of --epoch-steps {args.epoch_steps}")
    if bad := check_ckpt_steps(args.k, args.epoch_steps, args.ckpt_every):
        raise SystemExit(
            f"--ckpt-every {args.ckpt_every} writes hybrid checkpoints at epoch ends {bad[:5]}: pick one with no multiple of --epoch-steps up to --k"
        )
    keys = (
        "k",
        "epoch_steps",
        "batch",
        "accum",
        "ckpt_every",
        "jax_ckpt_every",
        "log_every",
        "num_train",
        "num_val",
        "workers",
        "pid_warmup_epochs",
        "seed_a",
        "seed_b",
        "tiny",
    )
    s = {key: getattr(args, key) for key in keys}
    s["data_dir"] = args.data_dir
    return s


def cmd_init(args) -> int:
    tag_dir = Path(args.root) / args.tag
    init_dir = tag_dir / "init"
    manifest_path = tag_dir / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        print(f"init already complete: {manifest_path}")
        return 0
    init_dir.mkdir(parents=True, exist_ok=True)
    s = settings_from(args)
    config = make_config(s, init_dir)
    log = init_dir / "init.log"

    hyb_env = env_with_src(KERAS_BACKEND="torch", G4_INIT_DIR=str(init_dir))
    cmd = [
        args.hybrid_python,
        HYBRID_SCRIPT,
        "init",
        "--config",
        config,
        "--trainer.default_root_dir",
        init_dir / "lightning",
        "--matmul_precision",
        "highest",
    ]
    if run_logged(cmd, hyb_env, log) != 0 or not (init_dir / "hybrid_init.ckpt").is_file():
        raise SystemExit(f"hybrid init failed; see {log}")

    jax_env = env_with_src(KERAS_BACKEND="jax")
    native = init_dir / "native_init.npz"
    cmd = [args.jax_python, "-m", "hepattn.experiments.clic.convert_to_native", "--config", config, "--ckpt", init_dir / "hybrid_init.ckpt"]
    cmd += ["--out", native, "--report", init_dir / "convert_report.json", "--overwrite"]
    if run_logged(cmd, jax_env, log) != 0:
        raise SystemExit(f"conversion failed; see {log}")
    cmd = [args.jax_python, __file__, "native-forward", "--config", config, "--ckpt", native, "--hybrid", init_dir / "hybrid_forward.npz"]
    cmd += ["--out", init_dir / "forward_check.json"]
    if run_logged(cmd, jax_env, log) != 0:
        raise SystemExit(f"native forward check failed; see {log}")

    files = {p.name: sha256(p) for p in sorted(init_dir.iterdir()) if p.is_file() and p.suffix in {".yaml", ".ckpt", ".npz", ".json"}}
    manifest = {
        "tag": args.tag,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "git": git_head(),
        "settings": s,
        "seeds": {"a": s["seed_a"], "b": s["seed_b"]},
        "arms": {arm: {"backend": b, "seed": s[f"seed_{o}"]} for arm, (b, o) in ARMS.items()},
        "config": str(config),
        "init_ckpt": str(init_dir / "hybrid_init.ckpt"),
        "native_init": str(native),
        "files_sha256": files,
        "forward_check": json.loads((init_dir / "forward_check.json").read_text()),
        "target_ebops": target_ebops(config),
        "complete": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("tag", "git", "seeds", "files_sha256", "forward_check")}, indent=1))
    return 0


def git_head() -> str:
    git = shutil.which("git")
    if git is None:
        return ""
    return subprocess.run([git, "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()


def target_ebops(config: Path) -> float | None:
    import yaml  # noqa: PLC0415

    for cb in yaml.safe_load(config.read_text())["trainer"]["callbacks"]:
        if cb.get("class_path", "").endswith("BetaPID"):
            return float(cb["init_args"]["target_ebops"])
    return None


def forward_errors(ref: dict, got: dict) -> dict:
    """Per output: max |native - hybrid| (padded mask-logit slots excluded), the reference scale, flips for booleans."""
    import numpy as np  # noqa: PLC0415

    out = {}
    for key, value in ref.items():
        other = np.asarray(got[key])
        if value.dtype == bool:
            out[key] = {"flips": int((value != other).sum()), "n": int(value.size)}
            continue
        keep = np.abs(value) < 1e30
        if not keep.any():
            continue
        diff = np.abs(value[keep].astype(np.float64) - other[keep].astype(np.float64))
        out[key] = {"max_abs": float(diff.max()), "ref_max_abs": float(np.abs(value[keep]).max())}
    return out


def forward_ok(errors: dict, rtol: float = FORWARD_RTOL) -> bool:
    """The precondition: every mask / class logit within ``rtol`` of its max |value|."""
    logits = {k: v for k, v in errors.items() if "logit" in k and "max_abs" in v}
    return bool(logits) and all(v["max_abs"] <= rtol * max(v["ref_max_abs"], 1.0) for v in logits.values())


def cmd_native_forward(args) -> int:
    """(JAX env) The native model from the converted checkpoint on the hybrid's init batch.

    Raises:
        SystemExit: If the native and hybrid output keys differ.
    """
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    from hepattn.keras_native import checkpoint as nckpt  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    with np.load(args.hybrid) as data:
        inputs = {k.removeprefix("in::"): data[k] for k in data.files if k.startswith("in::")}
        ref = {k.removeprefix("out::"): data[k] for k in data.files if k.startswith("out::")}
    rc = load_config([args.config])
    model = rc.build_model()
    model(inputs, training=False)
    (tv, ntv, _), _meta = nckpt.load_checkpoint(args.ckpt, model, None, expect_config_hash=rc.model_hash)
    nckpt.assign_state(model, None, (tv, ntv, []))
    got = {}

    def walk(tree, prefix=""):
        for key, value in tree.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, f"{name}.")
            else:
                got[name] = np.asarray(value)

    walk(model(inputs, training=False))
    if set(ref) != set(got):
        raise SystemExit(f"output keys differ: {sorted(set(ref) ^ set(got))}")
    errors = forward_errors(ref, got)
    result = {"events": int(next(iter(inputs.values())).shape[0]), "errors": errors, "ok": forward_ok(errors), "device": str(jax.devices()[0])}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("RESULT " + json.dumps(result))
    return 0


# --------------------------------------------------------------------------------------------- run
def load_manifest(root: str, tag: str) -> dict:
    path = Path(root) / tag / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist: run the init stage first")
    manifest = json.loads(path.read_text())
    if not manifest.get("complete"):
        raise SystemExit(f"{path}: init incomplete")
    return manifest


def arm_done(arm_dir: Path, k: int) -> bool:
    """The arm validated at step K (its metrics.csv row) and its last process ended cleanly there (``DONE``)."""
    rows = by_step(read_rows(arm_dir / "metrics.csv"))
    return k in rows and not math.isnan(rows[k].get("val/loss", math.nan)) and (arm_dir / "DONE").is_file()


def cmd_run(args) -> int:
    manifest = load_manifest(args.root, args.tag)
    s = manifest["settings"]
    backend, order = ARMS[args.arm]
    seed = manifest["seeds"][order]
    arm_dir = Path(args.root) / args.tag / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    if arm_done(arm_dir, s["k"]):
        print(f"{args.arm}: done (step {s['k']})")
        return 0
    job = os.environ.get("PBS_JOBID", "local")
    log = arm_dir / "logs" / f"{job}.log"
    if backend == "hybrid":
        last = arm_dir / "ckpts" / "last.ckpt"
        config = arm_hybrid_config(manifest, arm_dir, seed)
        cmd = [
            args.hybrid_python,
            HYBRID_SCRIPT,
            "fit",
            "--config",
            config,
            "--trainer.default_root_dir",
            arm_dir / "lightning",
            "--matmul_precision",
            "highest",
        ]
        env = env_with_src(KERAS_BACKEND="torch", G4_ARM_DIR=str(arm_dir))
        if last.is_file():
            cmd += ["--ckpt_path", last]
        else:
            env["G4_WARM_START"] = manifest["init_ckpt"]
    else:
        last = arm_dir / "ckpts" / "last.json"
        main = ["-c", JAX_KILL_CHILD] if os.environ.get("G4_KILL_AT_STEP") else ["-m", "hepattn.experiments.clic.main_jax"]
        cmd = [
            args.jax_python,
            *main,
            "fit",
            "--config",
            manifest["config"],
            "--run-dir",
            arm_dir,
            "--seed",
            str(seed),
            "--matmul-precision",
            "highest",
        ]
        cmd += ["--compile-cache", args.compile_cache, "--ckpt-every-n-steps", str(s["jax_ckpt_every"])]
        cmd += ["--resume", "last"] if last.is_file() else ["--init-from", manifest["native_init"]]
        if s["tiny"]:
            cmd.append("--dummy-data")
        env = env_with_src(KERAS_BACKEND="jax")
    start, resumed = progress_step(arm_dir), last.is_file()
    t0 = time.time()
    code = run_logged(cmd, env, log)
    wall = time.time() - t0
    end = progress_step(arm_dir)
    if code == 0 and end >= s["k"]:
        (arm_dir / "DONE").write_text(f"{end}\n")
    segment = {
        "job": job,
        "host": platform.node(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "start_step": start,
        "end_step": end,
        "wall_s": wall,
        "returncode": code,
        "resumed": resumed,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with (arm_dir / "segments.jsonl").open("a") as f:
        f.write(json.dumps(segment) + "\n")
    print(json.dumps({"arm": args.arm, **segment}))
    return code


def progress_step(arm_dir: Path) -> int:
    """The latest logged step of an arm (steps.csv, both backends): progress, not the resume point."""
    rows = read_rows(arm_dir / "steps.csv")
    return int(max((r["step"] for r in rows), default=0))


def cmd_status(args) -> int:
    manifest = load_manifest(args.root, args.tag)
    k = manifest["settings"]["k"]
    out = {}
    for arm in ARMS:
        arm_dir = Path(args.root) / args.tag / arm
        if not arm_dir.is_dir():
            continue
        segs = [json.loads(ln) for ln in (arm_dir / "segments.jsonl").read_text().splitlines()] if (arm_dir / "segments.jsonl").is_file() else []
        out[arm] = {"logged_step": progress_step(arm_dir), "k": k, "done": arm_done(arm_dir, k), "segments": len(segs)}
        if segs:
            out[arm]["last_segment"] = segs[-1]
    print(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------------------------- verdict
def coverage(c: float) -> float:
    """P(an exchangeable third run lies in M +- c |D|) under normal noise: (2 / pi) atan(c / sqrt(3/4))."""
    return 2.0 / math.pi * math.atan(c / math.sqrt(0.75))


def band(x: float, a: float, b: float, c: float, floor: float = 0.0) -> dict:
    """``x`` against two reference runs ``a``, ``b``: the strict [min, max] test and the band M +- max(c |D|, floor)."""
    m, d = 0.5 * (a + b), abs(a - b)
    half = max(c * d, floor)
    degenerate = floor <= 0 and d <= DEGENERATE * max(abs(m), 1e-30)
    z = math.inf if d == 0 else (x - m) / d
    return {
        "value": x,
        "ref": [a, b],
        "min": min(a, b),
        "max": max(a, b),
        "mean": m,
        "spread": d,
        "z": z,
        "strict": min(a, b) <= x <= max(a, b),
        "band": [m - half, m + half],
        "inside": (not degenerate) and abs(x - m) <= half,
        "degenerate": degenerate,
    }


def arm_series(arm_dir: Path) -> dict:
    return {
        "metrics": by_step(read_rows(arm_dir / "metrics.csv")),
        "steps": by_step(read_rows(arm_dir / "steps.csv")),
        "segments": [json.loads(ln) for ln in (arm_dir / "segments.jsonl").read_text().splitlines()]
        if (arm_dir / "segments.jsonl").is_file()
        else [],
    }


def throughput(series: dict, accum: int) -> dict:
    """Median wall ms per micro-batch of the logged windows (each spans log_every steps; windows with a validation or
    a process start are the slow tail, so the median is the training rate), and the end-to-end rate per segment.
    """
    ms = [r["time/ms_per_micro_batch"] for r in series["steps"].values() if not math.isnan(r.get("time/ms_per_micro_batch", math.nan))]
    segs = [
        {**s, "s_per_micro_batch": s["wall_s"] / ((s["end_step"] - s["start_step"]) * accum)}
        for s in series["segments"]
        if s["end_step"] > s["start_step"]
    ]
    total_wall = sum(s["wall_s"] for s in series["segments"])
    return {
        "median_ms_per_micro_batch": statistics.median(ms) if ms else None,
        "windows": len(ms),
        "segments": segs,
        "wall_s_total": total_wall,
    }


def verdict(arms: dict[str, dict], *, k: int, accum: int, target: float | None, forward: dict | None, c: float = C_DEFAULT) -> dict:
    """The G4 verdict from the arms' series (``arm_series``); see the module docstring."""
    out: dict = {"k": k, "c": c, "coverage": coverage(c), "coverage_strict": coverage(0.5)}
    missing = [
        arm for arm in GATE_ARMS if arm not in arms or k not in arms[arm]["metrics"] or math.isnan(arms[arm]["metrics"][k].get("val/loss", math.nan))
    ]
    out["throughput"] = {arm: throughput(series, accum) for arm, series in arms.items()}
    if missing:
        out["status"] = "INCOMPLETE"
        out["missing"] = missing
        out["progress"] = {arm: max(series["steps"], default=0) for arm, series in arms.items()}
        return out
    ha, hb, jx = (arms[a] for a in GATE_ARMS)

    # preconditions
    pre: dict = {}
    common = sorted(set(jx["steps"]) & set(ha["steps"]))
    lr = [abs(jx["steps"][s]["train/lr"] / ha["steps"][s]["train/lr"] - 1) for s in common if ha["steps"][s]["train/lr"] > 0]
    pre["lr_steps_compared"] = len(lr)
    pre["lr_max_rel_diff"] = max(lr) if lr else None
    pre["lr_ok"] = bool(lr) and max(lr) <= LR_RTOL
    pre["forward_ok"] = bool(forward) and forward_ok(forward["errors"])  # recomputed: the tolerance is this file's
    out["preconditions"] = pre

    # gate 1: validation loss at K, and the same test at every matched validation step
    steps = sorted(set(ha["metrics"]) & set(hb["metrics"]) & set(jx["metrics"]))
    trajectory = []
    for s in steps:
        vals = [arm["metrics"][s].get("val/loss", math.nan) for arm in (jx, ha, hb)]
        if any(math.isnan(v) for v in vals):
            continue
        trajectory.append({"step": s, **band(vals[0], vals[1], vals[2], c)})
    out["val_loss"] = {
        "at_k": band(jx["metrics"][k]["val/loss"], ha["metrics"][k]["val/loss"], hb["metrics"][k]["val/loss"], c),
        "trajectory": trajectory,
    }
    out["val_loss"]["inside_fraction"] = sum(t["inside"] for t in trajectory) / len(trajectory) if trajectory else None

    # gate 2: EBOPs vs setpoint
    ebops = {"target": target}
    if target:

        def err(arm, s):
            e = arm["metrics"][s].get("val/ebops", math.nan)
            return math.log10(e / target) if e > 0 else math.nan

        traj = [
            {
                "step": s,
                **{f"{name}_log10_ebops_over_target": err(arm, s) for name, arm in zip(GATE_ARMS, (ha, hb, jx), strict=True)},
                **{f"{name}_beta": arm["metrics"][s].get("train/quant_beta", math.nan) for name, arm in zip(GATE_ARMS, (ha, hb, jx), strict=True)},
            }
            for s in steps
        ]
        e_j, e_a, e_b = err(jx, k), err(ha, k), err(hb, k)
        if any(math.isnan(e) for e in (e_j, e_a, e_b)):
            ebops.update(ok=False, reason="no val/ebops at K")
        else:
            at_k = band(e_j, e_a, e_b, c, floor=EBOPS_FLOOR_DEX)
            closer = abs(e_j) <= max(abs(e_a), abs(e_b))
            ebops.update(at_k=at_k, no_farther_than_worst_hybrid=closer, ok=at_k["inside"] or closer)
        ebops["trajectory"] = traj
    else:
        ebops.update(ok=False, reason="no BetaPID target in the config")
    out["ebops"] = ebops

    # early agreement: same weights and order (JAX vs hybrid A) against another order (hybrid A vs B)
    early = []
    for s in sorted(set(jx["steps"]) & set(ha["steps"]) & set(hb["steps"]))[:5]:
        j, a, b = (arm["steps"][s]["train/loss"] for arm in (jx, ha, hb))
        early.append({"step": s, "jax_minus_a": j - a, "a_minus_b": a - b, "ratio": abs(j - a) / abs(a - b) if a != b else math.inf})
    out["early_agreement"] = early

    if "jax-b" in arms and k in arms["jax-b"]["metrics"]:
        out["jax_b_diagnostic"] = band(arms["jax-b"]["metrics"][k]["val/loss"], ha["metrics"][k]["val/loss"], hb["metrics"][k]["val/loss"], c)

    hy = out["throughput"]["hybrid-a"]["median_ms_per_micro_batch"]
    jm = out["throughput"]["jax"]["median_ms_per_micro_batch"]
    out["speedup_median_window"] = hy / jm if hy and jm else None

    v = out["val_loss"]["at_k"]
    if not (pre["lr_ok"] and pre["forward_ok"]):
        out["status"] = "INVALID"
    elif v["degenerate"]:
        out["status"] = "INCONCLUSIVE"
    else:
        out["status"] = "PASS" if v["inside"] and ebops.get("ok") else "FAIL"
    out["strict_min_max_at_k"] = v["strict"]
    return out


def cmd_report(args) -> int:
    manifest = load_manifest(args.root, args.tag)
    s = manifest["settings"]
    tag_dir = Path(args.root) / args.tag
    arms = {arm: arm_series(tag_dir / arm) for arm in ARMS if (tag_dir / arm).is_dir()}
    result = verdict(arms, k=s["k"], accum=s["accum"], target=manifest.get("target_ebops"), forward=manifest.get("forward_check"), c=args.c)
    result["tag"] = args.tag
    result["manifest"] = {k: manifest[k] for k in ("git", "seeds", "settings", "files_sha256", "created")}
    out = Path(args.out) if args.out else REPO / "logs" / f"g4_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    v = result.get("val_loss", {}).get("at_k", {})
    print(f"G4 {args.tag}: {result['status']}  (written to {out})")
    if v:
        print(
            f"  val/loss at K={s['k']}: jax {v['value']:.5f}  hybrids {v['ref'][0]:.5f} / {v['ref'][1]:.5f}"
            f"  band [{v['band'][0]:.5f}, {v['band'][1]:.5f}] (c={args.c}, coverage {coverage(args.c):.2f})"
            f"  strict [min,max]: {v['strict']}  z={v['z']:.2f}"
        )
    if "ebops" in result and "at_k" in result["ebops"]:
        e = result["ebops"]["at_k"]
        print(f"  log10(EBOPs/target) at K: jax {e['value']:.4f}  hybrids {e['ref'][0]:.4f} / {e['ref'][1]:.4f}  ok {result['ebops']['ok']}")
    print(f"  speedup (median window): {result.get('speedup_median_window')}")
    return 0 if result["status"] in {"PASS", "INCOMPLETE"} else 1


# --------------------------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(q):
        q.add_argument("--root", required=True, help="parent of the run directories (Polaris: $WORK_ROOT/g4)")
        q.add_argument("--tag", required=True)

    init = sub.add_parser("init")
    common(init)
    init.add_argument("--data-dir", default="")
    init.add_argument("--tiny", action="store_true", help="tiny sizes, dummy events, CPU")
    init.add_argument("--k", type=int, default=5000)
    init.add_argument("--epoch-steps", type=int, default=500)
    init.add_argument("--batch", type=int, default=32)
    init.add_argument("--accum", type=int, default=4)
    init.add_argument("--ckpt-every", type=int, default=96, help="hybrid step checkpoints (never on an epoch end)")
    init.add_argument("--jax-ckpt-every", type=int, default=200, help="JAX step checkpoints")
    init.add_argument("--log-every", type=int, default=10)
    init.add_argument("--num-train", type=int, default=-1, help="train events loaded (-1: all, the production config)")
    init.add_argument("--num-val", type=int, default=5000)
    init.add_argument("--workers", type=int, default=4, help="DataLoader workers per run (4 runs share a node's 32 cores)")
    init.add_argument("--pid-warmup-epochs", type=int, default=2)
    init.add_argument("--seed-a", type=int, default=42, help="data order A (hybrid-a, jax); the config's seed_everything")
    init.add_argument("--seed-b", type=int, default=1_000_042, help="data order B (hybrid-b, jax-b); far from A + epoch")
    init.add_argument("--hybrid-python", default=sys.executable)
    init.add_argument("--jax-python", default=sys.executable)

    run = sub.add_parser("run")
    common(run)
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--hybrid-python", default=sys.executable)
    run.add_argument("--jax-python", default=sys.executable)
    run.add_argument("--compile-cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))

    status = sub.add_parser("status")
    common(status)

    report = sub.add_parser("report")
    common(report)
    report.add_argument("--out", default="")
    report.add_argument(
        "--band",
        "--c",
        dest="c",
        type=float,
        default=C_DEFAULT,
        help="band half-width c in units of |hybrid A - hybrid B| (5.47: 90%% coverage; 1.5: [min - D, max + D], 67%%; 0.5: strict [min, max], 33%%)",
    )

    fwd = sub.add_parser("native-forward", help="(internal, JAX env) converted weights vs the hybrid init outputs")
    fwd.add_argument("--config", required=True)
    fwd.add_argument("--ckpt", required=True)
    fwd.add_argument("--hybrid", required=True)
    fwd.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return {"init": cmd_init, "run": cmd_run, "status": cmd_status, "report": cmd_report, "native-forward": cmd_native_forward}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
