r"""QAT warm start of the keras_native (JAX) driver from a float Lightning checkpoint: init, smoke, pilot arms, report.

Stages (``22_warmstart.pbs`` runs them on Polaris, ``WS_STAGE``):

``init``    The QAT config (``03_make_config.py`` with ``HGQ_SOURCE_CONFIG`` = the float run's saved config, fp32, one
            device, ``--weight-b0``, the PID setpoints; mini-epochs of ``--epoch-steps`` optimizer steps with a
            validation each, as G4), the converted checkpoint (``convert_to_native --target-config``) and the parity
            gate on the CPU (``21_warmstart_check.py parity``, with one mutant that must fail), then ``manifest.json``.
            The optimizer is not in the model hash, so every arm warm-starts from the same ``init.npz``.
            ``--sat-cal-batches N`` also builds a second model whose datalane quantizers saturate (``SAT``: trainable
            integer bits, no wrap; ``HGQ_DATALANE_OVERFLOW``), converts the checkpoint into it and seeds its integer
            and sign bits from the WRAP model's, traced over the first N train micro-batches (``seed-sat``):
            ``init_sat.npz``, used by the arms with ``"model": "sat"``.
``run``     One arm (``ARMS``: optimizer, peak LR, weight decay, quantizer LR scale and epsilon, BetaPID settings, the
            model) with ``main_jax fit --init-from`` (``--resume last`` when the arm has a checkpoint); ``--max-steps``
            caps it (the smoke). With the init's ``--val-calibration N`` every validation is repeated with the WRAP
            integer bits traced over N train micro-batches (``val_cal/*``). ``DONE`` at K.
``report``  Per arm: the first and last logged train rows, every validation (loss, EBOPs, beta, bits), ms per
            micro-batch. ``--out`` JSON.

    python polaris/22_warmstart.py init --root R --tag T --source-config RUN/config.yaml --ckpt RUN/x.ckpt --data-dir D \
        --model-ebops 4.8e10 --target-ebops 2.9e10 --init-beta 8.5e-11
    python polaris/22_warmstart.py run --root R --tag T --arm adamw [--max-steps 60 --run-name smoke]
    python polaris/22_warmstart.py report --root R --tag T --out logs/ws_T.json

Beta values in ``ARMS`` are what BetaPID writes and ``metrics.csv`` logs (``train/quant_beta``); the EBOPs term acts on
every 8th optimizer step only (PeriodicEBOPs), so the time-averaged pull is beta / 8, and ``--init-beta`` of the init is
in those time-averaged units (``03_make_config.py`` multiplies it by ``HGQ_EBOPS_EVERY``).
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("g4", REPO / "polaris/19_g4_compare.py")
g4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g4)

# Arms: the optimizer recipe; the model and the converted init are shared (optimizer, lrs_config and the beta callbacks
# are not in the model hash), except "model": "sat" (the init's SAT model). "pid": BetaPID init_args (logged units);
# "every": PeriodicEBOPs every_n_steps (the time-averaged pull is the logged beta / every).
ADAMW = {"optimizer": "AdamW", "max": 5e-5, "weight_decay": 1e-4}
# ws2: the clipped bitwidth gradients (~1e-10) sit below AdamW's epsilon 1e-8 (polaris/23_ws_diag.py grads), so the
# quantizer group gets epsilon 1e-12; the pilot ended at beta 8.9e-10 (train/quant_beta of adamw-q20, ws1)
WS2 = {**ADAMW, "quantizer_lr_scale": 20.0, "quantizer_epsilon": 1e-12}
ARMS = {
    "adamw": ADAMW,  # the G4-validated recipe
    "adamw-q20": {**ADAMW, "quantizer_lr_scale": 20.0},
    "lion-6e5": {"optimizer": "Lion", "max": 6.2e-5, "weight_decay": 0.1},  # the small float reference run's recipe at 1/10 of its peak LR
    "lion-6e4": {"optimizer": "Lion", "max": 6.2e-4, "weight_decay": 0.1},  # the small float reference run's recipe
    "ws2-a": {**WS2, "pid": {"init_beta": 8.9e-10}},
    "ws2-b": {**WS2, "pid": {"init_beta": 2.7e-9}},
    "ws2-c": {**WS2, "pid": {"init_beta": 8.9e-10, "i": 2.0}},
    # the EBOPs term on every step at ws2-a's time-averaged pull: with eps 1e-12 a penalty on 1 step in 8 inflates
    # AdamW's v: the replay (23_ws_diag.py replay) moves EBOPs 2.3x further with the term on every step
    "ws2-e1": {**WS2, "every": 1, "pid": {"init_beta": 8.9e-10 / 8}},
    "ws2-sat": {**WS2, "pid": {"init_beta": 8.9e-10}, "model": "sat"},
}
COLUMNS = (
    "train/loss", "train/ebops", "val/loss", "val/ebops", "val_cal/loss", "val_cal/ebops", "train/quant_beta", "train/pid_ebops",
    "val/bits_f_mean", "val/bits_b_mean", "val/bits_i_mean", "time/ms_per_micro_batch",
)  # fmt: skip


def make_config(args, init_dir: Path, name: str = "config", **extra_env: str) -> Path:
    import yaml  # noqa: PLC0415

    env = {
        "HGQ_SOURCE_CONFIG": str(Path(args.source_config).resolve()),
        "HGQ_MODEL_EBOPS": str(args.model_ebops),
        "HGQ_TARGET_EBOPS": str(args.target_ebops),
        "HGQ_INIT_BETA": str(args.init_beta),
        "HGQ_WEIGHT_B0": str(args.weight_b0),
        "HGQ_PRECISION": "32",
        "HGQ_DEVICES": "1",
        "HGQ_BATCH": str(args.batch),
        "HGQ_ACCUM": str(args.accum),
        "HGQ_PID_WARMUP_EPOCHS": str(args.pid_warmup_epochs),
        "DATA_DIR": args.data_dir,
        "CKPT_DIR": str(init_dir / "unused_ckpts"),
        **extra_env,
    }
    path = g4.generate_config(init_dir / f"generated_{name}.yaml" if name != "config" else init_dir / "generated.yaml", **env)
    cfg = yaml.safe_load(path.read_text())
    s = {
        "k": args.k,
        "epoch_steps": args.epoch_steps,
        "accum": args.accum,
        "log_every": args.log_every,
        "workers": args.workers,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "ckpt_every": args.ckpt_every,
        "pid_warmup_epochs": args.pid_warmup_epochs,
    }
    cfg = g4.g4_overrides(cfg, s, init_dir / "unused_ckpts")
    out = init_dir / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def cmd_init(args) -> int:
    root = Path(args.root) / args.tag
    init_dir = root / "init"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        raise SystemExit(f"{manifest_path} is complete; a new init needs a new --tag")
    init_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, init_dir)
    check_onecycle(config, args.k)
    log = init_dir / "init.log"
    cpu = g4.env_with_src(KERAS_BACKEND="jax", JAX_PLATFORMS="cpu")
    convert = [args.jax_python, "-m", "hepattn.experiments.clic.convert_to_native", "--config", args.source_config, "--ckpt", args.ckpt]
    convert += ["--target-config", config, "--out", init_dir / "init.npz", "--report", init_dir / "convert.json", "--overwrite"]
    if g4.run_logged(convert, cpu, log) != 0:
        raise SystemExit(f"conversion failed: {log}")
    parity = [args.jax_python, REPO / "polaris/21_warmstart_check.py", "parity", "--config", args.source_config, "--ckpt", args.ckpt]
    parity += ["--val", f"{args.data_dir}/val_clic_fix.root", "--num-events", "200", "--parity-events", "32", "--threads", "8"]
    parity += ["--qat-config", config, "--mutant", "tensor:register_tokens:1e-3", "--out", init_dir / "parity.json"]
    code = g4.run_logged(parity, cpu, log)
    report = json.loads((init_dir / "parity.json").read_text())
    conv = json.loads((init_dir / "convert.json").read_text())
    manifest = {
        "complete": code == 0,
        "git": g4.git_head(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(config),
        "native_init": str(init_dir / "init.npz"),
        "source": {"config": args.source_config, "ckpt": args.ckpt, "ckpt_sha256": conv["checkpoint"]["sha256"]},
        "settings": {k: v for k, v in vars(args).items() if k not in {"func", "command"}},
        "config_hash": conv["config_hash"],
        "renorm_lane": conv.get("renorm_lane"),
        "parity_ok": report["ok"],
        "parity_worst": {t: e["base"]["gate"] for t, e in report["targets"].items()},
        "files_sha256": {p.name: g4.sha256(p) for p in (config, init_dir / "init.npz", init_dir / "parity.json")},
    }
    if args.sat_cal_batches and manifest["complete"]:
        manifest["sat"] = make_sat(args, init_dir, config, log)
        manifest["complete"] = manifest["sat"]["ok"]
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(json.dumps({"complete": manifest["complete"], "parity_ok": manifest["parity_ok"], "config_hash": manifest["config_hash"]}))
    return 0 if manifest["complete"] else 1


def make_sat(args, init_dir: Path, wrap_config: Path, log: Path) -> dict:
    """The SAT-datalane model: config, conversion (``init_sat_spec.npz``), integer bits seeded from the WRAP model.

    Raises:
        SystemExit: If the conversion fails.
    """
    config = make_config(args, init_dir, name="config_sat", HGQ_DATALANE_OVERFLOW="SAT")
    cpu = g4.env_with_src(KERAS_BACKEND="jax", JAX_PLATFORMS="cpu")
    spec = init_dir / "init_sat_spec.npz"
    convert = [args.jax_python, "-m", "hepattn.experiments.clic.convert_to_native", "--config", args.source_config, "--ckpt", args.ckpt]
    convert += ["--target-config", config, "--out", spec, "--report", init_dir / "convert_sat.json", "--overwrite"]
    if g4.run_logged(convert, cpu, log) != 0:
        raise SystemExit(f"SAT conversion failed: {log}")
    seed = [args.jax_python, REPO / "polaris/22_warmstart.py", "seed-sat", "--wrap-config", wrap_config, "--wrap-init", init_dir / "init.npz"]
    seed += ["--sat-config", config, "--sat-init", spec, "--n", str(args.sat_cal_batches), "--out", init_dir / "init_sat.npz"]
    seed += ["--report", init_dir / "seed_sat.json", "--work-dir", init_dir / "seed_sat_run"]
    code = g4.run_logged(seed, g4.env_with_src(KERAS_BACKEND="jax"), log)
    report = json.loads((init_dir / "seed_sat.json").read_text()) if code == 0 else {}
    print(json.dumps({"sat_seed": report.get("summary")}), flush=True)
    return {
        "ok": code == 0 and bool(report.get("ok")),
        "config": str(config),
        "native_init": str(init_dir / "init_sat.npz"),
        "config_hash": report.get("config_hash"),
        "seed": report.get("summary"),
    }


def cmd_seed_sat(args) -> int:
    """``init_sat.npz``: the SAT model at the converted weights with each datalane quantizer's integer bits (trainable
    here) and sign bits set to the WRAP model's, traced over the first ``--n`` train micro-batches
    (``loop.Trainer.calibrated_ntv``). Variables are matched by their checkpoint key (auto-numbered names stripped).
    Report: the counts, the matched loss and EBOPs of one training-mode forward of each model on the first calibration
    batch (gate: losses within 1e-3 relative).
    """
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from hepattn.experiments.clic.main_jax import ClicSource  # noqa: PLC0415
    from hepattn.keras_native import checkpoint as nckpt  # noqa: PLC0415
    from hepattn.keras_native import loop  # noqa: PLC0415
    from hepattn.keras_native.config import load_config  # noqa: PLC0415
    from hepattn.keras_native.ebops import EbopsLeaves  # noqa: PLC0415
    from hepattn.keras_native.objective import matched_loss  # noqa: PLC0415

    rc = load_config([args.wrap_config])
    train = ClicSource(rc, "train", seed=42)
    sample = next(iter(train.epoch(0)))
    t = loop.Trainer(rc, loop.LoopSettings.from_run_config(rc, args.work_dir), sample=sample, train_batches=train.num_batches)
    t.warm_start(args.wrap_init)
    t._set_beta(1.0)  # noqa: SLF001  (EBOPs read-out = penalty)
    ntv, ebops_wrap_traced = t.calibrated_ntv(train, args.n)
    wrap = t.steps.model
    keys = nckpt.variable_keys(wrap)
    i_pos, k_pos = t._wrap_index  # noqa: SLF001
    # by position, not by name: a repeated key ends in "#n" (checkpoint._numbered)
    role = {keys[id(wrap.non_trainable_variables[p])]: r for r, ps in (("i", i_pos), ("k", k_pos)) for p in ps}
    traced = {keys[id(wrap.non_trainable_variables[p])]: np.asarray(ntv[p]) for p in (*i_pos, *k_pos)}
    untraced = [k for k, r in role.items() if r == "i" and float(traced[k].min()) <= -1e8]

    rc_s = load_config([args.sat_config])
    model = rc_s.build_model()
    model(sample.inputs, training=False)
    (tv, ntv_s, _), meta = nckpt.load_checkpoint(args.sat_init, model, None, expect_config_hash=rc_s.model_hash)
    keys_s = nckpt.variable_keys(model)
    tv, ntv_s, set_i, still_wrap, set_k = list(tv), list(ntv_s), 0, 0, 0
    for group, variables in ((tv, model.trainable_variables), (ntv_s, model.non_trainable_variables)):
        for n, var in enumerate(variables):
            key = keys_s[id(var)]
            if role.get(key) == "i":  # SAT: trainable; a quantizer that stays WRAP (not a datalane): set from the data anyway
                group[n] = np.clip(traced[key], -23, 23).astype(group[n].dtype)  # kif datalane ic = MinMax(-23, 23)
                set_i, still_wrap = set_i + var.trainable, still_wrap + (not var.trainable)
            elif role.get(key) == "k":
                group[n] = traced[key].astype(group[n].dtype)
                set_k += 1
    meta = {**meta, "seeded_from": {"wrap_init": str(args.wrap_init), "calibration_batches": args.n}}
    nckpt.save_checkpoint(args.out, model, None, (tv, ntv_s, []), meta, overwrite=True)

    def forward(m, tv_, ntv_):
        leaves = EbopsLeaves(m)

        @jax.jit  # eagerly, Keras' _clear_losses matches EBOPs terms by id() of concrete arrays (reused ids)
        def run(tv_, ntv_):
            with leaves.enabled(True):
                out, _, losses = m.stateless_call(tv_, ntv_, sample.inputs, training=True, return_losses=True)
            return matched_loss(out, sample.targets, rc.task_specs, n_jobs=rc.matcher_n_jobs)[0], leaves.split_losses(losses)[0]

        loss, terms = run(tv_, leaves.set_beta(ntv_, 1.0))
        return float(loss), float(sum(np.asarray(p, np.float64) for p in terms))

    # the first train batch is a calibration batch: inside the SAT ranges, so the SAT forward is the WRAP one without
    # overflow (WRAP lanes only round in training) and the losses must agree. Both at the traced bits: the kbi WRAP
    # quantizers (biases) keep b, so their stored i (the spec's i0 in the converted init) sets their fraction bits
    loss_sat, ebops_sat = forward(model, [jax.numpy.asarray(x) for x in tv], ntv_s)
    loss_wrap, ebops_wrap = forward(wrap, t.state.tv, ntv)
    summary = {
        "wrap_quantizers": len(i_pos), "untraced": len(untraced), "set_i_sat": set_i, "set_i_still_wrap": still_wrap, "set_k": set_k,
        "loss_train_mode_wrap": loss_wrap, "loss_train_mode_sat": loss_sat, "ebops_wrap": ebops_wrap, "ebops_sat": ebops_sat,
        "ebops_wrap_traced": ebops_wrap_traced,
    }  # fmt: skip
    ok = set_i > 0 and set_i + still_wrap == len(i_pos) == set_k and not untraced and abs(loss_sat - loss_wrap) <= 1e-3 * abs(loss_wrap)
    Path(args.report).write_text(json.dumps({"ok": ok, "config_hash": rc_s.model_hash, "summary": summary, "untraced": untraced}, indent=1) + "\n")
    print(json.dumps({"ok": ok, **summary}), flush=True)
    return 0 if ok else 1


def check_onecycle(config: Path, total_steps: int) -> None:
    """OneCycle needs pct_start x total steps > 1 (``optim.OneCycleLR``, as torch); fail before any job time is spent.

    Raises:
        SystemExit: If the run is too short for the config's ``pct_start``.
    """
    import yaml  # noqa: PLC0415

    pct = float(yaml.safe_load(config.read_text())["model"]["lrs_config"]["pct_start"])
    if pct * total_steps <= 1:
        raise SystemExit(f"pct_start {pct} x {total_steps} steps <= 1: OneCycle needs more steps (> {1 / pct:.0f})")


def arm_model(manifest: dict, arm: str) -> dict:
    """``{"config", "native_init"}`` of the model an arm trains (the init's, or its SAT model).

    Raises:
        SystemExit: For a SAT arm when the init built no SAT model.
    """
    if ARMS[arm].get("model") == "sat":
        if "sat" not in manifest:
            raise SystemExit(f"arm {arm} needs the SAT model: run the init with --sat-cal-batches")
        return manifest["sat"]
    return manifest


def arm_config(manifest: dict, arm: str, arm_dir: Path) -> Path:
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(Path(arm_model(manifest, arm)["config"]).read_text())
    recipe = ARMS[arm]
    cfg["model"]["optimizer"] = recipe["optimizer"]
    lrs = cfg["model"]["lrs_config"]
    lrs["max"], lrs["weight_decay"] = recipe["max"], recipe["weight_decay"]
    for key in ("quantizer_lr_scale", "quantizer_epsilon"):
        if key in recipe:
            lrs[key] = recipe[key]
    for key, name, update in (("pid", "BetaPID", recipe.get("pid")), ("every", "PeriodicEBOPs", {"every_n_steps": recipe.get("every")})):
        if key not in recipe:
            continue
        found = [cb for cb in cfg["trainer"]["callbacks"] if cb.get("class_path", "").endswith(name)]
        if len(found) != 1:
            raise SystemExit(f"arm {arm} sets {name} arguments but the config has {len(found)} {name} callbacks")
        found[0].setdefault("init_args", {}).update(update)
    out = arm_dir / "arm_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def cmd_run(args) -> int:
    root = Path(args.root) / args.tag
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit("the init is not complete (manifest.json): run the init stage first")
    arm_dir = root / (args.run_name or args.arm)
    arm_dir.mkdir(parents=True, exist_ok=True)
    k = args.max_steps or manifest["settings"]["k"]
    if (arm_dir / "DONE").is_file():
        print(json.dumps({"arm": args.arm, "done": True}))
        return 0
    config = arm_config(manifest, args.arm, arm_dir)
    check_onecycle(config, k)
    cmd = [args.jax_python, "-m", "hepattn.experiments.clic.main_jax", "fit", "--config", config, "--run-dir", arm_dir]
    cmd += ["--seed", "42", "--matmul-precision", "highest", "--compile-cache", args.compile_cache]
    cmd += ["--ckpt-every-n-steps", str(manifest["settings"]["ckpt_every"])]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.num_val:
        cmd += ["--num-val", str(args.num_val)]
    if manifest["settings"].get("val_calibration"):
        cmd += ["--val-calibration", str(manifest["settings"]["val_calibration"])]
    last = arm_dir / "ckpts" / "last.json"
    cmd += ["--resume", "last"] if last.is_file() else ["--init-from", arm_model(manifest, args.arm)["native_init"]]
    t0 = time.time()
    code = g4.run_logged(cmd, g4.env_with_src(KERAS_BACKEND="jax"), arm_dir / "logs" / f"{os.environ.get('PBS_JOBID', 'local')}.log")
    end = g4.progress_step(arm_dir)
    if code == 0 and end >= k:
        (arm_dir / "DONE").write_text(f"{end}\n")
    segment = {"job": os.environ.get("PBS_JOBID", "local"), "end_step": end, "wall_s": time.time() - t0, "returncode": code}
    with (arm_dir / "segments.jsonl").open("a") as f:
        f.write(json.dumps(segment) + "\n")
    print(json.dumps({"arm": args.arm, **segment}))
    return code


def pick(row: dict) -> dict:
    return {c: row[c] for c in COLUMNS if c in row and not (isinstance(row[c], float) and math.isnan(row[c]))}


def cmd_report(args) -> int:
    root = Path(args.root) / args.tag
    out = {"tag": args.tag, "arms": {}}
    for arm_dir in sorted(p for p in root.iterdir() if (p / "steps.csv").is_file()):
        steps, metrics = g4.read_rows(arm_dir / "steps.csv"), g4.read_rows(arm_dir / "metrics.csv")
        ms = [r["time/ms_per_micro_batch"] for r in steps if not math.isnan(r.get("time/ms_per_micro_batch", math.nan))]
        out["arms"][arm_dir.name] = {
            "steps_logged": len(steps),
            "first_train": {"step": steps[0].get("step"), **pick(steps[0])} if steps else None,
            "last_train": {"step": steps[-1].get("step"), **pick(steps[-1])} if steps else None,
            "validations": [{"step": r.get("step"), "epoch": r.get("epoch"), **pick(r)} for r in metrics],
            "median_ms_per_micro_batch": statistics.median(ms) if ms else None,
            "nonfinite_train_loss": sum(1 for r in steps if not math.isfinite(r.get("train/loss", 0.0))),
            "done": (arm_dir / "DONE").is_file(),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for name, a in out["arms"].items():
        last_val = a["validations"][-1] if a["validations"] else {}
        print(
            f"{name:10s} steps {a['last_train']['step'] if a['last_train'] else 0:>6} first train/loss "
            f"{(a['first_train'] or {}).get('train/loss', float('nan')):.4f} last val/loss {last_val.get('val/loss', float('nan')):.4f} "
            f"val/ebops {last_val.get('val/ebops', float('nan')):.3e} ms/micro-batch {a['median_ms_per_micro_batch']}"
            f" nonfinite {a['nonfinite_train_loss']}"
        )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="22_warmstart", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--tag", required=True)
        if name in {"init", "run"}:
            p.add_argument("--jax-python", default=sys.executable)
    seed = sub.add_parser("seed-sat", help="(called by init) the SAT model's integer bits from a traced WRAP model")
    for key in ("--wrap-config", "--wrap-init", "--sat-config", "--sat-init", "--out", "--report", "--work-dir"):
        seed.add_argument(key, required=True)
    seed.add_argument("--n", type=int, required=True, help="train micro-batches traced")
    init = sub.choices["init"]
    init.add_argument("--source-config", required=True, help="the float run's saved config.yaml")
    init.add_argument("--ckpt", required=True, help="its Lightning checkpoint")
    init.add_argument("--data-dir", required=True)
    init.add_argument("--model-ebops", type=float, required=True, help="EBOPs of the QAT model at the converted weights")
    init.add_argument("--target-ebops", type=float, required=True)
    init.add_argument("--init-beta", type=float, required=True)
    init.add_argument("--weight-b0", type=int, default=12)
    init.add_argument("--batch", type=int, default=128)
    init.add_argument("--accum", type=int, default=1)
    init.add_argument("--k", type=int, default=2000, help="optimizer steps per arm")
    init.add_argument("--epoch-steps", type=int, default=250, help="optimizer steps per mini-epoch (one validation each)")
    init.add_argument("--pid-warmup-epochs", type=int, default=2)
    init.add_argument("--log-every", type=int, default=10)
    init.add_argument("--ckpt-every", type=int, default=200)
    init.add_argument("--workers", type=int, default=4)
    init.add_argument("--num-train", type=int, default=400000)
    init.add_argument("--num-val", type=int, default=5000)
    init.add_argument("--val-calibration", type=int, default=0, help="train micro-batches traced for val_cal/* (0: none)")
    init.add_argument("--sat-cal-batches", type=int, default=0, help="also build the SAT-datalane model, seeded from N batches")
    run = sub.choices["run"]
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--run-name", help="run directory name (default: the arm)")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--num-val", type=int)
    run.add_argument("--compile-cache", default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""))
    sub.choices["report"].add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return {"init": cmd_init, "run": cmd_run, "report": cmd_report, "seed-sat": cmd_seed_sat}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
