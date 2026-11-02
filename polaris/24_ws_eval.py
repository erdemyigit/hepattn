#!/usr/bin/env python
r"""CLIC physics evaluation of native QAT checkpoints against the float model, on the same test events (CPU or GPU).

``predict``  The prediction h5 and ROOT file (``keras_native`` ``Trainer.test`` + ``predictionwriter.write_root``, the
             layout ``performance`` reads) of one model on ``--test-path`` (the inference file, ``is_inference``):
             - a QAT checkpoint (``--ckpt``) with the WRAP integer bits traced over the first ``--cal-batches`` train
               micro-batches (``Trainer.calibrated_ntv``: the ``val_cal`` bits of training, what ``trace_minmax`` fixes
               for a deployment); ``--cal-batches 0`` keeps the bits that training left (``val/*``);
             - the float model (``--init`` = the converted init, ``--float``): the same graph with every HGQ2
               Quantizer the identity (the W2 parity gate checked it against torch: 0.26x of its tolerance).
``jets``     ``performance.Performance`` on the ROOT files and the truth / Pandora file (``--truth``): jets clustered and
             matched to truth jets as the decoder-k study (``studies/linformer_dim/plot_jet_iqr.py`` on org main:
             ``dr_cut 0.1, leading_n_jets 2, pt_min 10``), jet-E response median and IQR, global and per truth-E bin,
             in the ``mpflow_proxy`` (default) and ``mpflow`` conventions; particle efficiency and fake rate after the
             particle matching (``hung_match_particles``), per class and for neutral hadrons / photons per pt bin
             (``plot_helper_particle.plot_eff_fr_purity`` definitions). Every number is computed on the events that all
             networks share (``reorder_and_find_intersection``). Uncertainties: bootstrap over matched jets (``--boot``)
             and binomial for the rates; they measure the finite test sample, not training-to-training spread.

    KERAS_BACKEND=jax PYTHONPATH=src python polaris/24_ws_eval.py predict --config ARM/arm_config.yaml \
        --ckpt ARM/ckpts/step-000020000.npz --cal-batches 256 --train-path TRAIN.root --test-path INFER.root --out ws2-a.h5
    KERAS_BACKEND=jax PYTHONPATH=src python polaris/24_ws_eval.py predict --config INIT/config.yaml --init INIT/init.npz \
        --float --train-path TRAIN.root --test-path INFER.root --out float.h5
    python polaris/24_ws_eval.py jets --truth RAW.root --net float=float.root --net ws2-a=ws2-a.root --ref float --out jets.json

The float checkpoint was trained with the data reader before org PR #5 (pre-#5 topo cos/sin columns); this branch has
that reader, which is what the evaluation needs.
"""

import argparse
import contextlib
import dataclasses
import itertools
import json
import os
import sys
import time
from pathlib import Path


@contextlib.contextmanager
def float_mode():
    """Every HGQ2 ``Quantizer`` the identity (as ``21_warmstart_check.py``); active while the eval step is traced."""
    from hgq.quantizer import Quantizer  # noqa: PLC0415

    original = Quantizer.call
    Quantizer.call = lambda self, x, training=None: x  # noqa: ARG005
    try:
        yield
    finally:
        Quantizer.call = original


def cmd_predict(args) -> dict:
    import jax  # noqa: PLC0415

    from hepattn.experiments.clic.main_jax import ClicSource  # noqa: PLC0415
    from hepattn.experiments.clic.predictionwriter import write_root  # noqa: PLC0415
    from hepattn.keras_native import loop  # noqa: PLC0415
    from hepattn.keras_native.config import load_config, resolve_file  # noqa: PLC0415

    if (args.ckpt is None) == (args.init is None):
        raise SystemExit("give exactly one of --ckpt (a QAT checkpoint) and --init (the converted init)")
    if args.float and args.cal_batches:
        raise SystemExit("--float has no integer bits to calibrate: use --cal-batches 0")
    if args.compile_cache:
        loop.enable_compilation_cache(args.compile_cache)
    jax.config.update("jax_default_matmul_precision", "highest")
    rc = load_config([args.config])
    data = dataclasses.replace(
        rc.data,
        batch_size=args.batch,
        num_workers=args.workers,
        train_path=args.train_path or rc.data.train_path,
        test_path=args.test_path,
        num_train=int(max(args.cal_batches, 8) * args.batch * 1.05) + 2 * args.batch,  # >= the accumulation: Trainer checks it
        num_test=args.num_test or -1,
        # a config copied from another host names its scale dict there: the packaged file of the same name then
        dataset_kwargs={
            **rc.data.dataset_kwargs,
            "is_inference": args.inference,
            "scale_dict_path": resolve_file(rc.data.dataset_kwargs["scale_dict_path"], [Path(args.config).parent]),
        },
    )
    rc = dataclasses.replace(rc, data=data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out.with_suffix(".run")
    settings = loop.LoopSettings.from_run_config(rc, run_dir, max_epochs=10_000, matmul_precision="highest")
    test = ClicSource(rc, "test", seed=42)
    train = ClicSource(rc, "train", seed=42)
    report = {"config": args.config, "ckpt": args.ckpt, "init": args.init, "float": args.float, "cal_batches": args.cal_batches}
    report["test_path"], report["test_events"], report["batch"] = args.test_path, test.num_events, args.batch
    if args.ckpt:
        meta = loop.ckpt.read_meta(args.ckpt)
        t = loop.Trainer(rc, settings, sample=next(iter(train.epoch(0))), train_batches=meta["micro_per_epoch"])
        t.total_steps, t.micro_per_epoch = meta["total_steps"], meta["micro_per_epoch"]
        t.restore(args.ckpt, check_seed=False)
        report["step"] = t.global_step
    else:
        t = loop.Trainer(rc, settings, sample=next(iter(train.epoch(0))), train_batches=train.num_batches)
        t.warm_start(args.init)
    if args.cal_batches:
        t._set_beta(1.0)  # noqa: SLF001  (EBOPs read-out = penalty; beta is not used by the eval step)
        ntv, ebops = t.calibrated_ntv(train, args.cal_batches)
        t.state = t.state._replace(ntv=ntv)
        report["ebops_calibrated"] = ebops
    start = time.time()
    with float_mode() if args.float else contextlib.nullcontext():
        metrics = t.test(test, out, test.num_events)
    report["seconds"] = round(time.time() - start, 1)
    report["metrics"] = metrics
    report["h5"] = str(out)
    report["root"] = str(write_root(out))
    Path(out.with_suffix(".json")).write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(
        "RESULT " + json.dumps({k: report[k] for k in ("root", "test_events", "ebops_calibrated", "seconds") if k in report}, default=float),
        flush=True,
    )
    return report


E_BINS = (0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
PT_BINS_LOG = (1.0, 100.0, 10)  # plot_eff_fr_purity: 10 log bins from 1 to 100 GeV
CLASSES = {0: "ch had", 1: "e", 2: "mu", 3: "neut had", 4: "photon"}


def jet_summary(ref_e, e_rel, boot: int, seed: int = 0) -> dict:
    """Global and per truth-E bin median and IQR of the jet-E response, with bootstrap sigmas."""
    import numpy as np  # noqa: PLC0415

    def stats(r, e):
        q25, q50, q75 = np.percentile(e, [25, 50, 75])
        per = []
        for a, b in itertools.pairwise(E_BINS):
            v = e[(r > a) & (r < b)]
            per.append((np.nan, np.nan) if v.size == 0 else (float(np.percentile(v, 50)), float(np.percentile(v, 75) - np.percentile(v, 25))))
        return q75 - q25, q50, per

    ref_e, e_rel = np.asarray(ref_e), np.asarray(e_rel)
    if ref_e.size == 0:
        return {"n_jets": 0}
    iqr, med, per = stats(ref_e, e_rel)
    rng = np.random.default_rng(seed)
    draws = [stats(ref_e[i], e_rel[i]) for i in (rng.integers(0, len(ref_e), len(ref_e)) for _ in range(boot))]
    per_iqr_sigma = np.nanstd(np.array([[p[1] for p in d[2]] for d in draws]), axis=0)
    return {
        "n_jets": len(ref_e),
        "iqr": float(iqr),
        "iqr_sigma": float(np.std([d[0] for d in draws])),
        "median": float(med),
        "median_sigma": float(np.std([d[1] for d in draws])),
        "per_bin": [
            {"e_lo": a, "e_hi": b, "n": int(((ref_e > a) & (ref_e < b)).sum()), "median": m, "iqr": q, "iqr_sigma": float(s)}
            for (a, b), (m, q), s in zip(itertools.pairwise(E_BINS), per, per_iqr_sigma, strict=True)
        ],
    }


def rate(k, n) -> dict:
    import math  # noqa: PLC0415

    p = k / n if n else float("nan")
    return {"value": p, "sigma": math.sqrt(p * (1 - p) / n) if n else float("nan"), "k": int(k), "n": int(n)}


def particle_summary(matched) -> dict:
    """Efficiency and fake rate (``plot_eff_fr_purity``), all classes and per class; neutral hadrons / photons per pt bin."""
    import numpy as np  # noqa: PLC0415

    ref, comp, ref_un, comp_un = matched
    t_cls, t_un_cls, p_cls, p_un_cls = (np.asarray(d["class"]) for d in (ref, ref_un, comp, comp_un))
    t_pt, t_un_pt, p_pt, p_un_pt = (np.asarray(d["pt"]) for d in (ref, ref_un, comp, comp_un))
    out = {"all": {"eff": rate(len(t_cls), len(t_cls) + len(t_un_cls)), "fake": rate(len(p_un_cls), len(p_cls) + len(p_un_cls))}}
    for c, name in CLASSES.items():
        out[name] = {
            "eff": rate((t_cls == c).sum(), (t_cls == c).sum() + (t_un_cls == c).sum()),
            "fake": rate((p_un_cls == c).sum(), (p_cls == c).sum() + (p_un_cls == c).sum()),
        }
    lo, hi, n = PT_BINS_LOG
    edges = np.logspace(np.log10(lo), np.log10(hi), n + 1)
    for c in (3, 4):
        rows = []
        for a, b in itertools.pairwise(edges):

            def inside(pt, cls, a=a, b=b, c=c):
                return (pt > a) & (pt <= b) & (cls == c)

            k_eff, n_eff = inside(t_pt, t_cls).sum(), inside(t_pt, t_cls).sum() + inside(t_un_pt, t_un_cls).sum()
            k_fr, n_fr = inside(p_un_pt, p_un_cls).sum(), inside(p_pt, p_cls).sum() + inside(p_un_pt, p_un_cls).sum()
            rows.append({"pt_lo": float(a), "pt_hi": float(b), "eff": rate(k_eff, n_eff), "fake": rate(k_fr, n_fr)})
        out[f"{CLASSES[c]} per pt"] = rows
    return out


def cmd_jets(args) -> dict:
    import math  # noqa: PLC0415

    from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig  # noqa: PLC0415

    nets = dict(n.split("=", 1) for n in args.net)
    if args.ref not in nets:
        raise SystemExit(f"--ref {args.ref} is not one of --net {sorted(nets)}")
    networks = [
        {"name": f"{name} [{conv}]", "path": path, "network_type": conv, "ind_threshold": args.ind_threshold}
        for name, path in nets.items()
        for conv in args.conventions
    ]
    perf = Performance(PerformanceConfig.from_dict({"truth_path": args.truth, "networks": networks}))
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=args.n_procs)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)
    perf.hung_match_particles(flatten=True, return_unmatched=True)
    report = {"truth": args.truth, "nets": nets, "ref": args.ref, "events": int(perf.n_events), "networks": {}}
    for name in perf.network_names:
        d = perf.data[name]
        entry = {}
        if "jet_residuals" in d:
            entry["jets"] = jet_summary(d["jet_residuals"]["ref_e"], d["jet_residuals"]["e_rel"], args.boot)
        if "matched_particles" in d:
            entry["particles"] = particle_summary(d["matched_particles"])
        report["networks"][name] = entry
    deltas = {}
    for conv in args.conventions:
        ref = report["networks"][f"{args.ref} [{conv}]"]
        for name in nets:
            if name == args.ref:
                continue
            got = report["networks"][f"{name} [{conv}]"]
            if not (got.get("jets", {}).get("n_jets") and ref.get("jets", {}).get("n_jets")):
                continue
            dj = {k: got["jets"][k] - ref["jets"][k] for k in ("iqr", "median")} | {
                f"{k}_sigma": math.hypot(got["jets"][f"{k}_sigma"], ref["jets"][f"{k}_sigma"]) for k in ("iqr", "median")
            }
            dp = {
                f"{cls} {r}": got["particles"][cls][r]["value"] - ref["particles"][cls][r]["value"]
                for cls in ("all", "neut had", "photon", "ch had")
                for r in ("eff", "fake")
            }
            deltas[f"{name} - {args.ref} [{conv}]"] = {"jets": dj, "particles": dp}
    report["deltas"] = deltas
    lines = [f"events {report['events']}"]
    for name, entry in report["networks"].items():
        j, p = entry.get("jets"), entry.get("particles")
        if j is not None and j["n_jets"] == 0:
            lines.append(f"{name:28s} no matched jets")
        elif j and p:
            lines.append(
                f"{name:28s} jet-E IQR {j['iqr']:.4f}+-{j['iqr_sigma']:.4f} median {j['median']:+.4f}+-{j['median_sigma']:.4f} "
                f"n_jets {j['n_jets']} | eff {p['all']['eff']['value']:.4f} fake {p['all']['fake']['value']:.4f} | "
                f"nh eff {p['neut had']['eff']['value']:.3f} fake {p['neut had']['fake']['value']:.3f} | "
                f"ph eff {p['photon']['eff']['value']:.3f} fake {p['photon']['fake']['value']:.3f}"
            )
    for name, d in deltas.items():
        lines.append(
            f"DELTA {name}: IQR {d['jets']['iqr']:+.4f}+-{d['jets']['iqr_sigma']:.4f} "
            f"median {d['jets']['median']:+.4f}+-{d['jets']['median_sigma']:.4f}"
        )
    report["summary"] = lines
    Path(args.out).write_text(json.dumps(report, indent=1, default=float) + "\n")
    print("\n".join(f"RESULT {line}" for line in lines), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="24_ws_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--config", required=True, help="the arm's arm_config.yaml (QAT) or the init's config.yaml (float)")
    p.add_argument("--ckpt", help="native QAT checkpoint (.npz with its .json)")
    p.add_argument("--init", help="the converted init.npz (the float model with --float)")
    p.add_argument("--float", action="store_true", help="every HGQ2 quantizer the identity")
    p.add_argument("--cal-batches", type=int, default=256, help="train micro-batches that set the WRAP integer bits (0: as stored)")
    p.add_argument("--train-path", help="calibration events (default: the config's train file)")
    p.add_argument("--test-path", required=True)
    p.add_argument("--inference", action=argparse.BooleanOptionalAction, default=True, help="CLICDataset is_inference (*_infer.root)")
    p.add_argument("--num-test", type=int)
    p.add_argument("--batch", type=int, default=128, help="128 = the training micro-batch: the calibration set of val_cal")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--compile-cache")
    p.add_argument("--out", required=True, help="prediction h5; the ROOT file and a JSON report go next to it")
    j = sub.add_parser("jets")
    j.add_argument("--truth", required=True, help="truth + Pandora file (test_clic_common_raw.root)")
    j.add_argument("--net", action="append", required=True, help="NAME=PREDICTION.root (repeat)")
    j.add_argument("--ref", required=True, help="the NAME the deltas refer to (the float model)")
    j.add_argument("--conventions", nargs="+", default=["mpflow_proxy", "mpflow"])
    j.add_argument("--ind-threshold", type=float, default=0.65)
    j.add_argument("--n-procs", type=int, default=16)
    j.add_argument("--boot", type=int, default=400)
    j.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "predict":
        if os.environ.get("KERAS_BACKEND") != "jax":
            raise SystemExit("predict needs KERAS_BACKEND=jax")
        cmd_predict(args)
    else:
        cmd_jets(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
