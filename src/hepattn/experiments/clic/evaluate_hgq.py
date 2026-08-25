"""Evaluation harness for a trained Keras/HGQ2 CLIC MaskFormer.

Produces a consolidated report with three parts:
  1. Physics metrics on the test split (per-task metrics(), the exact functions the
     validation stage logs), optionally side-by-side with a float baseline.
  2. EBOPs accounting (total + by region) — the FPGA-cost side of the tradeoff.
  3. hls4ml resource/latency estimate for the deployable subgraphs (head MLP,
     attention core).

    python -m hepattn.experiments.clic.evaluate_hgq \
        --config <run_dir>/config.yaml --state <ckpt-or-keras-state> \
        --num-events 2000 --out eval_report.md [--float-state <float-state>] [--hls]

The physics numbers reuse the repo's own task.metrics; EBOPs and hls4ml come from
hepattn.keras.{evaluate,export}. Ratio metrics (eff/fake/precision) are averaged over
batches — an approximation that tightens with more, larger batches; noted in the report.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from hepattn.experiments.clic.pflow_data import PflowDataModule
from hepattn.experiments.clic.port_to_keras import build_keras_maskformer_from_config
from hepattn.keras import set_keras_default_device
from hepattn.keras.evaluate import ebops_by_region, total_ebops


def load_state(model, state_path: str | Path) -> None:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:  # Lightning checkpoint
        state = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)


def build_loader(data_cfg: dict, num_events: int):
    cfg = dict(data_cfg)
    if num_events > 0:
        cfg["num_test"] = num_events
    dm = PflowDataModule(**cfg)
    dm.trainer = SimpleNamespace(is_global_zero=True, local_rank=0)  # datamodule only reads these for rank prints
    dm.setup("test")
    return dm.test_dataloader()


@torch.no_grad()
def physics_metrics(model, loader) -> dict[str, float]:
    """Event-weighted mean of every task's metrics() at the final decoder layer."""
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    n_events = 0
    for inputs, targets in loader:
        outputs = model(inputs)
        # loss() runs Hungarian matching and permutes outputs so preds align with targets
        _, matched_targets, _ = model.loss(outputs, dict(targets))
        preds = model.predict(outputs)
        batch_events = next(iter(inputs.values())).shape[0]
        final = preds.get("final", {})
        for task in model.tasks:
            if task.name not in final:
                continue
            for key, value in task.metrics(final[task.name], matched_targets).items():
                v = float(value)
                if not math.isnan(v):  # skip NaN (e.g. empty-target batches)
                    sums[f"{task.name}/{key}"] += v * batch_events
        n_events += batch_events
    return {k: v / max(1, n_events) for k, v in sorted(sums.items())}


def build_report(quant_metrics, quant_ebops, region_ebops, hls_report, float_metrics, num_events) -> str:
    lines = [
        "# Keras/HGQ2 CLIC MaskFormer — evaluation report",
        "",
        f"Test events: {num_events}",
        "",
        "## Physics metrics (per-task metrics() at final layer)",
        "",
    ]
    if float_metrics is not None:
        lines += ["| Metric | Quantized | Float baseline | Δ |", "|---|---|---|---|"]
        for k in sorted(set(quant_metrics) | set(float_metrics)):
            q = quant_metrics.get(k, float("nan"))
            f = float_metrics.get(k, float("nan"))
            lines.append(f"| `{k}` | {q:.5g} | {f:.5g} | {q - f:+.3g} |")
    else:
        lines += ["| Metric | Quantized |", "|---|---|"]
        lines += [f"| `{k}` | {q:.5g} |" for k, q in quant_metrics.items()]
    lines += [
        "",
        "_Ratio metrics (eff/fake/precision) are batch-averaged; treat as estimates._",
        "",
        "## Resource cost (EBOPs ≈ LUT + 55·DSP)",
        "",
        f"**Total EBOPs: {quant_ebops:.4g}**",
        "",
        "| Region | EBOPs | Share |",
        "|---|---|---|",
    ]
    for region, e in region_ebops.items():
        share = e / quant_ebops if quant_ebops else 0.0
        lines.append(f"| {region} | {e:.4g} | {share:.1%} |")
    lines.append("")
    if hls_report is not None:
        lines += ["## hls4ml conversion (deployable subgraphs)", "", "```json", json.dumps(hls_report, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML (as saved in the run directory)")
    parser.add_argument("--state", required=True, help="quantized keras state dict (.pt) or Lightning checkpoint (.ckpt)")
    parser.add_argument("--float-state", default=None, help="optional float-model state for a side-by-side baseline")
    parser.add_argument("--num-events", type=int, default=2000)
    parser.add_argument("--hls", action="store_true", help="also run hls4ml conversion + resource report on the classifier head")
    parser.add_argument("--out", default=None, help="write the markdown report here (also prints)")
    opts = parser.parse_args(args)

    set_keras_default_device("cpu")
    raw = yaml.safe_load(Path(opts.config).read_text())
    model_cfg = raw["model"]["model"]
    quant = model_cfg.get("init_args", {}).get("quant")

    loader = build_loader(raw["data"], opts.num_events)
    first_inputs, _ = next(iter(loader))

    qmodel = build_keras_maskformer_from_config(model_cfg, quant=quant).eval()
    with torch.no_grad():
        qmodel(first_inputs)  # materialize lazily-built quantized layers
    load_state(qmodel, opts.state)

    q_metrics = physics_metrics(qmodel, loader)
    q_ebops = total_ebops(qmodel, first_inputs)
    region_ebops = ebops_by_region(qmodel, first_inputs)

    f_metrics = None
    if opts.float_state:
        fmodel = build_keras_maskformer_from_config(model_cfg, quant=None).eval()
        load_state(fmodel, opts.float_state)
        f_metrics = physics_metrics(fmodel, loader)

    hls_report = None
    if opts.hls:
        from hepattn.keras.export import build_functional_dense, report_hls_resources  # noqa: PLC0415

        head = qmodel.tasks[0].net  # classification head, applied per query -> (num_queries, dim)
        n_queries = qmodel.decoder._num_queries  # noqa: SLF001
        functional = build_functional_dense(head, input_shape=(n_queries, qmodel.dim), name="clf_head")
        hls_report = report_hls_resources(functional, Path(opts.out or ".").parent / "hls_clf_head")

    report = build_report(q_metrics, q_ebops, region_ebops, hls_report, f_metrics, opts.num_events)
    print(report)
    if opts.out:
        Path(opts.out).write_text(report)
        print(f"wrote {opts.out}")


if __name__ == "__main__":
    main()
