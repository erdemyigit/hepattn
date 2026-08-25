"""Standalone (no-Lightning) inference for the Keras/HGQ2 CLIC pflow model.

Reads the same experiment YAML as training (same ROOT files via CLICDataset/uproot),
builds the KerasMaskFormer from the config's model section, loads a state dict
(either a keras state from port_to_keras.py or a Lightning checkpoint), and runs the
test split in eager torch: per-event losses, thresholded predictions, and throughput.

    python -m hepattn.experiments.clic.eval_keras \
        --config <run_dir>/config.yaml --state <keras_state.pt|lightning.ckpt> --out metrics.json
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from hepattn.experiments.clic.pflow_data import PflowDataModule
from hepattn.experiments.clic.port_to_keras import build_keras_maskformer_from_config
from hepattn.keras import set_keras_default_device


def load_state(model, state_path: str | Path) -> None:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:  # Lightning checkpoint
        state = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML (as saved in the run directory)")
    parser.add_argument("--state", required=True, help="keras state dict (.pt) or Lightning checkpoint (.ckpt)")
    parser.add_argument("--out", default=None, help="optional JSON output path for the metrics summary")
    parser.add_argument("--num-events", type=int, default=-1, help="limit the number of test events")
    opts = parser.parse_args(args)

    set_keras_default_device("cpu")
    raw = yaml.safe_load(Path(opts.config).read_text())

    data_cfg = dict(raw["data"])
    if opts.num_events > 0:
        data_cfg["num_test"] = opts.num_events
    datamodule = PflowDataModule(**data_cfg)
    datamodule.trainer = SimpleNamespace(is_global_zero=True, local_rank=0)  # the datamodule only reads these for rank prints
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    model_cfg = raw["model"]["model"]
    quant = model_cfg.get("init_args", {}).get("quant")
    model = build_keras_maskformer_from_config(model_cfg, quant=quant).eval()

    # materialize lazily-built quantized layers from the first batch, then load weights
    first_inputs, _ = next(iter(loader))
    with torch.no_grad():
        model(first_inputs)
    load_state(model, opts.state)

    loss_sums: dict[str, float] = {}
    num_events = 0
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, _, losses = model.loss(outputs, dict(targets))
            model.predict(outputs)
            for layer_name, layer_losses in losses.items():
                for task_name, task_losses in layer_losses.items():
                    for loss_name, value in task_losses.items():
                        key = f"{layer_name}/{task_name}/{loss_name}"
                        if torch.isfinite(value):
                            loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
            num_events += next(iter(inputs.values())).shape[0]
    elapsed = time.perf_counter() - start

    summary = {
        "num_events": num_events,
        "wall_time_s": elapsed,
        "events_per_s": num_events / elapsed if elapsed > 0 else float("nan"),
        "mean_losses": {k: v / max(1, num_events) for k, v in sorted(loss_sums.items())},
    }
    print(json.dumps(summary, indent=2))
    if opts.out:
        Path(opts.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
