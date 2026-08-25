#!/usr/bin/env python
"""Measure the EBOPs coefficient the training loss actually sees, and print calibrated betas.

Why this exists: the first Polaris beta sweep was inert -- four decades of beta moved the
mean learned bitwidth by 0.01% -- because `BETA_VALUES` was derived from `total_ebops`.
That number is a resource *estimate*; it is not what the objective minimizes.
`quant_losses()` is affine in beta, ``quant_loss = floor + beta * E_eff``, and E_eff is
orders of magnitude smaller (most of the reported total comes from QSoftmax layers whose
cost never enters the regularization loss). Calibrate against E_eff.

Runs on a LOGIN node: E_eff is a per-inference cost and is batch-independent (measured:
0.1% variation from batch 1 to 8), so a batch of 2 on CPU is enough. No GPU needed.

Usage, from the repo root inside the venv:

    python polaris/09_measure_beta.py                          # uses ./polaris_hgq.yaml
    python polaris/09_measure_beta.py --config other.yaml --ckpt path/to/last.ckpt

Feed the printed beta into `BETA_VALUES` in polaris/env.sh. Note that
`03_make_config.py` multiplies beta_end by `HGQ_EBOPS_EVERY`, so pass the *unscaled*
value there.
"""

import argparse
import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hepattn.keras import set_keras_default_device  # noqa: E402
from hepattn.keras.evaluate import ebops_by_region, effective_ebops, total_ebops  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--config", default=str(REPO / "polaris_hgq.yaml"), help="training config to measure")
p.add_argument("--ckpt", default=None, help="optional checkpoint, to measure the trained bitwidths")
p.add_argument("--batch", type=int, default=2, help="batch size for the probe forward (E_eff is batch-independent)")
p.add_argument("--task-loss", type=float, default=30.0, help="observed task loss, for the calibration table")
p.add_argument("--fractions", default="0.05,0.1,0.25,0.5", help="target resource-term fractions of the task loss")
p.add_argument("--step-scan", action="store_true", help="also sweep one bitwidth family downward and record hard vs soft cost")
p.add_argument("--scan-family", default="f", choices=["b", "f", "i"], help="which quantizer family to sweep (default f: EBOPs depends on it)")
p.add_argument("--scan-range", type=float, default=2.0, help="how many bits to sweep down")
p.add_argument("--scan-step", type=float, default=0.1, help="sweep resolution in bits")
p.add_argument("--scan-out", default="stepscan.json", help="where to write the scan")
args = p.parse_args()
# LightningCLI also reads sys.argv; clear it so our flags are not parsed twice.
sys.argv = [sys.argv[0]]

if not pathlib.Path(args.config).exists():
    raise SystemExit(f"config not found: {args.config}\nRun `python polaris/03_make_config.py` first.")

# Keras-torch would auto-select cuda/mps for new tensors; pin to CPU so this runs on a
# login node and so keras constants share a device with the model.
set_keras_default_device("cpu")

from hepattn.experiments.clic.lightning_module_hgq import MPflowHGQ  # noqa: E402
from hepattn.experiments.clic.pflow_data import PflowDataModule  # noqa: E402
from hepattn.utils.cli import CLI  # noqa: E402


class MeasureCLI(CLI):
    """CLI with the subcommand-specific setup skipped.

    Both repo hooks index `self.config[self.subcommand]`, which is None when run=False.
    Their bodies only timestamp the log directory (`fit`) and resolve a best-epoch
    checkpoint (`test`) — neither applies to a measurement — so they are skipped when
    there is no subcommand, and left untouched otherwise.
    """

    def before_instantiate_classes(self) -> None:
        if self.subcommand is not None:
            super().before_instantiate_classes()

    def after_instantiate_classes(self) -> None:
        if self.subcommand is not None:
            super().after_instantiate_classes()


# run=False builds model + datamodule through the normal jsonargparse path without
# training. Force CPU and drop the loggers: the config targets a GPU compute node and
# instantiating CometLogger/CSVLogger here would create stray run directories.
cli = MeasureCLI(
    model_class=MPflowHGQ,
    datamodule_class=PflowDataModule,
    args=[
        "--config",
        args.config,
        f"--data.batch_size={args.batch}",
        # setup("fit") builds BOTH loaders. Unbounded, that reads the whole 1M-event
        # train ROOT file just to measure a per-inference cost — bound both hard.
        f"--data.num_train={args.batch * 4}",
        f"--data.num_val={args.batch * 4}",
        "--data.num_workers=0",
        "--trainer.accelerator=cpu",
        "--trainer.devices=1",
        "--trainer.logger=false",
    ],
    run=False,
)
module, datamodule = cli.model, cli.datamodule
model = module.model

# run=False builds the trainer but never attaches it; PflowDataModule.setup reads
# self.trainer.is_global_zero, so wire it up by hand.
datamodule.trainer = cli.trainer
datamodule.setup("fit")
batch, _ = next(iter(datamodule.val_dataloader()))

# HGQ2 layers build lazily at first forward, so materialize before touching parameters.
model.eval()
with torch.no_grad():
    model(batch)

if args.ckpt:
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"loaded {args.ckpt}\n  missing={len(missing)} unexpected={len(unexpected)}")

model.train()
e_eff, floor = effective_ebops(model, batch)
reported = total_ebops(model, batch)
regions = ebops_by_region(model, batch)
bits = [p.detach().float().mean().item() for name, p in model.named_parameters() if name.endswith("/b")]

print("\n================ EBOPs accounting ================")
print(f"  reported total_ebops      {reported:>14.4g}   <- resource estimate, NOT the objective")
print(f"  effective E_eff (slope)   {e_eff:>14.4g}   <- what beta trades against the task loss")
print(f"  ratio total/E_eff         {reported / e_eff:>14.4g}")
print(f"  constant floor            {floor:>14.4g}   <- beta cannot influence this")
print("\n  by region (reported):")
for name, value in regions.items():
    print(f"    {name:<12} {value:>14.4g}")
if bits:
    print(f"\n  learned bitwidths: {len(bits)} layers, mean {sum(bits) / len(bits):.4f}")

print(f"\n=========== calibrated beta (task loss {args.task_loss:g}) ===========")
print(f"  {'resource term':>16}  {'fraction':>9}  {'beta':>12}")
for frac_str in args.fractions.split(","):
    frac = float(frac_str)
    print(f"  {frac * args.task_loss:>16.3g}  {frac:>9.2f}  {frac * args.task_loss / e_eff:>12.4g}")
print("\n  Put the chosen value(s) in BETA_VALUES in polaris/env.sh (unscaled -- the")
print("  config generator multiplies by HGQ_EBOPS_EVERY itself).")


# ----------------------------------------------------------------- step scan
if args.step_scan:
    # Hardware bitwidths are integers, so the REPORTED cost is piecewise constant in
    # them while the differentiable surrogate the optimizer sees is not. Sweeping one
    # family downward and recording both makes the gap explicit: sub-bit progress is
    # worth exactly zero until a tread is crossed. Measured on the dim-32 test model,
    # the first tread is at 0.6 bits while 9 epochs of training moved /f by 0.20.
    import json

    scan_params = [q for n, q in model.named_parameters() if "quantizer" in n and q.requires_grad and n.endswith("/" + args.scan_family)]
    if not scan_params:
        raise SystemExit(f"no trainable /{args.scan_family} parameters found — nothing to scan")

    original = [q.detach().clone() for q in scan_params]
    n_points = round(args.scan_range / args.scan_step) + 1
    print(f"\n=========== step scan: /{args.scan_family}, {n_points} points ===========")
    print("  (3 forwards per point; expect a few minutes on the dim-256 model)")
    print(f"  {'shift':>7} {'mean':>8} {'hard %':>9} {'soft %':>9}")

    shifts, hard, soft = [], [], []
    try:
        for k in range(n_points):
            shift = round(k * args.scan_step, 4)
            with torch.no_grad():
                for q, o in zip(scan_params, original, strict=True):
                    q.copy_(o - shift)
            h = total_ebops(model, batch)
            s_slope, _ = effective_ebops(model, batch)
            mean_bits = float(torch.stack([q.detach().float().mean() for q in scan_params]).mean())
            shifts.append(shift)
            hard.append(h / reported)
            soft.append(s_slope / e_eff)
            print(f"  {-shift:>7.2f} {mean_bits:>8.4f} {100 * h / reported:>8.2f}% {100 * s_slope / e_eff:>8.2f}%", flush=True)
    finally:
        with torch.no_grad():
            for q, o in zip(scan_params, original, strict=True):
                q.copy_(o)

    out = {
        "family": args.scan_family,
        "shift": shifts,
        "hard": hard,
        "soft": soft,
        "base_ebops": reported,
        "base_eeff": e_eff,
        "config": args.config,
        "ckpt": args.ckpt,
    }
    pathlib.Path(args.scan_out).write_text(json.dumps(out, indent=1))
    first = next((sh for sh, hv in zip(shifts, hard, strict=True) if hv < 0.999), None)
    print(f"\n  first tread at {first if first is not None else 'beyond the scan range'} bits")
    print(f"  wrote {args.scan_out}")
