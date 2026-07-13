"""Port a trained torch CLIC MaskFormer checkpoint into a KerasMaskFormer state dict.

Usage:
    python -m hepattn.experiments.clic.port_to_keras \
        --config <run_dir>/config.yaml --ckpt <run_dir>/ckpts/epoch=...ckpt --out keras_state.pt

The config is the experiment YAML (as saved by the CLI into the run directory). The
torch model is instantiated from its model.model section via jsonargparse — exactly
the mechanism LightningCLI uses — its attention backends are switched to the portable
"torch" SDPA path, the checkpoint weights are loaded, and everything is ported into a
KerasMaskFormer built from the same section. The result is saved as a torch state
dict, loadable with KerasMaskFormer.load_state_dict after building the model with
build_keras_maskformer_from_config (any quant spec — float weights warm-start QAT).
"""

import argparse
from pathlib import Path

import torch
import yaml
from jsonargparse import ArgumentParser
from torch import nn

from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.keras.porting import port_maskformer
from hepattn.models.encoder import change_attn_backends
from hepattn.models.maskformer import MaskFormer


def instantiate_model_section(model_cfg: dict) -> MaskFormer:
    """Instantiate the torch MaskFormer from a jsonargparse class_path/init_args config dict."""
    parser = ArgumentParser()
    parser.add_subclass_arguments(nn.Module, "model")
    cfg = parser.parse_object({"model": model_cfg})
    return parser.instantiate_classes(cfg).model


def build_keras_maskformer_from_config(model_cfg: dict, quant: dict | None = None) -> KerasMaskFormer:
    """Build a KerasMaskFormer from the torch model.model YAML section.

    input_nets / tasks / matcher are instantiated exactly as in the torch model (then
    net-swapped in place by KerasMaskFormer); the encoder/decoder class_path sections
    are reduced to their init_args dicts for the keras mirrors.
    """
    init_args = dict(model_cfg["init_args"])

    encoder_cfg = dict(init_args["encoder"].get("init_args", init_args["encoder"]))
    encoder_cfg.pop("dim", None)
    decoder_cfg = init_args["decoder"]
    decoder_cfg = dict(decoder_cfg.get("init_args", decoder_cfg))

    donor = instantiate_model_section(model_cfg)

    return KerasMaskFormer(
        input_nets=donor.input_nets,
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        tasks=donor.tasks,
        dim=init_args["dim"],
        target_object=init_args.get("target_object", "particle"),
        matcher=donor.matcher,
        encoder_tasks=donor.encoder_tasks if len(donor.encoder_tasks) else None,
        quant=quant,
    )


def load_torch_model(model_cfg: dict, ckpt_path: str | Path) -> MaskFormer:
    model = instantiate_model_section(model_cfg)
    change_attn_backends(model, "torch")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    return model.eval()


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML (as saved in the run directory)")
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint to port")
    parser.add_argument("--out", required=True, help="output path for the keras state dict (.pt)")
    opts = parser.parse_args(args)

    raw = yaml.safe_load(Path(opts.config).read_text())
    model_cfg = raw["model"]["model"]

    torch_model = load_torch_model(model_cfg, opts.ckpt)
    keras_model = build_keras_maskformer_from_config(model_cfg)
    port_maskformer(torch_model, keras_model)

    torch.save(keras_model.state_dict(), opts.out)
    print(f"ported {opts.ckpt} -> {opts.out}")


if __name__ == "__main__":
    main()
