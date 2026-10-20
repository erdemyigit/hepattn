"""Port a trained torch CLIC MaskFormer checkpoint into a KerasMaskFormer state dict.

Usage:
    python -m hepattn.experiments.clic.port_to_keras \
        --config <run_dir>/config.yaml --ckpt <run_dir>/ckpts/epoch=...ckpt --out keras_state.pt

The config is the experiment YAML (as saved by the CLI into the run directory). The
torch model is instantiated from its model.model section via jsonargparse — exactly
the mechanism LightningCLI uses — its attention backends other than linformer are
switched to the portable "torch" SDPA path (a linformer attention owns trained
projections and keeps its backend), the checkpoint weights are loaded, and everything
is ported into a KerasMaskFormer built from the same section. The keras model then runs
one forward on dummy events of the config's sizes (``materialize``: builds the lazy
layers and applies the ported Linformer projections; ``state_dict()`` of an unbuilt
linformer core raises) and registers its keras weights on the torch module tree
(``register_keras_parameters``; without it an HGQ2 layer's kernel, bias, beta and ebops
are not in ``state_dict()``). The result is saved as a torch state
dict, loadable with ``load_keras_state`` (build, materialize, register, strict load),
for any quant spec — float weights warm-start QAT.

For the keras_native (JAX) model use ``convert_to_native`` instead.
"""

import argparse
from pathlib import Path

import torch
import yaml
from jsonargparse import ArgumentParser
from torch import nn

from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.keras.porting import port_maskformer
from hepattn.models.attention import Attention
from hepattn.models.encoder import Encoder
from hepattn.models.maskformer import MaskFormer


def instantiate_model_section(model_cfg: dict) -> MaskFormer:
    """Instantiate the torch MaskFormer from a jsonargparse class_path/init_args config dict."""
    parser = ArgumentParser()
    parser.add_subclass_arguments(nn.Module, "model")
    cfg = parser.parse_object({"model": model_cfg})
    return parser.instantiate_classes(cfg).model


def build_keras_maskformer_from_config(model_cfg: dict, quant: dict | None = None) -> KerasMaskFormer:
    """Build a KerasMaskFormer from a model.model YAML section (torch or keras config).

    For a torch MaskFormer config, input_nets / tasks / matcher are instantiated exactly
    as in the torch model (then net-swapped in place by KerasMaskFormer) and the
    encoder/decoder class_path sections are reduced to their init_args dicts. A config
    that already describes a KerasMaskFormer is instantiated directly (optionally with
    a quant override) — it must not be wrapped a second time.
    """
    if model_cfg["class_path"].rsplit(".", 1)[-1] == "KerasMaskFormer":
        import copy  # noqa: PLC0415

        cfg = copy.deepcopy(model_cfg)
        if quant is not None:
            cfg.setdefault("init_args", {})["quant"] = quant
        model = instantiate_model_section(cfg)
        assert isinstance(model, KerasMaskFormer)
        return model

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
        input_sort_field=init_args.get("input_sort_field"),
        quant=quant,
    )


def portable_attention(model: nn.Module) -> int:
    """Switch every non-linformer attention to the "torch" backend; returns the number of modules switched.

    ``change_attn_backends(model, "torch")`` raises for a linformer attention (org: "linformer owns trained projection
    parameters"), and the hybrid ports a linformer attention to its masked-Linformer core, not to SDPA.
    """
    switched = 0
    for module in model.modules():
        if (isinstance(module, Encoder) and module.attn_type not in {"linformer", "torch"}) or (
            isinstance(module, Attention) and module.attn_type not in {"linformer", "torch"}
        ):
            module.set_backend("torch")
            switched += 1
    return switched


def read_model_state(path: str | Path) -> dict:
    """The model state of a Lightning checkpoint (``model.`` prefix removed) or of a saved state dict.

    ``weights_only=True``: a Lightning checkpoint of these models holds tensors and plain containers only, so no
    pickled code runs.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
    return state


def load_torch_model(model_cfg: dict, ckpt_path: str | Path) -> MaskFormer:
    model = instantiate_model_section(model_cfg)
    portable_attention(model)
    model.load_state_dict(read_model_state(ckpt_path))
    return model.eval()


def dummy_inputs(data_cfg: dict, num_events: int = 2) -> dict:
    """``CLICDataset`` dummy events at the config's node / object counts (to materialize the lazy keras layers)."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from hepattn.experiments.clic.pflow_data import CLICDataset  # noqa: PLC0415

    keys = ("inputs", "targets", "num_objects", "max_nodes", "incidence_cutval", "scale_dict_path")
    kwargs = {key: data_cfg[key] for key in keys if key in data_cfg}
    dataset = CLICDataset(filepath="unused", num_events=num_events, dummy_data=True, **kwargs)
    inputs, _ = next(iter(DataLoader(dataset, batch_size=num_events)))
    return inputs


def load_keras_state(model: KerasMaskFormer, state_path: str | Path, inputs: dict) -> KerasMaskFormer:
    """Materialize ``model`` on ``inputs``, register its keras weights, then load a state strictly.

    ``load_state_dict`` raises ``RuntimeError`` if the state does not hold exactly the model's keys.
    """
    model.materialize(inputs)
    model.register_keras_parameters()
    model.load_state_dict(read_model_state(state_path), strict=True)
    return model


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
    keras_model.materialize(dummy_inputs(raw["data"]))
    keras_model.register_keras_parameters()

    torch.save(keras_model.state_dict(), opts.out)
    print(f"ported {opts.ckpt} -> {opts.out}")


if __name__ == "__main__":
    main()
