"""The org/main CLIC model (base.yaml + linformer.yaml, ac0da8f) as torch and hybrid twins, built the way training builds them.

Both models are instantiated by jsonargparse from a YAML ``model:`` section, as the LightningCLI does in
``main.py`` / ``main_hgq.py``, so a config key that the hybrid signature rejects or misreads fails here too.

- The torch reference comes straight from the org configs (``merge_org_configs``), never from the generator, so
  that the generated Polaris config (``polaris/03_make_config.py``) is checked against org and not against itself.
- ``to_dense_attention`` swaps the masked-Linformer attention for dense attention on both sides (every other part
  unchanged: PR #8 layers, SwiGLU, phi sorting, tasks, matcher, losses). The R4a rows ran it before the hybrid had
  the masked-Linformer core (lane R4b); they are kept as the dense-attention twin of the full model.
"""

import copy
import os
import pathlib
import subprocess
import sys

import torch
import yaml
from jsonargparse import ArgumentParser

import hepattn
from hepattn.experiments.clic.pflow_data import CLICDataset

REPO = pathlib.Path(__file__).resolve().parents[2]
CLIC = pathlib.Path(hepattn.__file__).parent / "experiments/clic"
CONFIGS = CLIC / "configs"
MAKE_CONFIG = REPO / "polaris/03_make_config.py"
SCALE_DICT = str(CONFIGS / "clic_var_transform.yaml")
NUM_QUERIES = 150
MAX_NODES = 160
LINFORMER_KEYS = ("linformer_seq_len", "linformer_k")


def merge(base: dict, overlay: dict) -> dict:
    """Test-local overlay rule (dicts merge, other values replace); independent of the generator's copy."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def merge_org_configs() -> dict:
    """org/main's linformer model config: base.yaml with linformer.yaml over it."""
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    return merge(base, yaml.safe_load((CONFIGS / "linformer.yaml").read_text()))


def generate_polaris_config(out: pathlib.Path, **env: str) -> dict:
    """Run polaris/03_make_config.py with ``env`` overrides and load the YAML it writes."""
    full_env = {**os.environ, "HGQ_CONFIG_OUT": str(out), **env}
    subprocess.run([sys.executable, str(MAKE_CONFIG)], check=True, env=full_env, capture_output=True, text=True)
    return yaml.safe_load(out.read_text())


def _dense(attn_kwargs: dict) -> None:
    for key in LINFORMER_KEYS:
        attn_kwargs.pop(key)
    attn_kwargs["attn_type"] = "torch"


def to_dense_attention(cfg: dict) -> dict:
    """The same model section with dense attention everywhere (``cfg`` is a full config; a copy is returned)."""
    cfg = copy.deepcopy(cfg)
    net = cfg["model"]["model"]["init_args"]
    encoder = net["encoder"]["init_args"] if "class_path" in net["encoder"] else net["encoder"]
    _dense(encoder["attn_kwargs"])
    encoder["attn_type"] = "torch"
    _dense(net["decoder"]["decoder_layer_config"]["attn_kwargs"])
    return cfg


def torch_reference_config(cfg: dict) -> dict:
    """Org config ready to run on CPU: the encoder's flash-varlen becomes torch (same maths), the solver scipy."""
    cfg = copy.deepcopy(cfg)
    net = cfg["model"]["model"]["init_args"]
    if net["encoder"]["init_args"]["attn_type"] in {"flash", "flash-varlen"}:
        net["encoder"]["init_args"]["attn_type"] = "torch"
    net["matcher"]["init_args"].update({"default_solver": "scipy", "parallel_solver": False})
    return cfg


def instantiate(cfg: dict, quantized: bool | None = None):
    """The Lightning module of a full config, built by jsonargparse. Returns its ``.model``, in eval mode.

    ``quantized=False`` drops the HGQ2 ``quant`` section of a KerasMaskFormer config (the float twin).
    """
    from hepattn.experiments.clic.lightning_module import MPflow  # noqa: PLC0415

    cfg = copy.deepcopy(cfg)
    net = cfg["model"]["model"]["init_args"]
    module_cls = MPflow
    if cfg["model"]["model"]["class_path"].endswith("KerasMaskFormer"):
        from hepattn.experiments.clic.lightning_module_hgq import MPflowHGQ  # noqa: PLC0415

        module_cls = MPflowHGQ
        if quantized is False:
            net.pop("quant", None)
    for task in net["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task.get("init_args", {}):
            task["init_args"]["scale_dict_path"] = SCALE_DICT
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(module_cls, "model", fail_untyped=False)
    init = parser.instantiate_classes(parser.parse_object({"model": {**cfg["model"], "name": cfg["name"]}}))
    return init.model.model.eval()


def clic_dummy_batch(num_events: int = 2, seed: int = 0) -> tuple[dict, dict]:
    """CLIC dummy events (every event has at least one padded node: n_nodes < max_nodes)."""
    ds = CLICDataset(
        filepath="dummy",
        inputs={"node": ["features"]},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=SCALE_DICT,
        num_events=num_events,
        num_objects=NUM_QUERIES,
        max_nodes=MAX_NODES,
        dummy_data=True,
    )
    events = [ds[seed + i] for i in range(num_events)]
    inputs = {k: torch.stack([e[0][k] for e in events]) for k in events[0][0]}
    targets = {k: torch.stack([e[1][k] for e in events]) for k in events[0][1]}
    return inputs, targets
