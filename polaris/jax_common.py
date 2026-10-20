"""Shared helpers of the keras_native Polaris scripts (16_jax_smoke, 17_g3_step) and of their local tiny-dim checks.

- ``generate_config``: run ``polaris/03_make_config.py`` into a file (the production config both frameworks read).
- ``shrink_config``: the same config at smaller widths / depths / node and query counts, for a local CPU run. Every
  width that equals the model dim follows ``dim`` (input net, posenc, encoder, decoder layers, task nets, mask dim,
  the regression input ``2 dim + 6``); task hidden layers become ``[dim]``; the Linformer sequence lengths follow the
  node and query counts (encoder: nodes + registers; decoder: the longer of nodes and queries).
- ``median_ms`` / ``quantiles_ms``: timing summaries.

No keras / jax / torch import here: the scripts import this before they pick a backend.
"""

import copy
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAKE_CONFIG = REPO / "polaris/03_make_config.py"


def generate_config(out: Path, **env_overrides: str) -> Path:
    """Run ``03_make_config.py`` (``HGQ_*`` / ``DATA_DIR`` / ``CKPT_DIR`` overrides as keyword arguments) into ``out``."""
    env = {**os.environ, "HGQ_CONFIG_OUT": str(out), **{k: str(v) for k, v in env_overrides.items()}}
    subprocess.run([sys.executable, str(MAKE_CONFIG)], env=env, check=True, capture_output=True, text=True)
    return out


def _init_args(cfg: dict) -> dict:
    entry = cfg["model"]["model"]
    return entry.get("init_args", entry)


def shrink_config(
    cfg: dict,
    *,
    dim: int | None = None,
    heads: int | None = None,
    enc_layers: int | None = None,
    dec_layers: int | None = None,
    linformer_k: int | None = None,
    nodes: int | None = None,
    queries: int | None = None,
) -> dict:
    """A copy of a generated config (``03_make_config.py`` form) with the given sizes; None keeps a size."""
    cfg = copy.deepcopy(cfg)
    init = _init_args(cfg)
    enc, dec = init["encoder"], init["decoder"]
    dec_layer = dec["decoder_layer_config"]
    if enc_layers is not None:
        enc["num_layers"] = enc_layers
    if dec_layers is not None:
        dec["num_decoder_layers"] = dec_layers
    if queries is not None:
        dec["num_queries"] = queries
        cfg["data"]["num_objects"] = queries
    if nodes is not None:
        cfg["data"]["max_nodes"] = nodes
    n_nodes = cfg["data"].get("max_nodes", 160)
    n_queries = dec["num_queries"]
    regs = enc.get("num_register_tokens", 0)
    for attn, seq in ((enc["attn_kwargs"], n_nodes + regs), (dec_layer["attn_kwargs"], max(n_nodes, n_queries))):
        if heads is not None:
            attn["num_heads"] = heads
        if linformer_k is not None:
            attn["linformer_k"] = linformer_k
        if "linformer_seq_len" in attn:
            attn["linformer_seq_len"] = seq
    if dim is not None and dim != init["dim"]:
        old = init["dim"]
        init["dim"] = dec_layer["dim"] = dim
        for net in init["input_nets"]["init_args"]["modules"]:
            net["init_args"]["net"]["init_args"]["output_size"] = dim
            net["init_args"]["posenc"]["init_args"]["dim"] = dim
        n_fields = None
        for task in init["tasks"]["init_args"]["modules"]:
            targs = task["init_args"]
            if targs.get("dim") == old:
                targs["dim"] = dim
            n_fields = len(targs["fields"]) if "fields" in targs else n_fields
            for key in ("net", "object_net", "constituent_net", "node_net"):
                if key not in targs:
                    continue
                net = targs[key]["init_args"]
                regression_input = n_fields is not None and net["input_size"] == 2 * old + n_fields + 1
                net["input_size"] = 2 * dim + n_fields + 1 if regression_input else dim
                if net.get("hidden_layers"):
                    net["hidden_layers"] = [dim]
    return cfg


def median_ms(seconds: list[float]) -> float:
    return statistics.median(seconds) * 1e3


def quantiles_ms(seconds: list[float]) -> tuple[float, float]:
    """(p10, p90) in ms; the extremes for fewer than 2 samples."""
    if len(seconds) < 2:
        return (min(seconds) * 1e3, max(seconds) * 1e3) if seconds else (float("nan"), float("nan"))
    q = statistics.quantiles(seconds, n=10)
    return q[0] * 1e3, q[-1] * 1e3
