"""Weight porting: torch hepattn modules -> their hepattn.keras twins.

The load-bearing conventions:
- keras Dense kernel == torch Linear weight TRANSPOSED (y = x @ K + b vs y = x @ W.T + b).
- torch Attention packs Q/K/V as in_proj_weight of shape (3*dim, dim), consumed by
  F._in_projection_packed as chunk(3, dim=0) in q, k, v order (attention.py).
- torch norm modules are ported by state_dict (keras twins reuse the same classes).
"""

import numpy as np
import torch
from torch import nn

from hepattn.keras.attention import KerasAttention
from hepattn.keras.dense import KerasDense
from hepattn.keras.encoder import KerasEncoder, KerasEncoderLayer
from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder, EncoderLayer, LayerScale, Residual


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def assign_dense(layer, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    """Assign a torch-convention (out, in) weight matrix + bias to a keras Dense-like layer."""
    layer.kernel.assign(_to_numpy(weight).T)
    if bias is not None:
        layer.bias.assign(_to_numpy(bias))


@torch.no_grad()
def port_linear(linear: nn.Linear, layer) -> None:
    assign_dense(layer, linear.weight, linear.bias)


@torch.no_grad()
def port_dense(dense: Dense, kdense: KerasDense) -> None:
    linears = [m for m in dense.net if isinstance(m, nn.Linear)]
    klayers = [*kdense.hidden, kdense.final]
    assert len(linears) == len(klayers), f"structure mismatch: {len(linears)} torch Linears vs {len(klayers)} keras Dense layers"
    for lin, klayer in zip(linears, klayers, strict=True):
        port_linear(lin, klayer)


@torch.no_grad()
def port_attention(attn: Attention, kattn: KerasAttention) -> None:
    assert attn.dim == kattn.dim and attn.num_heads == kattn.num_heads, "attention shape mismatch"

    w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
    if attn.in_proj_bias is not None:
        b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
    else:
        b_q = b_k = b_v = None

    assign_dense(kattn.q_proj, w_q, b_q)
    assign_dense(kattn.k_proj, w_k, b_k)
    assign_dense(kattn.v_proj, w_v, b_v)
    port_linear(attn.out_proj, kattn.out_proj)

    if attn.qkv_norm:
        assert kattn.qkv_norm, "torch attention has qkv_norm but keras twin does not"
        kattn.q_norm.load_state_dict(attn.q_norm.state_dict())
        kattn.k_norm.load_state_dict(attn.k_norm.state_dict())
        kattn.v_norm.load_state_dict(attn.v_norm.state_dict())

    if attn.value_residual and not attn.is_first_layer:
        # torch: nn.Sequential(nn.Linear(dim, num_heads), nn.Sigmoid()); keras: Dense(num_heads, sigmoid)
        port_linear(attn.value_residual_mix[0], kattn.value_residual_mix)


@torch.no_grad()
def port_residual(res: Residual, kres: Residual, fn_porter) -> None:
    """Port the torch-side Residual wrapper state (norm, LayerScale) and delegate fn porting."""
    assert res.post_norm == kres.post_norm, "post_norm mismatch"
    if not isinstance(res.norm, nn.Identity):
        kres.norm.load_state_dict(res.norm.state_dict())
    if isinstance(res.ls, LayerScale):
        assert isinstance(kres.ls, LayerScale), "LayerScale mismatch"
        kres.ls.gamma.copy_(res.ls.gamma)
    fn_porter(res.fn, kres.fn)


@torch.no_grad()
def port_encoder_layer(layer: EncoderLayer, klayer: KerasEncoderLayer) -> None:
    port_residual(layer.attn, klayer.attn, port_attention)
    port_residual(layer.dense, klayer.dense, port_dense)


@torch.no_grad()
def port_encoder(encoder: Encoder, kencoder: KerasEncoder) -> None:
    assert encoder.num_layers == kencoder.num_layers, "encoder depth mismatch"
    if encoder.register_tokens is not None:
        assert kencoder.register_tokens is not None, "torch encoder has register tokens but keras twin does not"
        kencoder.register_tokens.copy_(encoder.register_tokens)
    for layer, klayer in zip(encoder.layers, kencoder.layers, strict=True):
        port_encoder_layer(layer, klayer)
