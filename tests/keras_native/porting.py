"""Weight porting: pure-torch hepattn modules -> their hepattn.keras_native twins (test utility; imports torch).

Kept outside ``src/hepattn/keras_native`` on purpose (that package must not import torch). Conventions:
- keras Dense kernel = torch Linear weight transposed.
- torch Attention packs Q/K/V in ``in_proj_weight`` (3*dim, dim), split as chunk(3, dim=0) in q, k, v order.
- torch LayerNorm weight/bias -> keras LayerNormalization gamma/beta.
Every native layer must be built (``layer.build`` or one eager call) before porting.
"""

import numpy as np
import torch
from torch import nn

from hepattn.keras_native.attention import Attention as NativeAttention
from hepattn.keras_native.linformer import LinformerAttention as NativeLinformer
from hepattn.models.attention import Attention
from hepattn.models.encoder import LayerScale
from hepattn.models.linformer import LinformerAttention


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def assign_dense(layer, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    if not layer.built:
        raise RuntimeError(f"keras layer '{layer.name}' is not built")
    layer.kernel.assign(_np(weight).T)
    if bias is not None:
        layer.bias.assign(_np(bias))
    elif getattr(layer, "bias", None) is not None:
        raise AssertionError(f"{layer.name} has a bias but the torch layer has none")


@torch.no_grad()
def port_dense(tdense, ndense) -> None:
    linears = [m for m in tdense.net if isinstance(m, nn.Linear)]
    layers = [*ndense.hidden, ndense.final]
    assert len(linears) == len(layers), f"structure mismatch: {len(linears)} torch Linears vs {len(layers)} keras Dense"
    for lin, layer in zip(linears, layers, strict=True):
        assign_dense(layer, lin.weight, lin.bias)


@torch.no_grad()
def port_norm(tnorm: nn.Module, nnorm) -> None:
    assert isinstance(tnorm, nn.LayerNorm), f"only LayerNorm is ported, got {type(tnorm).__name__}"
    assert abs(tnorm.eps - nnorm.epsilon) < 1e-12, f"epsilon differs: torch {tnorm.eps} vs keras {nnorm.epsilon}"
    nnorm.gamma.assign(_np(tnorm.weight))
    nnorm.beta.assign(_np(tnorm.bias))


@torch.no_grad()
def port_attention(tattn: Attention, nattn: NativeAttention) -> None:
    """Torch Attention (torch or linformer backend) -> native Attention of the same attn_type.

    Linformer: ``proj_k`` / ``proj_v`` (seq_len, k) go to the core's E / F. Rows past the key length are dropped when
    the native core has no tail (no renorm, e.g. decoder query self-attention): org never reads them there.
    """
    assert tattn.attn_type == nattn.attn_type, f"attn_type mismatch: torch {tattn.attn_type} vs native {nattn.attn_type}"
    assert tattn.dim == nattn.dim and tattn.num_heads == nattn.num_heads, "attention shape mismatch"
    w_q, w_k, w_v = tattn.in_proj_weight.chunk(3, dim=0)
    b_q = b_k = b_v = None
    if tattn.in_proj_bias is not None:
        b_q, b_k, b_v = tattn.in_proj_bias.chunk(3, dim=0)
    assign_dense(nattn.q_proj, w_q, b_q)
    assign_dense(nattn.k_proj, w_k, b_k)
    assign_dense(nattn.v_proj, w_v, b_v)
    assign_dense(nattn.out_proj, tattn.out_proj.weight, tattn.out_proj.bias)
    assert tattn.qkv_norm == nattn.qkv_norm, "qkv_norm mismatch"
    if tattn.qkv_norm:
        port_norm(tattn.q_norm, nattn.q_norm)
        port_norm(tattn.k_norm, nattn.k_norm)
        port_norm(tattn.v_norm, nattn.v_norm)
    if tattn.value_residual and not tattn.is_first_layer:
        lin = tattn.value_residual_mix[0]
        assign_dense(nattn.value_residual_mix, lin.weight, lin.bias)
    if tattn.attn_type == "linformer":
        assert nattn.core is not None, f"{nattn.name}: build the native attention before porting"
        nattn.core.E.assign_full(_np(tattn.proj_k))
        nattn.core.F.assign_full(_np(tattn.proj_v))


@torch.no_grad()
def port_linformer(tlin: LinformerAttention, nlin: NativeLinformer) -> None:
    """Helen's LinformerAttention (``hepattn.models.linformer``) -> native twin; kept for R5's old goldens until their removal."""
    assert not tlin.share_kv, "share_kv is not ported"
    assign_dense(nlin.to_q, tlin.to_q.weight, None)
    assign_dense(nlin.to_k, tlin.to_k.weight, None)
    assign_dense(nlin.to_v, tlin.to_v.weight, None)
    assign_dense(nlin.to_out, tlin.to_out.weight, tlin.to_out.bias)
    nlin.proj_k.assign(_np(tlin.proj_k))
    nlin.proj_v.assign(_np(tlin.proj_v))


def port_any_attention(tattn: Attention, nfn) -> None:
    """Torch Attention (torch or linformer backend) -> native Attention (``port_attention`` covers both)."""
    port_attention(tattn, nfn)


@torch.no_grad()
def port_residual(tres, nres, fn_porter) -> None:
    assert tres.post_norm == nres.post_norm, "post_norm mismatch"
    assert isinstance(tres.norm, nn.Identity) == (nres.norm is None), "norm presence mismatch"
    if nres.norm is not None:
        port_norm(tres.norm, nres.norm)
    assert isinstance(tres.ls, LayerScale) == (nres.ls is not None), "LayerScale presence mismatch"
    if nres.ls is not None:
        nres.ls.gamma.assign(_np(tres.ls.gamma))
    fn_porter(tres.fn, nres.fn)


def port_encoder_layer(tlayer, nlayer) -> None:
    port_residual(tlayer.attn, nlayer.attn, port_any_attention)
    port_residual(tlayer.dense, nlayer.dense, port_dense)


@torch.no_grad()
def port_encoder(tenc, nenc) -> None:
    assert tenc.num_layers == nenc.num_layers, "encoder depth mismatch"
    assert (tenc.register_tokens is None) == (nenc.num_register_tokens is None), "register token mismatch"
    if tenc.register_tokens is not None:
        nenc.register_tokens.assign(_np(tenc.register_tokens))
    for tlayer, nlayer in zip(tenc.layers, nenc.layers, strict=True):
        port_encoder_layer(tlayer, nlayer)


@torch.no_grad()
def port_input_net(tnet, nnet) -> None:
    port_dense(tnet.net, nnet.net)
    assert (tnet.posenc is None) == (nnet.posenc is None), "posenc presence mismatch"
    if tnet.posenc is not None:
        nnet.posenc.B.assign(_np(tnet.posenc.B))


@torch.no_grad()
def perturb_affine_params(module: nn.Module, seed: int) -> None:
    """Give biases, norm affines, LayerScale and register tokens non-default values.

    Default inits (zero biases, unit norm weights, constant LayerScale) would let a missing or swapped port of
    those parameters pass unnoticed; after this, a missing port changes outputs at O(0.1).
    """
    gen = torch.Generator().manual_seed(seed)
    for name, param in module.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"bias", "in_proj_bias", "gamma"} or (leaf == "weight" and param.dim() == 1):
            param.add_(0.1 * torch.randn(param.shape, generator=gen))


@torch.no_grad()
def port_decoder(tdec, ndec) -> None:
    """Torch MaskFormerDecoder -> native MaskFormerDecoder (initial queries + every layer's residual blocks)."""
    assert len(tdec.decoder_layers) == len(ndec.layers), "decoder depth mismatch"
    ndec.initial_queries.assign(_np(tdec.initial_queries))
    for tlayer, nlayer in zip(tdec.decoder_layers, ndec.layers, strict=True):
        port_residual(tlayer.q_ca, nlayer.q_ca, port_any_attention)
        port_residual(tlayer.q_sa, nlayer.q_sa, port_any_attention)
        port_residual(tlayer.q_dense, nlayer.q_dense, port_dense)
        assert tlayer.bidirectional_ca == nlayer.bidirectional_ca, "bidirectional_ca mismatch"
        if tlayer.bidirectional_ca:
            port_residual(tlayer.kv_ca, nlayer.kv_ca, port_any_attention)
            port_residual(tlayer.kv_dense, nlayer.kv_dense, port_dense)


TASK_NETS = ("net", "object_net", "constituent_net", "node_net")


@torch.no_grad()
def port_task(ttask, ntask) -> int:
    """Port every Dense head of a task (by attribute name); returns the number ported."""
    count = 0
    for attr in TASK_NETS:
        tnet = getattr(ttask, attr, None)
        nnet = getattr(ntask, attr, None)
        if isinstance(tnet, nn.Identity):
            tnet = None
        assert (tnet is None) == (nnet is None), f"{attr}: presence differs (torch {tnet is not None}, native {nnet is not None})"
        if tnet is not None:
            port_dense(tnet, nnet)
            count += 1
    return count


@torch.no_grad()
def port_maskformer(tmodel, nmodel) -> None:
    """Torch MaskFormer -> native MaskFormer (built). Task order and names must match."""
    for tnet, nnet in zip(tmodel.input_nets, nmodel.input_nets, strict=True):
        port_input_net(tnet, nnet)
    port_encoder(tmodel.encoder, nmodel.encoder)
    port_decoder(tmodel.decoder, nmodel.decoder)
    for ttask, ntask in zip(tmodel.tasks, nmodel.tasks, strict=True):
        assert ttask.name == ntask.task_name, f"task order differs: {ttask.name} vs {ntask.task_name}"
        assert port_task(ttask, ntask) > 0, f"task {ttask.name}: no Dense ported"
