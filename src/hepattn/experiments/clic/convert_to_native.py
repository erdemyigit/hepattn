r"""Convert a Lightning checkpoint of the CLIC MaskFormer into the keras_native (JAX) model and its checkpoint format.

    python -m hepattn.experiments.clic.convert_to_native --config <run_dir>/config.yaml \\
        --ckpt <run_dir>/ckpts/epoch=...ckpt --out converted.npz [--target-config polaris_hgq.yaml] [--report r.json]

Sources (picked from ``model.model.class_path`` of ``--config``, the ``config.yaml`` the run saved):

- **org float model** (``hepattn.models.MaskFormer``; ``main.py``, ``MPflow``): the torch model is instantiated from
  the config (jsonargparse, as LightningCLI does), the checkpoint is loaded into it strictly, and every module is
  ported to its native twin (``port_maskformer``): Linear weights transposed into kernels; the packed
  ``in_proj_weight`` / ``in_proj_bias`` split into q, k, v (rows in that order); a SwiGLU ``Linear(in, 2h)`` split
  into ``x1`` = rows ``[:h]`` and ``x2`` = rows ``[h:]``; ``proj_k`` / ``proj_v`` (seq_len, k) into the masked-Linformer
  core's ``E`` / ``F`` (rows ``[:n]`` the kernel, rows ``[n:]`` the renorm tail; without a tail, as in the decoder
  query self-attention, the rows past the key length are dropped: org never reads them there); LayerNorms (affine only;
  ``FastLayerNorm`` has no parameters); LayerScale; register tokens; initial queries; the Fourier ``B`` buffer.
- **hybrid HGQ2 model** (``hepattn.keras.maskformer.KerasMaskFormer``; ``main_hgq.py``, ``MPflowHGQ``): its keras
  variables are torch parameters keyed ``<module path>._torch_params.<keras variable path>`` (keras
  ``backend/torch/layer.py``), the same variable under every enclosing keras layer. Each distinct variable path is
  matched to the native variable with the same leaf-relative path (the layer names are shared, ``factory`` naming);
  keras' auto-numbered suffixes (``_<digits>``, e.g. ``fixed_point_quantizer_kif_51``) are dropped for the match, and
  variables that then share a path (a ``QEinsum``'s ``quantizer`` / ``quantizer_1``) pair in suffix order, which is
  the construction order on both sides. Every kernel, bias and quantizer variable (bitwidths ``b``/``i``/``f``,
  signs ``k``, decay speeds), each leaf's ``beta`` and ``ebops``, and the torch parameters the hybrid keeps in torch
  (register tokens, initial queries, ``B``) are copied.

Both directions are checked: every source tensor is read (or is in a documented ignore list: the task ``loss_weights``
buffers, compared with the native constants; the hybrid's ``proj_*_dead`` rows of the query self-attention) and every
native variable is written, except, for a float source into a quantized target (QAT warm start), the quantizer state,
``beta`` and ``ebops``, which keep the target's initial values. Any other gap raises ``ConversionError``.

The output is a keras_native checkpoint (``keras_native.checkpoint`` format: ``.npz`` + JSON sidecar) of the target
model's variables with no optimizer state (step 0), for ``main_jax fit --init-from`` (a warm start: weights and
quantizer state; the optimizer, LR schedule, beta controller and data position start fresh) and ``main_jax test
--ckpt``. ``--target-config`` builds another native model from the same weights (e.g. the QAT Polaris config for an org
float checkpoint); it defaults to ``--config``. The checkpoint's config hash is the target's, so ``main_jax`` refuses
it for another model.

For a quantized target the report also holds ``renorm_lane`` (``renorm_lane_check``: the largest masked-Linformer
column energy and the count of nonzero E entries the renorm lane rounds away; a nonzero count also warns).

A source matcher's GPU ``device_solver`` is dropped before the torch model is instantiated (``without_device_solver``;
reported as ``source_device_solver_dropped``): conversion never matches. The native target config must not have one.

Checkpoints are read with ``torch.load(weights_only=True)`` (no pickled code runs). This module imports torch, so it
lives outside ``hepattn.keras_native`` (import-hygiene rule); keras runs on JAX (set when ``KERAS_BACKEND`` is unset).
"""

import argparse
import copy
import hashlib
import json
import operator
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

TORCH_SOURCE = "torch"
HYBRID_SOURCE = "hybrid"
KERAS_PARAM_MARK = "._torch_params."
HYBRID_IGNORED = re.compile(r"(^|\.)(loss_weights|proj_k_dead|proj_v_dead)$")
_AUTO_SUFFIX = re.compile(r"_(\d+)$")
FLASH_TYPES = {"flash", "flash-varlen"}  # = keras_native.encoder.COERCE_TO_TORCH (asserted in test_convert.py)


class ConversionError(RuntimeError):
    """A checkpoint that does not map one-to-one onto the native model."""


# ----------------------------------------------------------------------------------------------- coverage tracking


class _Coverage:
    """Records which torch tensors were read and which native variables were written during a port."""

    def __init__(self, tmodel: nn.Module | None = None) -> None:
        self.names: dict[int, list[str]] = defaultdict(list)  # a shared module (the tasks) has several names
        if tmodel is not None:
            for name, tensor in [*tmodel.named_parameters(remove_duplicate=False), *tmodel.named_buffers(remove_duplicate=False)]:
                self.names[id(tensor)].append(name)
        self.read: set[str] = set()
        self.written: set[int] = set()
        self.dropped_rows: dict[str, int] = {}


_COVERAGE: _Coverage | None = None


def _np(t: torch.Tensor) -> np.ndarray:
    if _COVERAGE is not None and id(t) in _COVERAGE.names:
        _COVERAGE.read.update(_COVERAGE.names[id(t)])
    return t.detach().cpu().numpy()


def _assign(var, value: np.ndarray) -> None:
    value = np.asarray(value)
    if tuple(var.shape) != value.shape:
        raise ConversionError(f"{var.path}: native shape {tuple(var.shape)}, source {value.shape}")
    var.assign(value.astype(var.dtype, copy=False))
    if _COVERAGE is not None:
        _COVERAGE.written.add(id(var))


# ----------------------------------------------------------------------------------------------- torch modules -> native


def assign_dense(layer, weight: np.ndarray, bias: np.ndarray | None) -> None:
    """A torch Linear (weight (out, in), bias) into a built keras Dense / QDense (kernel (in, out)).

    Raises:
        RuntimeError: If the keras layer is not built.
        ConversionError: If the torch layer has no bias and the keras layer has one.
    """
    if not layer.built:
        raise RuntimeError(f"keras layer '{layer.name}' is not built")
    _assign(layer.kernel, weight.T)
    if bias is not None:
        _assign(layer.bias, bias)
    elif getattr(layer, "bias", None) is not None:
        raise ConversionError(f"{layer.name} has a bias but the torch layer has none")


@torch.no_grad()
def port_dense(tdense, ndense) -> None:
    """Torch Dense -> native Dense. A SwiGLU hidden Linear (2h, in) splits by rows: [:h] -> x1, [h:] -> x2.

    Raises:
        ConversionError: If the layer counts or the SwiGLU flags differ.
    """
    from hepattn.models.activation import SwiGLU  # noqa: PLC0415

    linears = [m for m in tdense.net if isinstance(m, nn.Linear)]
    layers = [*ndense.hidden, ndense.final]
    if len(linears) != len(layers):
        raise ConversionError(f"structure mismatch: {len(linears)} torch Linears vs {len(layers)} keras Dense")
    gated = any(isinstance(m, SwiGLU) for m in tdense.net)
    if gated != ndense.gate:
        raise ConversionError(f"SwiGLU mismatch: torch {gated}, native {ndense.gate}")
    for i, (lin, layer) in enumerate(zip(linears, layers, strict=True)):
        weight = _np(lin.weight)
        bias = _np(lin.bias) if lin.bias is not None else None
        if gated and i < len(ndense.hidden):
            w1, w2 = np.split(weight, 2, axis=0)
            b1, b2 = np.split(bias, 2) if bias is not None else (None, None)
            assign_dense(layer, w1, b1)
            assign_dense(ndense.gates[i], w2, b2)
        else:
            assign_dense(layer, weight, bias)


@torch.no_grad()
def port_norm(tnorm: nn.Module, nnorm) -> None:
    """Torch LayerNorm (affine or not, e.g. FastLayerNorm) -> keras LayerNormalization; the affine flags must agree.

    Raises:
        ConversionError: For another norm type, epsilon or affine flags.
    """
    if not isinstance(tnorm, nn.LayerNorm):
        raise ConversionError(f"only LayerNorm is ported, got {type(tnorm).__name__}")
    if abs(tnorm.eps - nnorm.epsilon) >= 1e-12:
        raise ConversionError(f"epsilon differs: torch {tnorm.eps} vs keras {nnorm.epsilon}")
    affine = tnorm.elementwise_affine
    if nnorm.scale != affine or nnorm.center != (affine and tnorm.bias is not None):
        raise ConversionError(f"{nnorm.name}: affine flags differ from torch")
    if affine:
        _assign(nnorm.gamma, _np(tnorm.weight))
        if tnorm.bias is not None:  # LayerNorm(bias=False): native center=False, checked above
            _assign(nnorm.beta, _np(tnorm.bias))
    elif nnorm.variables:
        raise ConversionError(f"{nnorm.name}: a non-affine norm must have no variables")


def _port_projection(seq_proj, proj: torch.Tensor) -> None:
    """A (seq_len, k) Linformer projection into a native ``SeqProjection`` (``masked_linformer.SeqProjection.assign_full``).

    Raises:
        ConversionError: If the projection is not (seq_len, k).
    """
    full = _np(proj)
    if full.shape != (seq_proj.seq_len, seq_proj.k):
        raise ConversionError(f"{seq_proj.name}: projection {full.shape} != (seq_len, k) = {(seq_proj.seq_len, seq_proj.k)}")
    _assign(seq_proj.proj.kernel, full[: seq_proj.n])
    if seq_proj.tail is not None:
        _assign(seq_proj.tail, full[seq_proj.n :])
    elif full.shape[0] > seq_proj.n and _COVERAGE is not None:
        _COVERAGE.dropped_rows[seq_proj.name] = full.shape[0] - seq_proj.n


def _native_attn_type(attn_type: str) -> str:
    """The native attention type of a torch one (``keras_native.encoder.COERCE_TO_TORCH``, without importing keras)."""
    return "torch" if attn_type in FLASH_TYPES else attn_type


def portable_attention(model: nn.Module) -> int:
    """Switch every non-linformer torch attention to the "torch" (SDPA) backend; returns the number of modules switched.

    ``flash`` / ``flash-varlen`` (org ``base.yaml``'s encoder) do not run on the CPU; the backend holds no parameters.
    ``change_attn_backends(model, "torch")`` raises for a linformer attention (org: "linformer owns trained projection
    parameters"), so those keep their backend.
    """
    from hepattn.models.attention import Attention  # noqa: PLC0415
    from hepattn.models.encoder import Encoder  # noqa: PLC0415

    switched = 0
    for module in model.modules():
        if isinstance(module, Encoder | Attention) and module.attn_type not in {"linformer", "torch"}:
            module.set_backend("torch")
            switched += 1
    return switched


@torch.no_grad()
def port_attention(tattn, nattn) -> None:
    """Torch Attention (torch or linformer backend) -> native Attention of the same attn_type.

    Linformer: ``proj_k`` / ``proj_v`` (seq_len, k) go to the core's E / F. Rows past the key length are dropped when
    the native core has no tail (no renorm, e.g. decoder query self-attention): org never reads them there. The torch
    types ``flash`` / ``flash-varlen`` compare as ``torch``, as the native Encoder and decoder coerce them (same
    parameters, same math on the dense-mask path).

    Raises:
        RuntimeError: If the native linformer core is not built.
        ConversionError: If the attention type, shape or ``qkv_norm`` differ.
    """
    if _native_attn_type(tattn.attn_type) != nattn.attn_type:
        raise ConversionError(f"attn_type mismatch: torch {tattn.attn_type} vs native {nattn.attn_type}")
    if tattn.dim != nattn.dim or tattn.num_heads != nattn.num_heads:
        raise ConversionError("attention shape mismatch")
    w_q, w_k, w_v = np.split(_np(tattn.in_proj_weight), 3, axis=0)
    b_q = b_k = b_v = None
    if tattn.in_proj_bias is not None:
        b_q, b_k, b_v = np.split(_np(tattn.in_proj_bias), 3)
    assign_dense(nattn.q_proj, w_q, b_q)
    assign_dense(nattn.k_proj, w_k, b_k)
    assign_dense(nattn.v_proj, w_v, b_v)
    assign_dense(nattn.out_proj, _np(tattn.out_proj.weight), _np(tattn.out_proj.bias) if tattn.out_proj.bias is not None else None)
    if tattn.qkv_norm != nattn.qkv_norm:
        raise ConversionError("qkv_norm mismatch")
    if tattn.qkv_norm:
        port_norm(tattn.q_norm, nattn.q_norm)
        port_norm(tattn.k_norm, nattn.k_norm)
        port_norm(tattn.v_norm, nattn.v_norm)
    if tattn.value_residual and not tattn.is_first_layer:
        lin = tattn.value_residual_mix[0]
        assign_dense(nattn.value_residual_mix, _np(lin.weight), _np(lin.bias) if lin.bias is not None else None)
    if tattn.attn_type == "linformer":
        if nattn.core is None:
            raise RuntimeError(f"{nattn.name}: build the native attention before porting")
        _port_projection(nattn.core.E, tattn.proj_k)
        _port_projection(nattn.core.F, tattn.proj_v)


def port_any_attention(tattn, nfn) -> None:
    """Torch Attention (torch or linformer backend) -> native Attention (``port_attention`` covers both)."""
    port_attention(tattn, nfn)


@torch.no_grad()
def port_residual(tres, nres, fn_porter) -> None:
    from hepattn.models.encoder import LayerScale  # noqa: PLC0415

    if tres.post_norm != nres.post_norm:
        raise ConversionError("post_norm mismatch")
    if isinstance(tres.norm, nn.Identity) != (nres.norm is None):
        raise ConversionError("norm presence mismatch")
    if nres.norm is not None:
        port_norm(tres.norm, nres.norm)
    if isinstance(tres.ls, LayerScale) != (nres.ls is not None):
        raise ConversionError("LayerScale presence mismatch")
    if nres.ls is not None:
        _assign(nres.ls.gamma, _np(tres.ls.gamma))
    fn_porter(tres.fn, nres.fn)


def port_encoder_layer(tlayer, nlayer) -> None:
    port_residual(tlayer.attn, nlayer.attn, port_any_attention)
    port_residual(tlayer.dense, nlayer.dense, port_dense)


@torch.no_grad()
def port_encoder(tenc, nenc) -> None:
    if tenc.num_layers != nenc.num_layers:
        raise ConversionError("encoder depth mismatch")
    if (tenc.register_tokens is None) != (nenc.num_register_tokens is None):
        raise ConversionError("register token mismatch")
    if tenc.register_tokens is not None:
        _assign(nenc.register_tokens, _np(tenc.register_tokens))
    for tlayer, nlayer in zip(tenc.layers, nenc.layers, strict=True):
        port_encoder_layer(tlayer, nlayer)


@torch.no_grad()
def port_input_net(tnet, nnet) -> None:
    port_dense(tnet.net, nnet.net)
    if (tnet.posenc is None) != (nnet.posenc is None):
        raise ConversionError("posenc presence mismatch")
    if tnet.posenc is not None:
        _assign(nnet.posenc.B, _np(tnet.posenc.B))


@torch.no_grad()
def port_decoder(tdec, ndec) -> None:
    """Torch MaskFormerDecoder -> native MaskFormerDecoder (initial queries + every layer's residual blocks).

    Raises:
        ConversionError: If the depth or ``bidirectional_ca`` differ.
    """
    if len(tdec.decoder_layers) != len(ndec.layers):
        raise ConversionError("decoder depth mismatch")
    _assign(ndec.initial_queries, _np(tdec.initial_queries))
    for tlayer, nlayer in zip(tdec.decoder_layers, ndec.layers, strict=True):
        port_residual(tlayer.q_ca, nlayer.q_ca, port_any_attention)
        port_residual(tlayer.q_sa, nlayer.q_sa, port_any_attention)
        port_residual(tlayer.q_dense, nlayer.q_dense, port_dense)
        if tlayer.bidirectional_ca != nlayer.bidirectional_ca:
            raise ConversionError("bidirectional_ca mismatch")
        if tlayer.bidirectional_ca:
            port_residual(tlayer.kv_ca, nlayer.kv_ca, port_any_attention)
            port_residual(tlayer.kv_dense, nlayer.kv_dense, port_dense)


TASK_NETS = ("net", "object_net", "constituent_net", "node_net")


@torch.no_grad()
def port_task(ttask, ntask) -> int:
    """Port every Dense head of a task (by attribute name); returns the number ported.

    Raises:
        ConversionError: If a head exists on one side only.
    """
    count = 0
    for attr in TASK_NETS:
        tnet = getattr(ttask, attr, None)
        nnet = getattr(ntask, attr, None)
        if isinstance(tnet, nn.Identity):
            tnet = None
        if (tnet is None) != (nnet is None):
            raise ConversionError(f"{attr}: presence differs (torch {tnet is not None}, native {nnet is not None})")
        if tnet is not None:
            port_dense(tnet, nnet)
            count += 1
    return count


@torch.no_grad()
def port_maskformer(tmodel, nmodel) -> None:
    """Torch MaskFormer -> native MaskFormer (built). Task order and names must match.

    Raises:
        ConversionError: If the task order differs or a task has no Dense head.
    """
    for tnet, nnet in zip(tmodel.input_nets, nmodel.input_nets, strict=True):
        port_input_net(tnet, nnet)
    port_encoder(tmodel.encoder, nmodel.encoder)
    port_decoder(tmodel.decoder, nmodel.decoder)
    for ttask, ntask in zip(tmodel.tasks, nmodel.tasks, strict=True):
        if ttask.name != ntask.task_name:
            raise ConversionError(f"task order differs: {ttask.name} vs {ntask.task_name}")
        if port_task(ttask, ntask) == 0:
            raise ConversionError(f"task {ttask.name}: no Dense ported")


# ----------------------------------------------------------------------------------------------- coverage checks


def quantizer_state_ids(nmodel) -> set[int]:
    """Ids of the HGQ2 state variables of ``nmodel``: every quantizer variable, and each leaf's ``beta`` / ``ebops``."""
    from hgq.layers.core.base import QLayerBase  # noqa: PLC0415
    from hgq.quantizer.internal.base import TrainableQuantizerBase  # noqa: PLC0415
    from hgq.quantizer.quantizer import Quantizer  # noqa: PLC0415

    ids: set[int] = set()
    for layer in nmodel._flatten_layers(include_self=True, recursive=True):  # noqa: SLF001
        if isinstance(layer, Quantizer | TrainableQuantizerBase):
            ids |= {id(v) for v in layer.variables}
        if isinstance(layer, QLayerBase):
            ids |= {id(v) for v in (getattr(layer, "_beta", None), getattr(layer, "_ebops", None)) if v is not None}
    return ids


def _loss_weight_owner(name: str, nmodel):
    """The native task whose ``loss_weights`` constant mirrors the torch buffer ``name`` (``[decoder.]tasks.<i>.loss_weights``).

    Raises:
        ConversionError: If ``name`` is not a task buffer.
    """
    match = re.search(r"tasks\.(\d+)\.loss_weights$", name)
    if match is None:
        raise ConversionError(f"{name}: a loss_weights buffer outside a task")
    return nmodel.tasks[int(match.group(1))]


def _check_loss_weights(name: str, value: np.ndarray, nmodel) -> None:
    """Compare a torch ``loss_weights`` buffer with the native constant.

    Raises:
        ConversionError: If they differ (another class-weight config).
    """
    expected = np.asarray(_loss_weight_owner(name, nmodel).loss_weights, dtype=np.float32)
    if value.shape != expected.shape or not np.array_equal(value.astype(np.float32), expected):
        raise ConversionError(f"{name}: {value.tolist()} differs from the native task constant {expected.tolist()} (class weights of the config)")


def port_torch_model(tmodel: nn.Module, nmodel) -> dict:
    """Port a torch MaskFormer into a built native model; returns the coverage report.

    Raises:
        ConversionError: If a torch parameter / buffer is not read, or a native weight is not written (the quantizer
            state may stay unwritten, it does not exist in a float model).
    """
    global _COVERAGE
    coverage = _Coverage(tmodel)
    _COVERAGE = coverage
    try:
        port_maskformer(tmodel, nmodel)
    finally:
        _COVERAGE = None
    state = tmodel.state_dict()
    unread = sorted(set(state) - coverage.read)
    ignored = [name for name in unread if name.endswith(".loss_weights")]
    for name in ignored:
        _check_loss_weights(name, state[name].numpy(), nmodel)
    unread = [name for name in unread if name not in ignored]
    if unread:
        raise ConversionError(f"{len(unread)} torch tensors were not ported: {unread[:8]}")
    qstate = quantizer_state_ids(nmodel)
    unwritten = [v for v in nmodel.variables if id(v) not in coverage.written]
    weights_unwritten = [v.path for v in unwritten if id(v) not in qstate]
    if weights_unwritten:
        raise ConversionError(f"{len(weights_unwritten)} native weights were not written: {weights_unwritten[:8]}")
    return {
        "source": TORCH_SOURCE,
        "torch_tensors": len(state),
        "torch_tensors_read": len(coverage.read),
        "ignored": ignored,
        "native_variables": len(nmodel.variables),
        "native_written": len(coverage.written),
        "native_kept_initial": len(unwritten),
        "dropped_rows": coverage.dropped_rows,
    }


# ----------------------------------------------------------------------------------------------- hybrid state -> native


def _segments(path: str) -> list[str]:
    return path.split("/")


def _canonical(segments: list[str]) -> str:
    return "/".join(_AUTO_SUFFIX.sub("", s) for s in segments)


def _suffix_order(segments: list[str]) -> tuple[int, ...]:
    return tuple(int(m.group(1)) if (m := _AUTO_SUFFIX.search(s)) else -1 for s in segments)


def hybrid_keras_state(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a hybrid state dict into (keras variable path -> tensor, other torch key -> tensor).

    Raises:
        ConversionError: If two module paths hold different values for one keras variable path.
    """
    keras_vars: dict[str, torch.Tensor] = {}
    torch_vars: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if KERAS_PARAM_MARK in key:
            path = key.split(KERAS_PARAM_MARK, 1)[1]
            if path in keras_vars:
                if not torch.equal(keras_vars[path], value):
                    raise ConversionError(f"{path}: different values under two module paths")
                continue
            keras_vars[path] = value
        else:
            torch_vars[key] = value
    return keras_vars, torch_vars


def _match_native(nmodel, groups: dict[str, list[list[str]]]) -> dict[str, list]:
    """Native variables by the hybrid canonical path they match (the longest leaf-relative suffix), in suffix order."""
    found: dict[str, list] = defaultdict(list)
    for var in nmodel.variables:
        segs = _segments(var.path)
        for i in range(len(segs)):
            canon = _canonical(segs[i:])
            if canon in groups:
                found[canon].append((_suffix_order(segs[i:]), var))
                break
    return {canon: [var for _, var in sorted(items, key=operator.itemgetter(0))] for canon, items in found.items()}


def port_hybrid_state(state: dict[str, torch.Tensor], nmodel) -> dict:
    """Copy a hybrid ``KerasMaskFormer`` state dict (``register_keras_parameters`` done) into a built native model.

    Raises:
        ConversionError: For a hybrid variable without a native twin (or the reverse), a shape or dtype mismatch, a
            torch key outside the known set, or class weights that differ from the target config.
    """
    keras_vars, torch_vars = hybrid_keras_state(state)
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for path in keras_vars:
        groups[_canonical(_segments(path))].append(_segments(path))
    native = _match_native(nmodel, groups)
    written: set[int] = set()
    casts: dict[str, int] = defaultdict(int)
    saturated: list[str] = []

    def put(var, value: torch.Tensor, what: str) -> None:
        array = value.detach().cpu().numpy()
        dtype = np.dtype(var.dtype)
        if tuple(var.shape) != array.shape:
            raise ConversionError(f"{what} -> {var.path}: source shape {array.shape}, native {tuple(var.shape)}")
        if dtype != array.dtype:
            # the hybrid keeps HGQ2's uint32 ``ebops`` counter as int64 (torch backend); integer -> integer only
            if not (np.issubdtype(array.dtype, np.integer) and np.issubdtype(dtype, np.integer)):
                raise ConversionError(f"{what} -> {var.path}: source dtype {array.dtype}, native {dtype}")
            info = np.iinfo(dtype)
            if array.size and (array.min() < info.min or array.max() > info.max):
                if not what.endswith("/ebops"):
                    raise ConversionError(f"{what}: values outside {dtype}")
                saturated.append(what)  # the JAX backend's uint32 counter saturates too (ebops.EbopsLeaves.stored_ebops)
                array = np.clip(array, info.min, info.max)
            casts[f"{array.dtype}->{dtype}"] += 1
            array = array.astype(dtype)
        var.assign(array)
        written.add(id(var))

    missing = []
    for canon, members in groups.items():
        targets = native.get(canon, [])
        if len(targets) != len(members):
            missing.append(f"{canon}: {len(members)} hybrid, {len(targets)} native")
            continue
        for segs, var in zip(sorted(members, key=_suffix_order), targets, strict=True):
            path = "/".join(segs)
            put(var, keras_vars[path], path)
    if missing:
        raise ConversionError(f"{len(missing)} hybrid variables without a native twin: {missing[:6]}")

    structural = {
        "encoder.register_tokens": lambda m: m.encoder.register_tokens,
        "decoder.initial_queries": lambda m: m.decoder.initial_queries,
    }
    ignored = []
    for key, value in torch_vars.items():
        posenc = re.fullmatch(r"input_nets\.(\d+)\.posenc\.B", key)
        if key in structural:
            put(structural[key](nmodel), value, key)
        elif posenc is not None:
            put(nmodel.input_nets[int(posenc.group(1))].posenc.B, value, key)
        elif HYBRID_IGNORED.search(key):
            if key.endswith("loss_weights"):
                _check_loss_weights(key, value.numpy(), nmodel)
            ignored.append(key)
        else:
            raise ConversionError(f"{key}: a torch tensor of the hybrid with no native mapping")
    unwritten = [v.path for v in nmodel.variables if id(v) not in written]
    if unwritten:
        raise ConversionError(f"{len(unwritten)} native variables have no hybrid source: {unwritten[:8]}")
    return {
        "source": HYBRID_SOURCE,
        "hybrid_keras_variables": len(keras_vars),
        "hybrid_state_keys": len(state),
        "torch_tensors": sorted(torch_vars),
        "ignored": sorted(ignored),
        "native_variables": len(nmodel.variables),
        "native_written": len(written),
        "integer_casts": dict(casts),
        "saturated_ebops": saturated,
    }


# ----------------------------------------------------------------------------------------------- files


def read_lightning_state(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """The ``model.`` part of a Lightning checkpoint's ``state_dict`` (prefix removed) and its step / epoch info.

    A bare state dict (no ``state_dict`` key, e.g. ``port_to_keras`` output) is returned as is.

    Raises:
        ConversionError: If the checkpoint has no ``model.`` keys.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    info = {key: ckpt.get(key) for key in ("epoch", "global_step", "pytorch-lightning_version")} if "state_dict" in ckpt else {}
    raw = ckpt.get("state_dict", ckpt)
    state = {key.removeprefix("model."): value for key, value in raw.items() if key.startswith("model.")}
    if not state and "state_dict" not in ckpt:
        state = dict(raw)
    if not state:
        raise ConversionError(f"{path}: no 'model.' entries in the state dict")
    return state, info


def source_kind(model_entry: dict) -> str:
    """``torch`` (``hepattn.models.MaskFormer``) or ``hybrid`` (``KerasMaskFormer``) from ``model.model.class_path``.

    Raises:
        ConversionError: For any other class.
    """
    name = str(model_entry.get("class_path", "")).rsplit(".", 1)[-1]
    if name == "MaskFormer":
        return TORCH_SOURCE
    if name == "KerasMaskFormer":
        return HYBRID_SOURCE
    raise ConversionError(f"model.model.class_path {model_entry.get('class_path')!r}: neither MaskFormer nor KerasMaskFormer")


def source_model_section(config_paths: list[str | Path]) -> dict:
    """``model.model`` of the source config(s), merged as LightningCLI merges them, with scale dicts made absolute."""
    from hepattn.keras_native import config as nconfig  # noqa: PLC0415

    sources = [Path(p) for p in config_paths]
    cfg = nconfig.read_yaml(sources)
    resolved = nconfig._resolve_scale_dicts(cfg["model"], nconfig._search_dirs(sources))  # noqa: SLF001
    return resolved["model"]


def without_device_solver(model_entry: dict) -> tuple[dict, str | None]:
    """A copy of a model entry without the matcher's GPU ``device_solver`` (and the value dropped).

    The torch source model is only loaded and run forward here; its matcher is never called. A run trained with
    ``device_solver: jv`` (``torch-linear-assignment``, a CUDA build) would otherwise fail to instantiate on a host
    without that build (the Matcher checks it in ``__init__``).
    """
    entry = copy.deepcopy(model_entry)
    matcher = (entry.get("init_args") or {}).get("matcher")
    args = matcher.get("init_args") if isinstance(matcher, dict) else None
    return entry, (args.pop("device_solver", None) if isinstance(args, dict) else None)


def instantiate_torch_model(model_entry: dict) -> nn.Module:
    """Instantiate the torch MaskFormer of a ``class_path`` / ``init_args`` entry with jsonargparse (as LightningCLI)."""
    from jsonargparse import ArgumentParser  # noqa: PLC0415

    parser = ArgumentParser()
    parser.add_subclass_arguments(nn.Module, "model")
    return parser.instantiate_classes(parser.parse_object({"model": model_entry})).model


def sample_batch(rc, num_events: int = 2) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Dummy CLIC events at the target config's sizes (to build the native model; also the verification batch)."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from hepattn.experiments.clic.native_data import make_dataset  # noqa: PLC0415

    dataset = make_dataset(rc.data, "val", dummy_data=True, num_events=num_events)
    inputs, targets = next(iter(DataLoader(dataset, batch_size=num_events)))
    return {k: v.numpy() for k, v in inputs.items()}, {k: v.numpy() for k, v in targets.items()}


def _flatten(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


@torch.no_grad()
def compare_forward(tmodel: nn.Module, nmodel, inputs: dict[str, np.ndarray]) -> dict[str, float]:
    """Max |native - torch| per output (eval mode) on ``inputs``; padded mask-logit slots (|value| >= 1e30) excluded.

    Raises:
        ConversionError: If the output keys differ.
    """
    tin = {k: torch.from_numpy(v.astype(np.int64) if v.dtype == np.int32 else v) for k, v in inputs.items()}
    ref = {k: v.numpy() for k, v in _flatten(tmodel.eval()(tin)).items()}
    got = {k: np.asarray(v) for k, v in _flatten(nmodel(inputs, training=False)).items()}
    if set(ref) != set(got):
        raise ConversionError(f"output keys differ: {sorted(set(ref) ^ set(got))}")
    out = {}
    for key, value in ref.items():
        if value.dtype == bool:
            out[key] = float((value != got[key]).sum())
            continue
        keep = np.abs(value) < 1e30
        out[key] = float(np.abs(value[keep].astype(np.float64) - got[key][keep]).max()) if keep.any() else 0.0
    return out


def renorm_lane_check(nmodel) -> dict | None:
    """The premise of the quantized masked-Linformer renorm lane (``masked_linformer.renorm_bits``) on these weights.

    Over every E / F projection of a quantized renorm core: the largest column energy ``S_c = sum_rows E^2`` and the
    number of nonzero ``E[:n]`` entries whose energy fraction ``W = E^2 / S_c`` the kept-lane quantizer rounds to 0.
    The renorm does not see such an entry: a mask row that selects only such entries reaches the rsqrt floor although
    its projection is not zero (``docs/hgq/FPGA_STATUS.md`` footnote 5). The count is 0 while ``S_c < 32``: the
    smallest nonzero ``E^2`` is ``2**-(2 cap)``, half a lane step is ``2**-(2 cap + 5)``. ``None`` for a model without
    quantized renorm cores (a float target has no lane).
    """
    from hepattn.keras_native.masked_linformer import AttendCore, ProjectCore  # noqa: PLC0415

    projections, s_max, dropped, worst = 0, 0.0, 0, None
    for core in nmodel._flatten_layers(include_self=False, recursive=True):  # noqa: SLF001
        if not isinstance(core, AttendCore | ProjectCore) or not core.renorm:
            continue
        for proj, kept in ((core.E, core.kept_k), (core.F, core.kept_v)):
            if not hasattr(kept, "iq"):  # float leaves
                return None
            head, tail = (None if t is None else np.asarray(t, np.float64) for t in proj.rows(False))
            column = (head**2).sum(0) + (0.0 if tail is None else (tail**2).sum(0))
            energy = np.asarray(kept.iq.quantizers[1](proj.energy(False)[None], training=False))[0]
            projections += 1
            dropped += int(((head != 0) & (energy == 0)).sum())
            if column.max() > s_max:
                s_max, worst = float(column.max()), proj.path
    if not projections:
        return None
    return {"projections": projections, "s_c_max": s_max, "s_c_max_at": worst, "dropped_entries": dropped}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(
    config: list[str | Path],
    ckpt_path: str | Path,
    *,
    target_config: list[str | Path] | None = None,
    verify: bool = True,
) -> tuple[Any, Any, dict]:
    """Build the target native model and fill it from the checkpoint. Returns ``(rc, model, report)``.

    ``verify``: for a torch source into a float target, also report the forward difference on two dummy events.

    Raises:
        ConversionError: If the checkpoint does not map one-to-one onto the target model (see the module docstring).
    """
    from hepattn.keras_native.config import load_config  # noqa: PLC0415

    model_entry = source_model_section(config)
    kind = source_kind(model_entry)
    rc = load_config(target_config or config)
    inputs, _ = sample_batch(rc)
    nmodel = rc.build_model()
    nmodel(inputs, training=False)
    state, info = read_lightning_state(ckpt_path)
    dropped_solver = None
    if kind == TORCH_SOURCE:
        model_entry, dropped_solver = without_device_solver(model_entry)
        tmodel = instantiate_torch_model(model_entry)
        portable_attention(tmodel)  # for the verify forward on the CPU
        result = tmodel.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:  # strict raises; kept for type checkers
            raise ConversionError(f"state dict mismatch: {result}")
        report = port_torch_model(tmodel, nmodel)
        if verify and rc.quant.spec is None:
            report["max_abs_diff"] = compare_forward(tmodel, nmodel, inputs)
    else:
        report = port_hybrid_state(state, nmodel)
    lane = renorm_lane_check(nmodel)
    if lane is not None:
        report["renorm_lane"] = lane
        if lane["dropped_entries"]:
            warnings.warn(f"renorm lane premise fails: {lane}; see docs/hgq/FPGA_STATUS.md footnote 5", stacklevel=2)
    if dropped_solver is not None:
        report["source_device_solver_dropped"] = dropped_solver
    report["checkpoint"] = {"path": str(ckpt_path), "sha256": _sha256(ckpt_path), **info}
    report["config"] = [str(p) for p in config]
    report["target_config"] = [str(p) for p in (target_config or config)]
    report["config_hash"] = rc.model_hash
    return rc, nmodel, report


def write_native_checkpoint(path: str | Path, rc, nmodel, report: dict, *, overwrite: bool = False) -> Path:
    """The model state as a keras_native checkpoint without optimizer state (step 0); the report goes in the metadata."""
    from hepattn.keras_native import checkpoint as nckpt  # noqa: PLC0415

    tv, ntv, _ = nckpt.current_state(nmodel)
    meta = {"step": 0, "epoch": 0, "config_hash": rc.model_hash, "converted_from": report}
    return nckpt.save_checkpoint(path, nmodel, None, (tv, ntv, []), meta, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="convert_to_native", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", required=True, help="the run's saved config.yaml (repeat to merge in order)")
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint (.ckpt) of that run")
    parser.add_argument("--out", required=True, help="output keras_native checkpoint (.npz; a .json sidecar is written next to it)")
    parser.add_argument("--target-config", action="append", help="config of the native model to fill (default: --config)")
    parser.add_argument("--report", help="also write the conversion report (JSON) here")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    parser.add_argument("--no-verify", action="store_true", help="skip the torch-vs-native forward comparison")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the converter.

    Raises:
        SystemExit: If ``KERAS_BACKEND`` names a backend other than JAX.
    """
    args = build_parser().parse_args(argv)
    backend = os.environ.setdefault("KERAS_BACKEND", "jax")
    if backend != "jax":
        raise SystemExit(f"convert_to_native builds the keras_native model on JAX; KERAS_BACKEND is {backend!r}")
    rc, nmodel, report = convert(args.config, args.ckpt, target_config=args.target_config, verify=not args.no_verify)
    out = write_native_checkpoint(args.out, rc, nmodel, report, overwrite=args.overwrite)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print("RESULT " + json.dumps({"out": str(out), **report}, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
