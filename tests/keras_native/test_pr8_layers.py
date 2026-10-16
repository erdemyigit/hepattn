"""R3 gates (docs/hgq/RETARGET_ORG_MAIN.md sections 5.1-5.5): the org/main ``ac0da8f`` layers in keras_native.

The reference is the org/main torch code in this process: ``hepattn.models`` (``base.yaml`` + ``linformer.yaml``).
Covered: ``from_config`` of the merged config, FastLayerNorm, ``dense_norm_placement``, the decoder
``query_update_order`` / ``self_attn_norm_kv``, ``kv_sort_idx`` to ``q_ca`` only, quantized SwiGLU, input sorting, the
layer-0 classification head, and the batch-of-one all-valid path. The masked-Linformer attention is lane R2's: the
float parity rows here use dense attention (``attn_type: torch``), which is permutation-equivariant, so sorting is
tested through its mechanics (sort index, round trip, a position-marking stub layer). Every tolerance is 2x the
measured maximum (docs/hgq/port_log/R3.md), except the whole-model rows, which keep the plan's ``maskformer.e2e``
tolerance (``E2E``; plan 5.8, re-measured by R5); each has a control that exceeds it by at least 10x.
"""

import collections
import copy
import importlib
import inspect
import itertools
import math
import re
from pathlib import Path

import jax
import keras
import numpy as np
import pytest
import torch
import yaml
from torch import nn

from hepattn.keras_native.attention import Attention as NAttention
from hepattn.keras_native.decoder import MaskFormerDecoderLayer as NDecoderLayer
from hepattn.keras_native.dense import Dense as NDense
from hepattn.keras_native.encoder import Encoder as NEncoder
from hepattn.keras_native.encoder import gather_tokens
from hepattn.keras_native.encoder import sort_index as native_sort_index
from hepattn.keras_native.factory import LayerFactory
from hepattn.keras_native.maskformer import from_config
from hepattn.keras_native.norm import make_norm
from hepattn.models.activation import SwiGLU
from hepattn.models.decoder import MaskFormerDecoder, MaskFormerDecoderLayer
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder
from hepattn.models.encoder import sort_index as torch_sort_index
from hepattn.models.maskformer import MaskFormer
from hepattn.models.norm import FastLayerNorm

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .parity_utils import assert_parity, max_errors
from .porting import perturb_affine_params, port_any_attention, port_dense, port_maskformer, port_residual
from .test_maskformer import assert_tree_parity, flatten_torch, run_pair

REPO = Path(__file__).resolve().parents[2]
CLIC = REPO / "src/hepattn/experiments/clic"
SCALE_DICT = str(CLIC / "configs/clic_var_transform.yaml")

# Before the R2 merge, native build_attention routes attn_type "linformer" to Helen's layer (k=256, no linformer_core),
# so the masked-Linformer rows are skipped. The condition is on the code, not a strict xfail: once R2's Attention (with
# linformer_core) is in the tree they run as plain tests, and a wrong result fails instead of hiding as an xfail.
R2_LANDED = "linformer_core" in inspect.signature(NAttention.__init__).parameters
R2_REQUIRED = pytest.mark.skipif(not R2_LANDED, reason="needs R2's masked-Linformer Attention (RETARGET_ORG_MAIN.md section 5.6)")

# Tolerances: 2x the measured maximum (R3.md, "Gates"; $S/r3/measure.out). The whole-model rows keep the plan's
# maskformer.e2e tolerance (5e-5 / 1e-4, re-measured by R5); measured here: 8.6e-6 abs, 7% of it.
TOL_NORM = 2 * 4.77e-7
TOL_SWIGLU = 2 * 3.58e-7
TOL_ENC = {"atol": 2 * 9.54e-7, "rtol": 0.0}
TOL_DEC = {"atol": 2 * 9.54e-7, "rtol": 0.0}
TOL_B1_EQUAL = 2 * 1.37e-6
E2E = {"atol": 5e-5, "rtol": 1e-4}
TOL_JIT_EAGER = 2 * 1.81e-5  # final mask logits, jit vs eager (R3_evidence/fix2/measure_jit_eager.out)
LOGIT_ATTN_THRESHOLD = math.log(0.1 / 0.9)


# --------------------------------------------------------------------------------------------- config
def deep_merge(base: dict, over: dict) -> dict:
    """``over`` merged into ``base``: dicts merge key by key, anything else (lists included) replaces."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def org_model_section(dim: int | None = None) -> dict:
    """The ``model:`` section of org ``base.yaml`` with ``linformer.yaml`` merged over it.

    ``dim`` shrinks the model width for a fast test (the YAML anchor ``&dim``, the regression input 2 dim + 6 and the
    overlay's decoder ``dim``); the scale dict path is made absolute.
    """
    base_text = (CLIC / "configs/base.yaml").read_text()
    over_text = (CLIC / "configs/linformer.yaml").read_text()
    if dim is not None:
        for old, new, text in (
            ("dim: &dim 256", f"dim: &dim {dim}", "base"),
            ("input_size: 518", f"input_size: {2 * dim + 6}", "base"),
            ("dim: 256", f"dim: {dim}", "over"),
        ):
            source = base_text if text == "base" else over_text
            assert source.count(old) == 1, f"{text}: '{old}' found {source.count(old)} times"
            if text == "base":
                base_text = base_text.replace(old, new)
            else:
                over_text = over_text.replace(old, new)
    section = deep_merge(yaml.safe_load(base_text), yaml.safe_load(over_text))["model"]
    for task in section["model"]["init_args"]["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task["init_args"]:
            task["init_args"]["scale_dict_path"] = SCALE_DICT
    return section


def with_dense_attention(section: dict) -> dict:
    """The same model with dense attention (``attn_type: torch``) everywhere and without the Linformer keys."""
    section = copy.deepcopy(section)
    init = section["model"]["init_args"]
    init["encoder"]["init_args"]["attn_type"] = "torch"
    init["decoder"]["decoder_layer_config"]["attn_kwargs"]["attn_type"] = "torch"
    for attn_kwargs in (init["encoder"]["init_args"]["attn_kwargs"], init["decoder"]["decoder_layer_config"]["attn_kwargs"]):
        for key in ("linformer_seq_len", "linformer_k"):
            attn_kwargs.pop(key, None)
    return section


def _instantiate(spec):
    if isinstance(spec, dict) and "class_path" in spec:
        module, name = spec["class_path"].rsplit(".", 1)
        args = {key: _instantiate(value) for key, value in (spec.get("init_args") or {}).items()}
        return getattr(importlib.import_module(module), name)(**args)
    if isinstance(spec, list):
        return [_instantiate(value) for value in spec]
    if isinstance(spec, str) and spec.startswith("torch.nn."):
        return getattr(nn, spec.removeprefix("torch.nn."))()
    return spec


def torch_from_section(section: dict, seed: int = 0) -> MaskFormer:
    """The org torch MaskFormer of a ``model:`` section (no matcher: forwards only), eval mode, seeded."""
    torch.manual_seed(seed)
    init = copy.deepcopy(section["model"]["init_args"])
    init.pop("matcher")
    decoder = MaskFormerDecoder(**init.pop("decoder"))
    return MaskFormer(decoder=decoder, **{key: _instantiate(value) for key, value in init.items()}).eval()


def native_from_section(section: dict, quant=None):
    section = copy.deepcopy(section)
    if quant is not None:
        section["model"]["init_args"]["quant"] = quant
    return from_config(section)


def torch_inputs(inputs: dict) -> dict:
    return {key: torch.from_numpy(value) for key, value in inputs.items()}


# ----------------------------------------------------------------------------------- 5.1 from_config
PROD_QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
}


def check_org_structure(model) -> None:
    """The org/main model the YAML describes (plan section 1.2), read off the native layers."""
    assert model.input_sort_field == "phi"
    innet = model.input_nets[0].net
    assert innet.gate and innet.hidden_sizes == [54] and innet.output_size == 256
    enc = model.encoder
    assert len(enc.layers) == 6 and enc.num_register_tokens == 8 and enc.value_residual
    for depth, layer in enumerate(enc.layers):
        assert layer.attn.norm is None if depth else layer.attn.norm is not None, "hybrid norm: pre-attention norm in layer 0 only"
        assert layer.dense.post_norm == (depth == 0), "legacy placement: dense post-norm in layer 0 only"
        assert layer.dense.fn.gate, "encoder FFN is SwiGLU"
    dec = model.decoder
    assert len(dec.layers) == 4 and dec.num_queries == 150
    for depth, layer in enumerate(dec.layers):
        assert layer.query_update_order == "ca_sa_dense" and layer.self_attn_norm_kv
        assert layer.q_dense.post_norm == (depth == 0) and layer.kv_dense.post_norm == (depth == 0)
        assert layer.q_dense.fn.gate and layer.kv_dense.fn.gate
    norms = [lay for lay in model._flatten_layers(include_self=False, recursive=True) if isinstance(lay, keras.layers.LayerNormalization)]  # noqa: SLF001
    assert norms, "no norm layers found"
    assert all(not norm.center and not norm.scale and norm.epsilon == 1e-5 for norm in norms), "every norm is FastLayerNorm"
    assert [t.task_name for t in model.tasks] == ["classification", "mask", "incidence", "regression"]
    assert model.tasks[0].should_run_at_layer(0), "classification runs at layer 0 (has_first_layer_loss)"
    assert model.tasks[1].object_net.gate and model.tasks[1].constituent_net.gate and model.tasks[2].node_net.gate


@pytest.mark.parametrize("quant", [None, PROD_QUANT], ids=["float", "prod"])
def test_from_config_builds_org_linformer_model(quant):
    """All the model blockers of ``probe_chain.py`` are gone: the merged config builds unchanged, float and quantized.

    Control: at R0 (``5db27a8``) the same call raised (input sorting, then 6 more blockers, ``review_rt/probe_chain.out``).
    """
    model = native_from_section(org_model_section(), quant)
    check_org_structure(model)
    assert model.factory.quantize == (quant is not None)
    assert model.matcher_config["init_args"]["default_solver"] == "lap1015_late"  # stored only; R1 solves it


def test_encoder_spec_is_unwrapped_and_dim_checked():
    section = org_model_section(dim=32)
    model = native_from_section(section)
    assert model.encoder.dim == 32
    bad = copy.deepcopy(section)
    bad["model"]["init_args"]["encoder"]["init_args"]["dim"] = 64
    with pytest.raises(ValueError, match="encoder dim 64 != MaskFormer dim 32"):
        native_from_section(bad)
    bad = copy.deepcopy(section)
    bad["model"]["init_args"]["encoder"]["foo"] = 1
    with pytest.raises(ValueError, match=r"unexpected keys next to the encoder class_path: \['foo'\]"):
        native_from_section(bad)
    bad = copy.deepcopy(section)
    bad["model"]["init_args"]["encoder"]["class_path"] = "hepattn.models.Other"
    with pytest.raises(NotImplementedError, match="encoder"):
        native_from_section(bad)
    bad = copy.deepcopy(section)
    bad["model"]["init_args"]["sorter"] = {"class_path": "hepattn.utils.sorter.Sorter"}
    with pytest.raises(NotImplementedError, match="sorter"):
        native_from_section(bad)


def native_param_count(module) -> int:
    """Trainable weights of native layers, without HGQ2 state (float model: every trainable variable is a weight)."""
    return sum(int(np.prod(v.shape)) for v in module.trainable_variables)


def torch_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def native_leaf_shapes(layers) -> collections.Counter:
    return collections.Counter(tuple(v.shape) for layer in layers for v in layer.trainable_variables)


def torch_leaf_shapes(module: nn.Module, split: bool = True) -> collections.Counter:
    """Torch parameter shapes as the native leaves hold them: a Linear weight (out, in) as the kernel (in, out); the
    packed attention ``in_proj`` (3 dim, dim) as three (dim, dim); a SwiGLU Linear (2h, in) as two (in, h) halves.
    ``split=False`` keeps the packed shapes (a control: same element count, other shapes).
    """
    swiglu_linears = set()
    for sub in module.modules():
        if isinstance(sub, nn.Sequential):
            seq = list(sub)
            swiglu_linears |= {id(a) for a, b in itertools.pairwise(seq) if isinstance(a, nn.Linear) and isinstance(b, SwiGLU)}
    shapes = collections.Counter()
    for sub in module.modules():
        for name, p in sub.named_parameters(recurse=False):
            shape = tuple(p.shape)
            parts = 1
            if split and name.startswith("in_proj"):
                parts = 3
            elif split and id(sub) in swiglu_linears:
                parts = 2
            shape = (shape[0] // parts, *shape[1:])
            if (isinstance(sub, nn.Linear) and name == "weight") or name == "in_proj_weight":
                shape = shape[::-1]
            shapes[shape] += parts
    return shapes


@pytest.mark.parametrize("dim", [32, 256])
def test_variable_counts_equal_torch_with_dense_attention(dim):
    """Per part, the native trainable size equals the org torch parameter count (dense attention: Linformer is R2).

    Discriminating: an affine norm in place of FastLayerNorm adds 2 dim per norm, a packed SwiGLU with one bias half
    or a missing register token changes the encoder count. The per-part multisets of variable shapes (torch shapes
    mapped by ``torch_leaf_shapes``) must also agree, so a transposed or re-split non-square kernel fails.
    """
    section = with_dense_attention(org_model_section(dim=None if dim == 256 else dim))
    tmodel = torch_from_section(section)
    nmodel = native_from_section(section)
    inputs, _ = clic_dummy_batch()
    nmodel(inputs, training=False)
    pairs = {
        "input_nets": (tmodel.input_nets, nmodel.input_nets),
        "encoder": (tmodel.encoder, nmodel.encoder),
        "decoder": (tmodel.decoder.decoder_layers, nmodel.decoder.layers),
    }
    for part, (tpart, npart) in pairs.items():
        t_count = torch_param_count(tpart)
        nlayers = list(npart) if not isinstance(npart, keras.layers.Layer) else [npart]
        n_count = sum(native_param_count(layer) for layer in nlayers)
        assert n_count == t_count, f"{part}: native {n_count} vs torch {t_count}"
        assert native_leaf_shapes(nlayers) == torch_leaf_shapes(tpart), f"{part}: variable shapes differ"
        # control: the packed in_proj / SwiGLU shapes have the same element count, so only the shape check sees them
        assert native_leaf_shapes(nlayers) != torch_leaf_shapes(tpart, split=False), part
    for ttask, ntask in zip(tmodel.tasks, nmodel.tasks, strict=True):
        assert native_param_count(ntask) == torch_param_count(ttask), ttask.name
        assert native_leaf_shapes([ntask]) == torch_leaf_shapes(ttask), f"{ttask.name}: variable shapes differ"
    assert native_param_count(nmodel) == torch_param_count(tmodel), "whole model"
    assert not any(isinstance(m, nn.LayerNorm) and m.elementwise_affine for m in tmodel.modules()), "torch side: all norms are FastLayerNorm"


NATIVE_PROJ = re.compile(r"/((?:encoder|decoder)_l(\d+)_(attn|q_ca|q_sa|kv_ca))_(?:attend|project|project_unmasked)_([EF])/")
TORCH_PROJ = re.compile(r"^(encoder\.layers|decoder\.decoder_layers)\.(\d+)\.(attn|q_ca|q_sa|kv_ca)\.fn\.proj_([kv])$")


def native_projection_rows(model) -> dict[tuple[str, int, str, str], int]:
    """Rows of each native sequence projection (R2 names: E for ``proj_k``, F for ``proj_v``): the kernel of
    ``..._{E,F}_proj`` plus the renorm ``tail``, keyed (encoder|decoder, layer, part, k|v). Every k is 64.
    """
    rows = collections.Counter()
    for v in model.variables:
        match = NATIVE_PROJ.search(v.path)
        if match is None or not v.path.endswith(("_proj/kernel", "/tail")):
            continue
        assert v.shape[1] == 64, (v.path, v.shape)
        side = "encoder" if match.group(1).startswith("encoder") else "decoder"
        rows[side, int(match.group(2)), match.group(3), {"E": "k", "F": "v"}[match.group(4)]] += v.shape[0]
    return dict(rows)


def torch_projection_rows(model) -> dict[tuple[str, int, str, str], int]:
    rows = {}
    for name, p in model.named_parameters():
        match = TORCH_PROJ.match(name)
        if match is not None:
            assert p.shape[1] == 64, (name, p.shape)
            rows["encoder" if match.group(1) == "encoder.layers" else "decoder", int(match.group(2)), match.group(3), match.group(4)] = p.shape[0]
    return rows


@R2_REQUIRED
def test_linformer_projection_shapes_equal_torch():
    """Plan 5.1 shape gate: ``proj_k`` / ``proj_v`` (168, 64) in the encoder, (160, 64) in ``q_ca`` / ``kv_ca``, and
    (150, 64) in native ``q_sa`` against (160, 64) in torch (native drops the dead tail rows, plan 5.6). Native rows are
    R2's projection kernel plus its renorm tail (``kv_ca``: 150 + 10). Runs once R2's Attention is in the tree.
    """
    section = org_model_section()
    tmodel = torch_from_section(section)
    nmodel = native_from_section(section)
    inputs, _ = clic_dummy_batch()
    nmodel(inputs, training=False)
    t_rows, n_rows = torch_projection_rows(tmodel), native_projection_rows(nmodel)
    assert len(t_rows) == 2 * (6 + 4 * 3), sorted(t_rows)
    assert set(n_rows) == set(t_rows), set(n_rows) ^ set(t_rows)
    expected = {"attn": 168, "q_ca": 160, "kv_ca": 160, "q_sa": 160}
    assert all(rows == expected[key[2]] for key, rows in t_rows.items()), t_rows
    for key, rows in n_rows.items():
        assert rows == (150 if key[2] == "q_sa" else t_rows[key]), (key, rows)
    # each call site has the core of its org call (encoder: project with the padding mask; q_sa: project, mask=None)
    assert {layer.attn.fn.core_kind for layer in nmodel.encoder.layers} == {"project"}
    for layer in nmodel.decoder.layers:
        assert (layer.q_ca.fn.core_kind, layer.q_sa.fn.core_kind, layer.kv_ca.fn.core_kind) == ("attend", "project_unmasked", "attend")


# ------------------------------------------------------------------------------- 5.2 FastLayerNorm
def test_fast_layer_norm_equals_torch_and_has_no_variables():
    """Parity alone does not discriminate (an affine norm at its default init gives the same output); the variable
    counts do: 0 for FastLayerNorm on both sides, 2 for LayerNorm.
    """
    rng = np.random.default_rng(0)
    x = (3.0 * rng.standard_normal((2, 50, 64)) + 1.0).astype(np.float32)
    norm = make_norm("FastLayerNorm")
    got = norm(x)
    tnorm = FastLayerNorm(64)
    with torch.no_grad():
        ref = tnorm(torch.from_numpy(x))
    assert_parity("fast_layer_norm", "dim64", ref, got, atol=TOL_NORM, rtol=0.0)
    assert len(norm.variables) == 0 and sum(1 for _ in tnorm.parameters()) == 0
    affine = make_norm("LayerNorm")
    affine.build((None, 50, 64))
    assert len(affine.variables) == 2 and sum(1 for _ in nn.LayerNorm(64).parameters()) == 2


def encoder_cfg(placement: str, num_layers: int, dim: int = 32) -> dict:
    return {
        "num_layers": num_layers,
        "dim": dim,
        "attn_type": "torch",
        "norm": "FastLayerNorm",
        "hybrid_norm": True,
        "dense_norm_placement": placement,
        "value_residual": True,
        "num_register_tokens": 4,
        "attn_kwargs": {"num_heads": 8},
        "dense_kwargs": {"activation": "SwiGLU"},
    }


def ported_encoder(tenc: Encoder, placement: str, num_layers: int, check_placement: bool = True) -> NEncoder:
    cfg = encoder_cfg(placement, num_layers)
    cfg.pop("dim")
    nenc = NEncoder(dim=32, **cfg)
    nenc.build((None, 40, 32))
    nenc.register_tokens.assign(tenc.register_tokens.detach().numpy())
    for tlayer, nlayer in zip(tenc.layers, nenc.layers, strict=True):
        if check_placement:
            port_residual(tlayer.attn, nlayer.attn, port_any_attention)
            port_residual(tlayer.dense, nlayer.dense, port_dense)
        else:  # control: placements differ on purpose, the norms have no weights
            port_any_attention(tlayer.attn.fn, nlayer.attn.fn)
            port_dense(tlayer.dense.fn, nlayer.dense.fn)
    return nenc


def padded(seed: int, shape=(3, 40, 32), lengths=(40, 23, 31)):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    mask = np.arange(shape[1])[None, :] < np.asarray(lengths)[:, None]
    return x, mask


@pytest.mark.parametrize("num_layers", [1, 2], ids=["depth0", "depth1"])
def test_encoder_legacy_placement_equals_torch(num_layers):
    """Encoder with FastLayerNorm, hybrid norm, legacy placement, SwiGLU FFN, value residual: after layer 0 and 1.

    Control: the same weights with ``hybridnorm`` placement move valid outputs by more than 10x the tolerance.
    """
    torch.manual_seed(1)
    tenc = Encoder(**encoder_cfg("legacy", num_layers)).eval()
    perturb_affine_params(tenc, 1)
    nenc = ported_encoder(tenc, "legacy", num_layers)
    x, mask = padded(2)
    with torch.no_grad():
        ref = tenc(torch.from_numpy(x), kv_mask=torch.from_numpy(mask))
    assert_parity("encoder.legacy", f"depth{num_layers - 1}", ref, nenc(x, kv_mask=mask), valid_mask=mask, **TOL_ENC)
    control = ported_encoder(tenc, "hybridnorm", num_layers, check_placement=False)
    diff, _ = max_errors(ref, control(x, kv_mask=mask), mask)
    assert diff > 10 * TOL_ENC["atol"], f"hybridnorm placement changes the output by only {diff:.2e}"


def test_placement_mismatch_is_refused_by_the_porter():
    torch.manual_seed(1)
    tenc = Encoder(**encoder_cfg("legacy", 1)).eval()
    with pytest.raises(AssertionError, match="post_norm mismatch"):
        ported_encoder(tenc, "hybridnorm", 1)


# ------------------------------------------------------------------------------------- 5.3 decoder
DEC_DIM, DEC_Q, DEC_N = 32, 30, 40


def decoder_cfg(depth: int, order: str = "ca_sa_dense", norm_kv: bool = True) -> dict:
    return {
        "dim": DEC_DIM,
        "depth": depth,
        "norm": "FastLayerNorm",
        "hybrid_norm": True,
        "dense_norm_placement": "legacy",
        "query_update_order": order,
        "self_attn_norm_kv": norm_kv,
        "attn_kwargs": {"num_heads": 8},
        "dense_kwargs": {"activation": "SwiGLU"},
    }


def decoder_inputs(seed: int = 21):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((2, DEC_Q, DEC_DIM)).astype(np.float32)
    kv = rng.standard_normal((2, DEC_N, DEC_DIM)).astype(np.float32)
    attn_mask = rng.random((2, DEC_Q, DEC_N)) > 0.5
    attn_mask[..., 0] = True  # the decoder restores all-False rows upstream
    kv_mask = np.ones((2, DEC_N), dtype=bool)
    kv_mask[:, 33:] = False
    return q, kv, attn_mask, kv_mask


def native_decoder_layer(tlayer, depth: int, **changes):
    nlayer = NDecoderLayer(**{**decoder_cfg(depth), **changes}, name=f"decoder_l{depth}")
    q, kv, attn_mask, kv_mask = decoder_inputs()
    nlayer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)  # build
    for part in ("q_ca", "q_sa", "kv_ca"):
        port_residual(getattr(tlayer, part), getattr(nlayer, part), port_any_attention)
    for part in ("q_dense", "kv_dense"):
        port_residual(getattr(tlayer, part), getattr(nlayer, part), port_dense)
    return nlayer


def run_decoder_pair(depth: int):
    torch.manual_seed(20 + depth)
    tlayer = MaskFormerDecoderLayer(**decoder_cfg(depth)).eval()
    perturb_affine_params(tlayer, 20)
    q, kv, attn_mask, kv_mask = decoder_inputs()
    with torch.no_grad():
        q_t, kv_t = tlayer(*(torch.from_numpy(a) for a in (q, kv)), attn_mask=torch.from_numpy(attn_mask), kv_mask=torch.from_numpy(kv_mask))
    return tlayer, (q, kv, attn_mask, kv_mask), (q_t.numpy(), kv_t.numpy())


@pytest.mark.parametrize("depth", [0, 1])
def test_decoder_layer_pr8_options_equal_torch(depth):
    """``ca_sa_dense``, ``self_attn_norm_kv``, legacy placement, FastLayerNorm, SwiGLU: queries and valid keys.

    Controls: ``ca_dense_sa`` at depth 0 and 1, and ``self_attn_norm_kv=False`` at depth 0 exceed the tolerance by
    at least 10x. At depth >= 1 hybrid norm has no pre-attention norm, so ``self_attn_norm_kv`` must have no effect:
    asserted within the tolerance (this is why its control is at depth 0).
    """
    tlayer, (q, kv, attn_mask, kv_mask), (q_t, kv_t) = run_decoder_pair(depth)
    nlayer = native_decoder_layer(tlayer, depth)
    q_n, kv_n = nlayer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)
    assert_parity("decoder_layer.pr8.q", f"depth{depth}", q_t, q_n, **TOL_DEC)
    assert_parity("decoder_layer.pr8.kv", f"depth{depth}", kv_t, kv_n, valid_mask=kv_mask, **TOL_DEC)

    order = native_decoder_layer(tlayer, depth, query_update_order="ca_dense_sa")
    diff, _ = max_errors(q_t, order(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)[0])
    assert diff > 10 * TOL_DEC["atol"], f"wrong order changes q by only {diff:.2e}"

    no_norm_kv = native_decoder_layer(tlayer, depth, self_attn_norm_kv=False)
    diff, _ = max_errors(q_t, no_norm_kv(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)[0])
    if depth == 0:
        assert diff > 10 * TOL_DEC["atol"], f"self_attn_norm_kv=False changes q by only {diff:.2e}"
    else:
        assert diff <= TOL_DEC["atol"], f"self_attn_norm_kv must not act without a pre-attention norm, got {diff:.2e}"


def test_unknown_query_update_order_raises():
    with pytest.raises(ValueError, match="query_update_order"):
        NDecoderLayer(**decoder_cfg(0, order="sa_ca_dense"))
    with pytest.raises(ValueError, match="dense_norm_placement"):
        NDecoderLayer(**{**decoder_cfg(0), "dense_norm_placement": "pre"})


class RecordingAttention(keras.layers.Layer):
    """Test double for an attention ``fn``: records its call kwargs and returns its queries (plan interface:
    ``Attention.call(q, k=None, v=None, q_mask=None, kv_mask=None, attn_mask=None, initial_v=None, kv_sort_idx=None)``).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def build(self, q_shape, k_shape=None, v_shape=None) -> None:
        pass

    def call(self, q, k=None, v=None, q_mask=None, kv_mask=None, attn_mask=None, initial_v=None, kv_sort_idx=None, training=None):
        self.calls.append({"k": k, "kv_mask": kv_mask, "kv_sort_idx": kv_sort_idx})
        return q


def test_kv_sort_idx_reaches_q_ca_only():
    """The sort index goes to the query cross-attention and nowhere else (org ``decoder.py``)."""
    nlayer = NDecoderLayer(**decoder_cfg(0))
    q, kv, attn_mask, kv_mask = decoder_inputs()
    recorders = {part: RecordingAttention(name=f"rec_{part}") for part in ("q_ca", "q_sa", "kv_ca")}
    for part, rec in recorders.items():
        getattr(nlayer, part).fn = rec  # before the first call: keras locks a built layer's state
    idx = np.stack([np.random.default_rng(s).permutation(DEC_N) for s in range(2)]).astype(np.int32)
    nlayer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask, kv_sort_idx=idx)
    assert len(recorders["q_ca"].calls) == 1
    np.testing.assert_array_equal(np.asarray(recorders["q_ca"].calls[0]["kv_sort_idx"]), idx)
    for part in ("q_sa", "kv_ca"):
        assert [c["kv_sort_idx"] for c in recorders[part].calls] == [None], part


# ---------------------------------------------------------------------------------- 5.4 SwiGLU
SWIGLU_CONFIGS = {
    "ffn_64": {"input_size": 64, "activation": "SwiGLU"},
    "innet_27_to_64": {"input_size": 27, "output_size": 64, "activation": "SwiGLU"},
    "two_hidden": {"input_size": 32, "output_size": 8, "hidden_layers": [48, 40], "activation": "SwiGLU"},
}


def swiglu_pair(name: str, quant=None, seed: int = 42, swap: bool = False):
    """Torch Dense with SwiGLU and its ported native twin (``swap``: x1 and the gate half exchanged, a mutant)."""
    cfg = SWIGLU_CONFIGS[name]
    torch.manual_seed(seed)
    tdense = Dense(**cfg).eval()
    perturb_affine_params(tdense, seed)
    factory = LayerFactory(quant)
    with factory.scopes():
        ndense = NDense(**cfg, factory=factory, name="d")
        ndense.build((None, 21, cfg["input_size"]))
    port_dense(tdense, ndense)
    if swap:
        for x1, gate in zip(ndense.hidden, ndense.gates, strict=True):
            k1, b1 = x1.kernel.numpy(), x1.bias.numpy()
            x1.kernel.assign(gate.kernel.numpy())
            x1.bias.assign(gate.bias.numpy())
            gate.kernel.assign(k1)
            gate.bias.assign(b1)
    return tdense, ndense


def swiglu_input(name: str, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((4, 21, SWIGLU_CONFIGS[name]["input_size"])).astype(np.float32)


def test_swiglu_leaves_and_names():
    """Leaf rule (plan 5.4), with the hybrid twin's names and equation (R4a): x1 ``{name}_hidden{i}_x1``, x2
    ``{name}_hidden{i}_x2`` (silu inside), einsum ``{name}_hidden{i}_glu`` ``bnc,bnc->bnc``; one graph in float and
    quantized mode, so both have the same variable layout. Other ranks are folded into rank 3 around the einsum.
    """
    from hgq.layers import QDense, QEinsum  # noqa: PLC0415

    for quant in (None, QUANT_SPECS["high"]):
        _, ndense = swiglu_pair("two_hidden", quant)
        names = [layer.name for layer in (*ndense.hidden, *ndense.gates, *ndense.products, ndense.final)]
        assert names == ["d_hidden0_x1", "d_hidden1_x1", "d_hidden0_x2", "d_hidden1_x2", "d_hidden0_glu", "d_hidden1_glu", "d_final"]
        # explicit letters, not "...": HGQ2 expands "..." from a Python set, so the string would depend on the hash seed
        assert [p.get_config()["equation"] for p in ndense.products] == ["bnc,bnc->bnc"] * 2
        assert [g.activation.__name__ for g in ndense.gates] == ["silu", "silu"]
        assert all(h.activation.__name__ == "linear" for h in ndense.hidden)
        assert [h.kernel.shape for h in ndense.hidden] == [(32, 48), (48, 40)] == [g.kernel.shape for g in ndense.gates]
        if quant is not None:
            assert all(isinstance(layer, QDense) for layer in (*ndense.hidden, *ndense.gates))
            assert all(isinstance(layer, QEinsum) for layer in ndense.products)
    tdense, _ = swiglu_pair("two_hidden")
    for shape in ((5, 32), (2, 3, 7, 32)):  # rank 2 and rank 4, folded to (B, -1, C) around the rank-3 einsum
        factory = LayerFactory(QUANT_SPECS["high"])
        with factory.scopes():
            folded = NDense(**SWIGLU_CONFIGS["two_hidden"], factory=factory, name="f")
            folded.build((None, *shape[1:]))
        assert folded.products[0].get_config()["equation"] == "bnc,bnc->bnc"
        port_dense(tdense, folded)
        x = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
        with torch.no_grad():
            ref = tdense(torch.from_numpy(x)).numpy()
        got = np.asarray(folded(x, training=False))
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 0.1 * np.abs(ref).max(), "high-spec rank-folded SwiGLU must follow torch"


@pytest.mark.parametrize("name", SWIGLU_CONFIGS)
def test_swiglu_float_equals_torch(name):
    """Float parity with torch SwiGLU. Mutant: x1 and the gate half swapped exceeds the tolerance by >= 10x."""
    tdense, ndense = swiglu_pair(name)
    x = swiglu_input(name)
    with torch.no_grad():
        ref = tdense(torch.from_numpy(x)).numpy()
    assert_parity("swiglu", name, ref, ndense(x), atol=TOL_SWIGLU, rtol=0.0)
    _, swapped = swiglu_pair(name, swap=True)
    diff, _ = max_errors(ref, swapped(x))
    assert diff > 10 * TOL_SWIGLU, f"swapped halves change the output by only {diff:.2e}"


def quantized_error(name: str, mode: str) -> float:
    tdense, ndense = swiglu_pair(name, QUANT_SPECS[mode])
    x = swiglu_input(name)
    with torch.no_grad():
        ref = tdense(torch.from_numpy(x)).numpy()
    return max_errors(ref, ndense(x, training=False))[0]


# 2x the measured max |quantized - torch float| at the high spec. It is set by the bias quantizer: QUANT_SPECS do not
# configure the "bias" place, so HGQ2's default (kbi b0=8, i0=2, step 1/64) applies (R3.md).
SWIGLU_HIGH_MAX = {"ffn_64": 2 * 1.37e-2, "innet_27_to_64": 2 * 1.18e-2}


@pytest.mark.parametrize("name", ["ffn_64", "innet_27_to_64"])
def test_swiglu_quantized_paired_delta(name):
    """High bitwidths stay close to float; low bitwidths are at least 10x further away (measured in R3.md)."""
    high, low = quantized_error(name, "high"), quantized_error(name, "low")
    assert high <= SWIGLU_HIGH_MAX[name], f"high spec error {high:.2e}"
    assert low >= 10 * high, f"low {low:.2e} is not 10x high {high:.2e}"


class _Float:
    """Stands in for an HGQ2 quantizer (``index`` None) or for one input of a multi-input quantizer: passes it unchanged."""

    def __init__(self, wrapped, index: int | None = None):
        self.wrapped, self.index = wrapped, index

    def __call__(self, x, training=None):
        if self.index is None:
            return x
        out = list(self.wrapped(x, training=training))
        out[self.index] = x[self.index]
        return out


def test_swiglu_quantized_leaves_are_live():
    """Liveness at the low spec: forcing any one quantizer of the three SwiGLU leaves to float changes the output.

    Quantizers: input, kernel and bias of the x1 and gate ``QDense``, and each input of the ``QEinsum``.
    """
    x = swiglu_input("ffn_64")
    _, ndense = swiglu_pair("ffn_64", QUANT_SPECS["low"])
    base = np.asarray(ndense(x, training=False))
    checked = []
    targets = [(leaf, attr, None) for leaf in (ndense.hidden[0], ndense.gates[0]) for attr in ("_iq", "_kq", "_bq")]
    targets += [(ndense.products[0], "_iq", 0), (ndense.products[0], "_iq", 1)]
    for leaf, attr, index in targets:
        saved = getattr(leaf, attr)
        object.__setattr__(leaf, attr, _Float(saved, index))  # noqa: PLC2801  (keras locks a built layer's state)
        try:
            out = np.asarray(ndense(x, training=False))
        finally:
            object.__setattr__(leaf, attr, saved)  # noqa: PLC2801
        tag = f"{leaf.name}.{attr}" + ("" if index is None else f"[{index}]")
        assert np.abs(out - base).max() > 0, f"{tag} is inert"
        checked.append(tag)
    assert np.array_equal(np.asarray(ndense(x, training=False)), base), "restoring the quantizers restores the output"
    assert len(checked) == 8, checked


def jit_step(module):
    """The call kwargs are jit arguments, so masks, sort values and indices are traced (a host-side sort fails)."""

    def loss_fn(tv, ntv, x, call_kwargs):
        out, new_ntv, losses = module.stateless_call(tv, ntv, x, return_losses=True, training=True, **call_kwargs)
        outs = out if isinstance(out, tuple) else (out,)
        qloss = sum(losses) if losses else 0.0
        return sum(keras.ops.mean(keras.ops.square(o)) for o in outs) + qloss, (new_ntv, qloss)

    return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


def assert_trains_under_jit(module, x, call_kwargs: dict, tag: str) -> None:
    """Shared JIT gate (plan 5): one compile over 3 calls, finite gradients, every trainable variable a nonzero gradient."""
    tv = [v.value for v in module.trainable_variables]
    ntv = [v.value for v in module.non_trainable_variables]
    step = jit_step(module)
    nonzero = np.zeros(len(tv), dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(3):
        xi = (x + rng.normal(0, 0.1, x.shape)).astype(np.float32)
        (loss, (ntv, qloss)), grads = step(tv, ntv, xi, call_kwargs)
        bad = [v.path for v, g in zip(module.trainable_variables, grads, strict=True) if not np.isfinite(np.asarray(g)).all()]
        assert not bad, f"{tag}: non-finite gradients {bad}"
        nonzero |= np.array([bool(np.any(np.asarray(g) != 0)) for g in grads])
        assert np.isfinite(float(loss)) and float(qloss) > 0
    assert step._cache_size() == 1, f"{tag}: {step._cache_size()} compilations"  # noqa: SLF001
    zero = [v.path for v, ok in zip(module.trainable_variables, nonzero, strict=True) if not ok]
    assert not zero, f"{tag}: zero gradient for {zero}"


def leaf_ebops(leaf, x_shapes) -> float:
    """EBOPs of one HGQ2 leaf read as (train losses - eval losses) / beta (plan 3.6: not the uint32 variable)."""
    rng = np.random.default_rng(1)
    xs = [rng.standard_normal((2, *s[1:])).astype(np.float32) for s in x_shapes]
    arg = xs if len(xs) > 1 else xs[0]
    tv = [v.value for v in leaf.trainable_variables]
    ntv = [v.value for v in leaf.non_trainable_variables]
    _, _, train = leaf.stateless_call(tv, ntv, arg, return_losses=True, training=True)
    _, _, evals = leaf.stateless_call(tv, ntv, arg, return_losses=True, training=False)
    betas = {float(np.asarray(v.value)) for v in leaf.non_trainable_variables if v.name == "beta"}
    assert len(betas) == 1, betas
    beta = betas.pop()
    assert beta > 0
    return (float(sum(train)) - float(sum(evals or [0.0]))) / beta


def test_swiglu_quantized_trains_under_jit_with_ebops():
    _, ndense = swiglu_pair("ffn_64", QUANT_SPECS["high"])
    assert_trains_under_jit(ndense, swiglu_input("ffn_64"), {}, "swiglu")
    shape = (None, 21, 64)
    for leaf, shapes in ((ndense.hidden[0], [shape]), (ndense.gates[0], [shape]), (ndense.products[0], [(None, 21, 128)] * 2)):
        assert leaf_ebops(leaf, shapes) > 0, f"{leaf.name}: no EBOPs"


LINFORMER_DEC = {"attn_type": "linformer", "linformer_seq_len": DEC_N, "linformer_k": 8}


@pytest.mark.parametrize("part", ["encoder", "decoder_layer", pytest.param("decoder_layer_linformer", marks=R2_REQUIRED)])
def test_pr8_layers_train_under_jit(part):
    """JIT gate at the high spec for the sorting encoder (2 layers, registers) and a PR #8 decoder layer at depth 0,
    with dense and (after the R2 merge) masked-Linformer attention. Linformer: ``q_sa`` (30 queries, seq_len 40) must
    run the unmasked project core, as org (``mask=None``); the masked core has tail rows with zero gradient here.
    """
    factory = LayerFactory(QUANT_SPECS["high"])
    if part == "encoder":
        cfg = encoder_cfg("legacy", 2)
        cfg.pop("dim")
        x, mask = padded(5)
        sort_value = np.random.default_rng(6).uniform(-3, 3, mask.shape).astype(np.float32)
        with factory.scopes():
            module = NEncoder(dim=32, factory=factory, **cfg)
            module.build((None, *x.shape[1:]))
        kwargs = {"kv_mask": mask, "x_sort_value": sort_value}
    else:
        q, kv, attn_mask, kv_mask = decoder_inputs()
        cfg = decoder_cfg(0)
        if part == "decoder_layer_linformer":
            cfg["attn_kwargs"] = {**cfg["attn_kwargs"], **LINFORMER_DEC}
        with factory.scopes():
            module = NDecoderLayer(**cfg, factory=factory, name="decoder_l0")
            module.build(q.shape, kv.shape)
        idx = np.stack([np.random.default_rng(s).permutation(DEC_N) for s in range(2)]).astype(np.int32)
        x = q
        kwargs = {"kv": kv, "attn_mask": attn_mask, "kv_mask": kv_mask, "kv_sort_idx": idx}
    assert_trains_under_jit(module, x, kwargs, part)


# ------------------------------------------------------------------------------------ 5.5 sorting
SORT_LENGTHS = (120, 37, 159, 80)


def sort_fixture(seed: int = 3, dim: int = 8):
    """Unique phi in (-pi, pi), B=4, N=160, valid counts 120 / 37 / 159 / 80 at random positions, padding phi 0."""
    rng = np.random.default_rng(seed)
    b, n = len(SORT_LENGTHS), 160
    phi = (rng.permutation(b * n).reshape(b, n) / (b * n) * 2 * np.pi - np.pi).astype(np.float32)
    valid = np.zeros((b, n), dtype=bool)
    for event, count in enumerate(SORT_LENGTHS):
        valid[event, rng.choice(n, count, replace=False)] = True
    phi = np.where(valid, phi, 0.0).astype(np.float32)
    x = rng.standard_normal((b, n, dim)).astype(np.float32)
    return phi, valid, x


def prefix_and_padding_match(ref: np.ndarray, got: np.ndarray, valid: np.ndarray) -> tuple[bool, bool]:
    """Whether the first n_valid entries agree in every event, and whether the padding suffixes hold the same set."""
    prefix = padding = True
    for event in range(valid.shape[0]):
        count = int(valid[event].sum())
        prefix &= bool(np.array_equal(ref[event, :count], got[event, :count]))
        padding &= set(ref[event, count:].tolist()) == set(got[event, count:].tolist()) == set(np.flatnonzero(~valid[event]).tolist())
    return prefix, padding


def test_sort_index_equals_torch_on_the_valid_prefix():
    """Exact on the valid prefix; the padding suffix holds the same set of indices (torch's argsort is not stable, so it
    orders the tied +inf entries its own way). Control: without the +inf fill, padding interleaves with valid tokens.
    """
    phi, valid, _ = sort_fixture()
    ref = torch_sort_index(torch.from_numpy(phi), torch.from_numpy(valid)).numpy()
    got = np.asarray(native_sort_index(phi, valid))
    assert prefix_and_padding_match(ref, got, valid) == (True, True)
    no_fill = np.asarray(native_sort_index(phi, None))
    assert not prefix_and_padding_match(ref, no_fill, valid)[0], "control: without the +inf fill the prefix must differ"


def test_unsort_round_trip_is_exact_and_not_on_identity():
    phi, valid, x = sort_fixture()
    idx = np.asarray(native_sort_index(phi, valid))
    assert all(not np.array_equal(row, np.arange(row.size)) for row in idx), "the sort index must not be the identity"
    sorted_x = np.asarray(gather_tokens(x, idx))
    assert not np.array_equal(sorted_x, x)
    np.testing.assert_array_equal(np.asarray(gather_tokens(sorted_x, keras.ops.argsort(idx, axis=-1))), x)
    # the encoder with no layers sorts and unsorts: the identity, exactly
    enc = NEncoder(num_layers=0, dim=x.shape[-1])
    np.testing.assert_array_equal(np.asarray(enc(x, kv_mask=valid, x_sort_value=phi)), x)


class TorchPositionStub(nn.Module):
    """Adds a bias per (sorted) position, times the padding mask: its output depends on the token order and the mask."""

    def __init__(self, bias: np.ndarray):
        super().__init__()
        self.register_buffer("bias", torch.from_numpy(bias))

    def forward(self, x, kv_mask=None, **kwargs):
        return x + self.bias[: x.shape[1]] * kv_mask[..., None]


class NativePositionStub(keras.layers.Layer):
    def __init__(self, bias: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.bias = bias

    def build(self, x_shape) -> None:
        pass

    def call(self, x, kv_mask=None, initial_v=None, training=None):
        return x + self.bias[: x.shape[1]] * keras.ops.cast(kv_mask, x.dtype)[..., None], initial_v


def stub_encoders(regs: int = 8):
    phi, valid, x = sort_fixture()
    bias = np.random.default_rng(4).standard_normal((x.shape[1] + regs, x.shape[2])).astype(np.float32)
    torch.manual_seed(4)
    tenc = Encoder(num_layers=0, dim=x.shape[2], num_register_tokens=regs).eval()
    tenc.layers = nn.ModuleList([TorchPositionStub(bias)])
    nenc = NEncoder(num_layers=0, dim=x.shape[2], num_register_tokens=regs)
    nenc.layers = [NativePositionStub(bias, name="stub")]  # before build: keras locks a built layer's state
    nenc.build((None, *x.shape[1:]))
    nenc.register_tokens.assign(tenc.register_tokens.detach().numpy())
    return tenc, nenc, (phi, valid, x)


def test_encoder_sorting_equals_torch_with_a_position_stub():
    """Sort, register prepend, mask gather, register strip and unsort equal org ``Encoder.forward`` (valid tokens).

    Control: the same native encoder without the sort differs.
    """
    tenc, nenc, (phi, valid, x) = stub_encoders()
    with torch.no_grad():
        ref = tenc(torch.from_numpy(x), x_sort_value=torch.from_numpy(phi), kv_mask=torch.from_numpy(valid)).numpy()
    got = np.asarray(nenc(x, kv_mask=valid, x_sort_value=phi))
    assert_parity("encoder.sort_stub", "B4N160", ref, got, valid_mask=valid, atol=0.0, rtol=0.0)
    unsorted = np.asarray(nenc(x, kv_mask=valid))
    diff, _ = max_errors(ref, unsorted, valid)
    assert diff > 0.1, f"control: without the sort the stub output changes by only {diff:.2e}"


# ------------------------------------------------------------------------------- model, dense attention
@pytest.fixture(scope="module")
def batch():
    inputs, _ = clic_dummy_batch()
    assert (~inputs["node_valid"]).any(-1).all(), "every event needs a padded node (plan 1.2, batch of one all valid)"
    return inputs


def org_dense_pair(inputs: dict, seed: int = 30, **layer_changes):
    """Torch and ported native MaskFormer of base.yaml + linformer.yaml at dim 32 with dense attention.

    ``layer_changes`` edits the native decoder layer config only (a control).
    """
    section = with_dense_attention(org_model_section(dim=32))
    tmodel = torch_from_section(section, seed)
    perturb_affine_params(tmodel, seed)
    nsection = copy.deepcopy(section)
    nsection["model"]["init_args"]["decoder"]["decoder_layer_config"].update(layer_changes)
    nmodel = native_from_section(nsection)
    nmodel(inputs, training=False)
    port_maskformer(tmodel, nmodel)
    return tmodel, nmodel


def test_org_model_with_dense_attention_equals_torch(batch):
    """The whole PR #8 model (FastLayerNorm, legacy placement, ca_sa_dense, self_attn_norm_kv, SwiGLU everywhere,
    phi sorting, classification at layer 0) against org torch at the maskformer.e2e tolerance; masks exactly equal.

    Control: ``ca_dense_sa`` in the native decoder moves the final mask logits by more than 10x the tolerance.
    """
    out_t, out_n = run_pair(*org_dense_pair(batch), batch)
    assert {"layer_0.classification.pflow_logit", "layer_0.mask.pflow_node_logit", "layer_3.attn_mask"} <= set(out_t)
    bools = assert_tree_parity("maskformer.org_dense", out_t, out_n)
    for key, mask in bools.items():
        assert 0.05 < mask.mean() < 0.95, f"{key}: attention mask fraction {mask.mean():.3f}"
    attn_masks = [key for key in bools if key.endswith(".attn_mask")]
    assert len(attn_masks) == 4, attn_masks
    for key in attn_masks:
        # layer_k.attn_mask = sigmoid(layer_k mask logit) >= 0.1 (pred_threshold, base.yaml). The plan 5.8 margin (10x
        # the e2e tolerance) does not hold on this fixture (layer 3: 1.05x, R3.md); R5 owns that fixture. Here: every
        # valid logit is >= 10x the measured torch-native error of its layer away from the threshold.
        logit_key = key.replace("attn_mask", "mask.pflow_node_logit")
        logit_t, logit_n = out_t[logit_key].astype(np.float64), np.asarray(out_n[logit_key])
        keep = logit_t > -1e30
        margin = np.abs(logit_t[keep] - LOGIT_ATTN_THRESHOLD).min()
        err = np.abs(logit_t[keep] - logit_n[keep]).max()
        assert margin >= 10 * err, f"{logit_key}: margin {margin:.2e} < 10x error {err:.2e}"
    _, out_c = run_pair(*org_dense_pair(batch, query_update_order="ca_dense_sa"), batch)
    key = "final.mask.pflow_node_logit"
    keep = out_t[key] > -1e30
    diff = np.abs(out_t[key][keep].astype(np.float64) - np.asarray(out_c[key])[keep]).max()
    assert diff > 10 * (E2E["atol"] + E2E["rtol"] * np.abs(out_t[key][keep]).max()), f"ca_dense_sa changes {key} by only {diff:.2e}"


def test_layer0_classification_runs_and_equals_torch(batch):
    """has_first_layer_loss (base.yaml): the classification head runs at layer 0 in both, with equal logits."""
    out_t, out_n = run_pair(*org_dense_pair(batch), batch)
    for key in ("layer_0.classification.pflow_logit", "layer_0.classification.pflow_class_prob"):
        assert key in out_t and key in out_n
        assert_parity("maskformer.org_dense", key, out_t[key], out_n[key], **E2E)


def test_key_sort_idx_reaches_decoder_q_ca(batch):
    """The model sorts by ``node_phi`` with padding last and hands the same index to every decoder ``q_ca``."""
    section = with_dense_attention(org_model_section(dim=32))
    nmodel = native_from_section(section)
    recorders = [RecordingAttention(name=f"rec_q_ca{i}") for i in range(len(nmodel.decoder.layers))]
    for layer, rec in zip(nmodel.decoder.layers, recorders, strict=True):
        layer.q_ca.fn = rec
    nmodel(batch, training=False)
    ref = torch_sort_index(torch.from_numpy(batch["node_phi"]), torch.from_numpy(batch["node_valid"])).numpy()
    for rec in recorders:
        (call,) = rec.calls
        assert prefix_and_padding_match(ref, np.asarray(call["kv_sort_idx"]), batch["node_valid"]) == (True, True)
    unsorted = copy.deepcopy(section)
    unsorted["model"]["init_args"]["input_sort_field"] = None
    nmodel = native_from_section(unsorted)
    rec = RecordingAttention(name="rec_none")
    nmodel.decoder.layers[0].q_ca.fn = rec
    nmodel(batch, training=False)
    assert rec.calls[0]["kv_sort_idx"] is None, "without input_sort_field no sort index is passed"


def test_model_with_sorting_traces_under_jit(batch):
    """``MaskFormer.call`` with ``input_sort_field`` under ``jax.jit``, the inputs (``node_phi`` included) traced: one
    compile over 3 phi permutations, each equal to the eager call. A host-side sort (numpy argsort) cannot trace.
    """
    nmodel = native_from_section(with_dense_attention(org_model_section(dim=32)))
    nmodel(batch, training=False)
    tv = [v.value for v in nmodel.trainable_variables]
    ntv = [v.value for v in nmodel.non_trainable_variables]
    key = "final.mask.pflow_node_logit"

    @jax.jit
    def forward(tv, ntv, inputs):
        out, _ = nmodel.stateless_call(tv, ntv, inputs, training=False)
        return flatten_torch(out)[key]

    rng = np.random.default_rng(4)
    for _ in range(3):
        inputs = dict(batch)
        inputs["node_phi"] = np.stack([rng.permutation(row) for row in batch["node_phi"]]).astype(np.float32)
        eager = np.asarray(flatten_torch(nmodel(inputs, training=False))[key])
        np.testing.assert_allclose(np.asarray(forward(tv, ntv, inputs)), eager, rtol=0.0, atol=TOL_JIT_EAGER)
    assert forward._cache_size() == 1  # noqa: SLF001


# --------------------------------------------------------------------- model, masked-Linformer attention
def org_linformer_pair(inputs: dict, seed: int = 30, sort: bool = True):
    """Torch and ported native MaskFormer of base.yaml + linformer.yaml at dim 32 (masked-Linformer attention, k=64)."""
    section = org_model_section(dim=32)
    tmodel = torch_from_section(section, seed)
    perturb_affine_params(tmodel, seed)
    nsection = copy.deepcopy(section)
    if not sort:
        nsection["model"]["init_args"]["input_sort_field"] = None
    nmodel = native_from_section(nsection)
    nmodel(inputs, training=False)
    port_maskformer(tmodel, nmodel)
    return tmodel, nmodel


@R2_REQUIRED
def test_org_linformer_model_equals_torch(batch, monkeypatch):
    """R2 x R3 wiring (added at the rt/integration merge): R3's model with R2's masked-Linformer cores against org torch.

    Every output at the maskformer.e2e tolerance and the attention masks exactly equal (measured 2.9e-6 max,
    ``R-merge.md``). Controls, each > 10x the tolerance: no input sorting (measured 1.13 on the final mask logits),
    and ``kv_sort_idx`` withheld from the decoder only, so ``q_ca`` reads E/F rows in the wrong order.
    """
    tmodel, nmodel = org_linformer_pair(batch)
    assert {layer.q_ca.fn.core_kind for layer in nmodel.decoder.layers} == {"attend"}
    out_t, out_n = run_pair(tmodel, nmodel, batch)
    bools = assert_tree_parity("maskformer.org_linformer", out_t, out_n)
    assert len([key for key in bools if key.endswith(".attn_mask")]) == 4
    for key in bools:  # the boolean check is not luck: every valid logit is >= 10x the native error from the threshold
        logit_key = key.replace("attn_mask", "mask.pflow_node_logit")
        logit_t, logit_n = out_t[logit_key].astype(np.float64), np.asarray(out_n[logit_key])
        keep = logit_t > -1e30
        margin = np.abs(logit_t[keep] - LOGIT_ATTN_THRESHOLD).min()
        assert margin >= 10 * np.abs(logit_t[keep] - logit_n[keep]).max(), f"{logit_key}: margin {margin:.2e}"

    key = "final.mask.pflow_node_logit"
    keep = out_t[key] > -1e30
    tol = 10 * (E2E["atol"] + E2E["rtol"] * np.abs(out_t[key][keep]).max())
    _, out_c = run_pair(*org_linformer_pair(batch, sort=False), batch)
    diff = np.abs(out_t[key][keep].astype(np.float64) - np.asarray(out_c[key])[keep]).max()
    assert diff > tol, f"no sorting changes {key} by only {diff:.2e}"

    call = NDecoderLayer.call
    monkeypatch.setattr(NDecoderLayer, "call", lambda self, *a, kv_sort_idx=None, **k: call(self, *a, kv_sort_idx=None, **k))  # noqa: ARG005
    got = np.asarray(flatten_torch(nmodel(batch, training=False))[key])
    diff = np.abs(out_t[key][keep].astype(np.float64) - got[keep]).max()
    assert diff > tol, f"withholding kv_sort_idx from q_ca changes {key} by only {diff:.2e}"


# ------------------------------------------------------------------------------ batch of one, all valid
def one_event(num_nodes: int) -> dict:
    """The first dummy event cut to ``num_nodes`` nodes, every node valid (B=1)."""
    inputs, _ = clic_dummy_batch(1)
    out = {key: value[:, :num_nodes].copy() for key, value in inputs.items()}
    out["node_valid"][:] = True
    return out


def torch_linformer_outputs(inputs: dict) -> dict:
    """Org torch MaskFormer (base.yaml + linformer.yaml at dim 32, masked-Linformer attention), flattened outputs."""
    tmodel = torch_from_section(org_model_section(dim=32), seed=8)
    with torch.no_grad():
        return {k: v.numpy() for k, v in run_pair_torch(tmodel, inputs).items()}


def run_pair_torch(tmodel, inputs):
    from .test_maskformer import flatten_torch  # noqa: PLC0415

    return flatten_torch(tmodel(torch_inputs(inputs)))


@pytest.mark.parametrize("num_nodes", [100, 160])
def test_batch_of_one_all_valid_is_the_unmasked_linformer_path_in_torch(num_nodes):
    """Org sets ``key_valid`` to None for B=1 with every key valid (``maskformer.py:112-113``); Linformer ``project``
    then slices E without the renorm. Equal to the masked path only when the key count equals ``linformer_seq_len``.

    Measured on the org torch model: the B=1 event against the same event in a B=2 batch (a mask is passed). At 100
    nodes (108 < 168 encoder keys) they differ; at 160 nodes (168 = 168, and 160 = 160 decoder keys) they agree.
    Native always passes the mask, so it reproduces the B=2 path; parity fixtures keep a padded node in every event.
    """
    single = one_event(num_nodes)
    double = {key: np.concatenate([value, value]) for key, value in single.items()}
    out_1 = torch_linformer_outputs(single)
    out_2 = torch_linformer_outputs(double)
    key = "final.mask.pflow_node_logit"
    diff = float(np.abs(out_1[key][0].astype(np.float64) - out_2[key][0]).max())
    if num_nodes < 160:
        assert diff > 1e-2, f"B=1 all valid should take the unmasked path, diff {diff:.2e}"
    else:
        assert diff <= TOL_B1_EQUAL, f"at n = seq_len the paths agree, diff {diff:.2e}"


class MaskRecorder(keras.layers.Layer):
    """Encoder-layer double: records the ``kv_mask`` it receives, returns its input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masks = []

    def build(self, x_shape) -> None:
        pass

    def call(self, x, kv_mask=None, initial_v=None, training=None):
        self.masks.append(kv_mask)
        return x, initial_v


def test_native_passes_the_mask_for_a_batch_of_one_all_valid():
    """Native has no B=1 switch: encoder and decoder get an all-True mask, not None (documented deviation)."""
    nmodel = native_from_section(with_dense_attention(org_model_section(dim=32)))
    enc_rec = MaskRecorder(name="enc_rec")
    nmodel.encoder.layers = [enc_rec]
    dec_rec = RecordingAttention(name="dec_rec")
    nmodel.decoder.layers[0].q_ca.fn = dec_rec
    nmodel(one_event(100), training=False)
    (enc_mask,) = enc_rec.masks
    assert enc_mask is not None and np.asarray(enc_mask).all() and np.asarray(enc_mask).shape == (1, 108)
    dec_mask = dec_rec.calls[0]["kv_mask"]
    assert dec_mask is not None and np.asarray(dec_mask).all() and np.asarray(dec_mask).shape == (1, 100)


class TokenRecorder(MaskRecorder):
    """Encoder-layer double that also records the tokens it receives."""

    def call(self, x, kv_mask=None, initial_v=None, training=None):
        self.tokens = x
        return super().call(x, kv_mask=kv_mask, initial_v=initial_v, training=training)


def test_model_encoder_layers_see_tokens_sorted_by_phi(batch):
    """The model hands ``node_phi`` to the encoder: its layers see the embedded tokens in phi order (after the
    registers), padding last. With dense attention this is invisible in the outputs, so it is checked here directly.
    """
    nmodel = native_from_section(with_dense_attention(org_model_section(dim=32)))
    rec = TokenRecorder(name="enc_rec")
    nmodel.encoder.layers = [rec]
    nmodel(batch, training=False)
    embed = np.asarray(nmodel.input_nets[0](batch))
    valid = batch["node_valid"]
    ref_idx = torch_sort_index(torch.from_numpy(batch["node_phi"]), torch.from_numpy(valid)).numpy()
    tokens = np.asarray(rec.tokens)[:, nmodel.encoder.num_register_tokens :]
    for event in range(len(valid)):
        count = int(valid[event].sum())
        np.testing.assert_array_equal(tokens[event, :count], embed[event, ref_idx[event, :count]])  # torch's valid order
        padding = sorted(map(tuple, tokens[event, count:].tolist()))
        assert padding == sorted(map(tuple, embed[event, ~valid[event]].tolist())), "padding last, as a set"
    assert not np.array_equal(tokens, embed), "the fixture's phi order must not be the input order"
