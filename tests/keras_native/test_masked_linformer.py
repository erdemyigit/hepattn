"""R2 gates (docs/hgq/RETARGET_ORG_MAIN.md section 5.6): the masked-Linformer core and ``Attention(attn_type="linformer")``.

References, in this process: ``masked_linformer`` dce2e15 (``attend``, ``project``, ``_rescale``) and org/main
``hepattn.models.attention.Attention``. Shapes are the production ones of ``configs/linformer.yaml``: decoder ``q_ca``
(150 queries over 160 constituents, seq_len 160), ``kv_ca`` (160 over 150: the E/F rows 150-159 exist only for the
renorm), encoder self-attention (168 = 160 + 8 registers, seq_len 168), decoder ``q_sa`` (no mask: ``E[:150]``), plus a
short encoder (40 tokens, seq_len 168) where the renorm reads the tail rows.

Tolerances are 2x the value measured on this implementation (macOS arm64 CPU, JAX 0.11.2), written next to each one;
the stage log docs/hgq/port_log/R2.md lists the controls and the mutants that exceed them.
"""

import contextlib
import itertools
import math

import jax
import keras
import masked_linformer as ml
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hgq.layers import QDense, QEinsum, QEinsumDense, QSoftmax, QUnaryFunctionLUT
from hgq.layers.core.base import QLayerBase

from hepattn.keras_native import masked_linformer as nml
from hepattn.keras_native.attention import Attention as NAttention
from hepattn.keras_native.attention import build_attention
from hepattn.keras_native.factory import EinsumOp, LayerFactory, SoftmaxOp
from hepattn.models.attention import Attention

from .golden.cases import QUANT_SPECS
from .parity_utils import assert_parity
from .porting import perturb_affine_params, port_attention

DIM, HEADS, HEAD_DIM, K = 256, 16, 16, 64
B = 2
# src/hepattn/experiments/clic/configs/pflow_hgq.yaml quant block; beta0 = 1 so the EBOPs loss term is EBOPs itself
PROD_SPEC = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0},
}
# The low end of the paired delta. The golden "low" / "mid" specs make this layer return a constant (std over the
# queries 0.0, measured; tests/keras_native/golden/cases.py notes the same for Helen's layer), so they never reach the
# quantized attention graph. This one does (std 0.188).
COARSE_SPEC = {
    "weight": {"default_q_type": "kbi", "b0": 6, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 8, "f0": 4},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 4},
}

SHAPES = {  # (q_len, kv_len, linformer_seq_len, linformer_core)
    "q_ca": (150, 160, 160, "attend"),
    "kv_ca": (160, 150, 160, "attend"),
    "enc": (168, 168, 168, "project"),
    "enc_short": (40, 40, 168, "project"),
    "q_sa": (150, 150, 160, "project_unmasked"),
}
SELF = {"enc", "enc_short", "q_sa"}


def t(x):
    return None if x is None else torch.from_numpy(np.asarray(x))


def n(x) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(x))


# ------------------------------------------------------------------------------------------------- fixtures
def make_inputs(shape: str, seed: int, sort: bool = False) -> dict:
    """Inputs of one call site. Key padding is last (event 0: 7 padded, event 1: 1 padded); q_ca has an all-False row."""
    q_len, kv_len, _, kind = SHAPES[shape]
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, q_len, DIM)).astype(np.float32)
    kv = q if shape in SELF else rng.standard_normal((B, kv_len, DIM)).astype(np.float32)
    kv_mask = attn_mask = idx = None
    if shape in {"q_ca", "enc", "enc_short"}:
        kv_mask = np.ones((B, kv_len), bool)
        kv_mask[0, -7:] = False
        kv_mask[1, -1:] = False
    if kind == "attend":
        attn_mask = rng.random((B, q_len, kv_len)) < 0.4
        attn_mask[0, 3] = False
    if sort:
        idx = np.stack([rng.permutation(kv_len) for _ in range(B)])
        assert (idx != np.arange(kv_len)).any()
    return {"q": q, "kv": kv, "kv_mask": kv_mask, "attn_mask": attn_mask, "kv_sort_idx": idx}


def torch_attention(shape: str, seed: int, **kwargs) -> Attention:
    seq_len = SHAPES[shape][2]
    torch.manual_seed(seed)
    tattn = Attention(DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=seq_len, linformer_k=K, **kwargs).eval()
    perturb_affine_params(tattn, seed)
    return tattn


def native_attention(shape: str, tattn: Attention | None, spec=None, name: str = "attn", **kwargs) -> NAttention:
    """The other call sites take ``linformer_core="auto"``; ``q_sa`` names its core (no masks), as the decoder must."""
    q_len, kv_len, seq_len, kind = SHAPES[shape]
    if kind == "project_unmasked":
        kwargs = {"linformer_core": kind, **kwargs}
    factory = LayerFactory(spec)
    with factory.scopes():
        nattn = NAttention(
            DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=seq_len, linformer_k=K, factory=factory, name=name, **kwargs
        )
        q_shape = (None, q_len, DIM)
        nattn.build(q_shape, q_shape, q_shape) if shape in SELF else nattn.build(q_shape, (None, kv_len, DIM))
    if tattn is not None:
        port_attention(tattn, nattn)
    return nattn


def torch_out(tattn: Attention, x: dict, **kwargs) -> np.ndarray:
    kv = None if x["kv"] is x["q"] else t(x["kv"])
    with torch.no_grad():
        out = tattn(t(x["q"]), kv, kv, kv_mask=t(x["kv_mask"]), attn_mask=t(x["attn_mask"]), kv_sort_idx=t(x["kv_sort_idx"]), **kwargs)
    return out.numpy()


def native_out(nattn: NAttention, x: dict, training: bool = False, **kwargs):
    kv = None if x["kv"] is x["q"] else x["kv"]
    return nattn(x["q"], kv, kv, kv_mask=x["kv_mask"], attn_mask=x["attn_mask"], kv_sort_idx=x["kv_sort_idx"], training=training, **kwargs)


def core_projections(seq_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-(K**-0.5), K**-0.5, (seq_len, K)).astype(np.float32) for _ in range(2))


def qleaves(module) -> dict:
    """HGQ2 leaves by name (QSoftmax's own exp/inv tables are part of the QSoftmax leaf)."""
    found = module._flatten_layers(include_self=False, recursive=True)  # noqa: SLF001
    return {layer.name: layer for layer in found if isinstance(layer, QLayerBase) and not layer.name.endswith(("_exp_table", "_inv_table"))}


@contextlib.contextmanager
def coarsened(leaf, frac_bits: float = 1.0):
    """Every quantizer of ``leaf`` at <= ``frac_bits`` fractional bits (KBI: b <= i + frac_bits); restored on exit.

    Yields:
        The number of bit variables changed.
    """
    saved = [(v, n(v.value)) for v in leaf.variables if v.name in {"f", "b"}]
    ints = {v.path.rsplit("/", 1)[0]: n(v.value) for v in leaf.variables if v.name == "i"}
    try:
        for var, value in saved:
            cap = frac_bits if var.name == "f" else ints[var.path.rsplit("/", 1)[0]] + frac_bits
            var.assign(np.minimum(value, cap))
        yield len(saved)
    finally:
        for var, value in saved:
            var.assign(value)


# ------------------------------------------------------------------------------------------- float: core
# measured (max over q_ca, kv_ca): forward 8.345e-7; gradients q 8.345e-7, k 3.278e-7, v 3.576e-7, E 4.244e-5, F 5.913e-5
# (|grad E|, |grad F| up to about 28)
ATTEND_FWD_TOL = 1.67e-6
ATTEND_GRAD_TOL = {"q": 1.67e-6, "k": 6.56e-7, "v": 7.15e-7, "E": 8.49e-5, "F": 1.18e-4}


@pytest.mark.parametrize("shape", ["q_ca", "kv_ca"])
def test_attend_core_forward_and_gradients_equal_masked_linformer(shape):
    """Forward and the gradients for q, k, v, E, F; padded k/v rows hold NaN; an all-False query row gives exactly 0."""
    q_len, kv_len, seq_len, _ = SHAPES[shape]
    rng = np.random.default_rng(1)
    q = rng.standard_normal((B, HEADS, q_len, HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32)
    kv_mask = np.ones((B, kv_len), bool)
    kv_mask[0, -7:] = False
    kv_mask[1, -1:] = False
    attn_mask = (rng.random((B, q_len, kv_len)) < 0.5) & kv_mask[:, None]
    attn_mask[0, 1] = False
    k = np.where(kv_mask[:, None, :, None], k, np.nan).astype(np.float32)
    v = np.where(kv_mask[:, None, :, None], v, np.nan).astype(np.float32)
    pk, pv = core_projections(seq_len, 2)

    core = nml.AttendCore(LayerFactory(), HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, name="att")
    core.build()
    core.E.assign_full(pk)
    core.F.assign_full(pv)
    assert core.E.has_tail == (seq_len > kv_len)

    tq, tk, tv, tpk, tpv = (torch.tensor(a, requires_grad=True) for a in (q, k, v, pk, pv))
    ref = ml.attend(tq, tk, tv, tpk, tpv, t(attn_mask))
    cot = rng.standard_normal(ref.shape).astype(np.float32)
    (ref * t(cot)).sum().backward()

    out = n(core(q, k, v, attn_mask))
    assert np.isfinite(out).all()
    assert_parity("masked_linformer.attend", shape, ref.detach().numpy(), out, atol=ATTEND_FWD_TOL, rtol=0.0)
    all_false = ~attn_mask.any(-1)  # (B, Q)
    assert all_false.any()
    assert np.abs(out.transpose(0, 2, 1, 3)[all_false]).max() == 0.0

    paths = [var.path for var in core.trainable_variables]

    def loss(tvs, q, k, v):
        out, _ = core.stateless_call(tvs, [], q, k, v, attn_mask)
        return (out * cot).sum()

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))([var.value for var in core.trainable_variables], q, k, v)
    by_path = dict(zip(paths, grads[0], strict=True))

    def full(proj: str) -> np.ndarray:
        rows = [n(by_path[f"att/att_{proj}/att_{proj}_proj/kernel"])]
        if f"att/att_{proj}/tail" in by_path:
            rows.append(n(by_path[f"att/att_{proj}/tail"]))
        return np.concatenate(rows, 0)

    got = {"q": n(grads[1]), "k": n(grads[2]), "v": n(grads[3]), "E": full("E"), "F": full("F")}
    want = {"q": tq.grad.numpy(), "k": tk.grad.numpy(), "v": tv.grad.numpy(), "E": tpk.grad.numpy(), "F": tpv.grad.numpy()}
    padded = np.broadcast_to(~kv_mask[:, None, :, None], got["k"].shape)
    for name in ("k", "v"):
        assert np.abs(got[name][padded]).max() == 0.0, f"gradient into padded {name}"
        assert np.abs(want[name][padded]).max() == 0.0
        got[name], want[name] = got[name][~padded], want[name][~padded]
    for name, tol in ATTEND_GRAD_TOL.items():
        assert got[name].shape == want[name].shape, name
        assert_parity("masked_linformer.attend_grad", f"{shape}.{name}", want[name], got[name], atol=tol, rtol=0.0)


# measured: enc 1.013e-6, enc_short (tail rows in the renorm) 5.960e-7, q_sa (no mask) 2.980e-7
PROJECT_TOL = {"enc": 2.03e-6, "enc_short": 1.19e-6, "q_sa": 5.96e-7}


@pytest.mark.parametrize("shape", ["enc", "enc_short", "q_sa"])
def test_project_core_equals_project_then_unmasked_attention(shape):
    q_len, kv_len, seq_len, kind = SHAPES[shape]
    masked = kind == "project"
    rng = np.random.default_rng(3)
    q, k, v = (rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32) for _ in range(3))
    kv_mask = None
    if masked:
        kv_mask = np.ones((B, kv_len), bool)
        kv_mask[0, -9:] = False
        kv_mask[1, -1:] = False
        k = np.where(kv_mask[:, None, :, None], k, np.nan).astype(np.float32)
        v = np.where(kv_mask[:, None, :, None], v, np.nan).astype(np.float32)
    pk, pv = core_projections(seq_len, 4)
    core = nml.ProjectCore(LayerFactory(), HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, masked=masked, name="prj")
    core.build()
    core.E.assign_full(pk)
    core.F.assign_full(pv)
    assert core.E.has_tail == (masked and seq_len > kv_len)
    out = n(core(q, k, v, kv_mask))
    tm = t(kv_mask)
    ref = F.scaled_dot_product_attention(t(q), ml.project(t(k), t(pk), tm), ml.project(t(v), t(pv), tm)).numpy()
    assert np.isfinite(out).all()
    assert_parity("masked_linformer.project", shape, ref, out, atol=PROJECT_TOL[shape], rtol=0.0)


# ------------------------------------------------------------------------------ float: Attention vs org/main
# measured max |native - org|: q_ca 6.258e-7, kv_ca 5.960e-7, enc 6.258e-7, enc_short 5.364e-7, q_sa 5.662e-7;
# q_ca sorted 5.364e-7; batch of one 4.321e-7
ATTN_TOL = 1.25e-6
SORT_SHAPES = {"q_ca": True}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_linformer_attention_equals_org(shape):
    """qkv_norm, perturbed biases and norm affines, kv_mask / attn_mask / kv_sort_idx as at the org call site."""
    tattn = torch_attention(shape, 5, qkv_norm=True, norm="LayerNorm")
    nattn = native_attention(shape, tattn, qkv_norm=True, norm="LayerNorm")
    assert nattn.core_kind == SHAPES[shape][3]
    x = make_inputs(shape, 6, sort=SORT_SHAPES.get(shape, False))
    assert_parity("attention.linformer", shape, torch_out(tattn, x), native_out(nattn, x), atol=ATTN_TOL, rtol=0.0)


def test_kv_sort_idx_gathers_keys_masks_and_attn_mask_columns():
    """Control for the q_ca row above: the same call without the gather is O(1) off, so the gather is what matches org."""
    tattn = torch_attention("q_ca", 7)
    nattn = native_attention("q_ca", tattn)
    x = make_inputs("q_ca", 8, sort=True)
    ref = torch_out(tattn, x)
    assert_parity("attention.linformer", "q_ca.sorted", ref, native_out(nattn, x), atol=ATTN_TOL, rtol=0.0)
    unsorted = n(native_out(nattn, {**x, "kv_sort_idx": None}))
    assert np.abs(unsorted - ref).max() > 1e3 * ATTN_TOL  # measured 0.333
    # gathering k, v but not the masks is also wrong (the masks index the same keys)
    only_kv = {**x, "kv_mask": x["kv_mask"][np.arange(B)[:, None], np.argsort(x["kv_sort_idx"])]}
    assert np.abs(n(native_out(nattn, only_kv)) - ref).max() > 1e3 * ATTN_TOL  # measured 0.0855


# measured: layer 0 4.619e-7, layer 1 4.470e-7
VR_TOL = 9.24e-7


def test_linformer_value_residual_two_layers_equal_org():
    """Encoder layers 0 and 1: layer 1 mixes layer 0's values before the sequence projection, as org."""
    t0 = torch_attention("enc", 9, value_residual=True, is_first_layer=True)
    t1 = torch_attention("enc", 10, value_residual=True, is_first_layer=False)
    n0 = native_attention("enc", t0, name="a0", value_residual=True, is_first_layer=True)
    n1 = native_attention("enc", t1, name="a1", value_residual=True, is_first_layer=False)
    x = make_inputs("enc", 11)
    iv: dict = {}
    with torch.no_grad():
        h_t = t0(t(x["q"]), kv_mask=t(x["kv_mask"]), initial_values=iv)
        out_t = t1(h_t, kv_mask=t(x["kv_mask"]), initial_values=iv)
    h_n, cache = n0(x["q"], kv_mask=x["kv_mask"])
    out_n, passed = n1(h_n, kv_mask=x["kv_mask"], initial_v=cache)
    assert passed is cache
    assert_parity("attention.linformer_value_residual", "layer0", h_t, h_n, atol=VR_TOL, rtol=0.0)
    assert_parity("attention.linformer_value_residual", "layer1", out_t, out_n, atol=VR_TOL, rtol=0.0)
    no_mix, _ = n1(h_n, kv_mask=x["kv_mask"], initial_v=None)
    assert np.abs(n(no_mix) - n(out_n)).max() > 1e3 * VR_TOL  # control: the mix is not a no-op (measured 0.078)


def test_q_sa_rows_past_the_key_length_are_dead_in_org_and_absent_in_native():
    """Org ``project(t, E, None)`` reads ``E[:n]``: random changes to rows 150-159 change nothing (exactly), to row 149 they do.

    The perturbation is random per element: adding one constant to a whole row of E shifts every score of a query by
    the same amount, which the softmax cancels, so such a probe could not see a row that is read.
    """
    tattn = torch_attention("q_sa", 12)
    x = make_inputs("q_sa", 13)
    ref = torch_out(tattn, x)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        tattn.proj_k[150:] += torch.randn(tattn.proj_k[150:].shape, generator=gen)
    np.testing.assert_array_equal(torch_out(tattn, x), ref)
    with torch.no_grad():
        tattn.proj_k[149] += torch.randn(K, generator=gen)
    assert np.abs(torch_out(tattn, x) - ref).max() > 1e-2  # positive control
    nattn = native_attention("q_sa", None)
    assert tuple(nattn.core.E.kernel.shape) == (150, K) and nattn.core.E.tail is None
    assert not [v.path for v in nattn.trainable_variables if "tail" in v.path]


# measured: |mask of ones - no mask| = 0.120 at n = 40 < seq_len = 168; 1.043e-7 at n = seq_len = 168
ALL_VALID_EQUAL_TOL = 2.09e-7


@pytest.mark.parametrize("shape", ["enc_short", "enc"])
def test_batch_of_one_all_valid_mask_differs_from_no_mask_below_seq_len(shape):
    """Plan 1.2: org MaskFormer passes kv_mask=None for B=1 with every key valid, and a mask of ones is not the same
    as no mask when n < seq_len (the renorm turns on). Both native calls equal org's; at n = seq_len they agree.
    """
    tattn = torch_attention(shape, 14)
    nattn = native_attention(shape, tattn)
    x = make_inputs(shape, 15)
    x = {**x, "q": x["q"][:1], "kv": x["q"][:1]}
    ones = {**x, "kv_mask": np.ones((1, SHAPES[shape][1]), bool)}
    none = {**x, "kv_mask": None}
    got_ones, got_none = n(native_out(nattn, ones)), n(native_out(nattn, none))
    assert_parity("attention.linformer_b1", f"{shape}.ones", torch_out(tattn, ones), got_ones, atol=ATTN_TOL, rtol=0.0)
    assert_parity("attention.linformer_b1", f"{shape}.none", torch_out(tattn, none), got_none, atol=ATTN_TOL, rtol=0.0)
    gap = np.abs(got_ones - got_none).max()
    if SHAPES[shape][1] < SHAPES[shape][2]:
        assert gap > 1e-2
    else:
        assert gap <= ALL_VALID_EQUAL_TOL


# ----------------------------------------------------------------------------------------------- exactness
@pytest.mark.parametrize("mode", ["float", "high"])
@pytest.mark.parametrize("shape", ["q_ca", "enc"])
def test_padded_tokens_do_not_reach_valid_outputs(shape, mode):
    """Valid outputs change by exactly 0.0 when padded tokens change (NaN in float); a valid token changes them."""
    tattn = torch_attention(shape, 16)
    nattn = native_attention(shape, tattn, spec=QUANT_SPECS[mode])
    x = make_inputs(shape, 17, sort=shape == "q_ca")
    base = n(native_out(nattn, x))
    pad = ~x["kv_mask"]
    fill = np.nan if mode == "float" else 100.0
    kv = np.where(pad[..., None], fill, x["kv"]).astype(np.float32)
    perturbed = {**x, "kv": kv, "q": kv if shape in SELF else x["q"]}
    out = n(native_out(nattn, perturbed))
    rows = np.ones(base.shape[:2], bool) if shape not in SELF else x["kv_mask"]  # output rows that are real tokens
    np.testing.assert_array_equal(out[rows], base[rows])
    valid_kv = x["kv"].copy()
    valid_kv[0, 0] += 5.0
    changed = {**x, "kv": valid_kv, "q": valid_kv if shape in SELF else x["q"]}
    assert np.abs(n(native_out(nattn, changed))[rows] - base[rows]).max() > 1e-3  # positive control


@pytest.mark.parametrize("mode", ["float", "high"])
def test_key_outside_a_query_mask_does_not_reach_that_query(mode):
    """attend: perturbing a key that query 0 does not see changes query 0 by exactly 0.0 and changes the queries that see it."""
    tattn = torch_attention("q_ca", 18)
    nattn = native_attention("q_ca", tattn, spec=QUANT_SPECS[mode])
    x = make_inputs("q_ca", 19)
    j = int(np.flatnonzero(~x["attn_mask"][0, 0] & x["kv_mask"][0] & x["attn_mask"][0, 1])[0])
    kv = x["kv"].copy()
    kv[0, j] += 5.0
    base, out = n(native_out(nattn, x)), n(native_out(nattn, {**x, "kv": kv}))
    np.testing.assert_array_equal(out[0, 0], base[0, 0])
    seen = x["attn_mask"][0, :, j]
    assert np.abs(out[0, seen] - base[0, seen]).max(-1).min() > 1e-4  # every query that sees key j moves


# ---------------------------------------------------------------------------------------------- quantized
def test_projection_fractional_bits_are_capped_and_the_table_fits_hls4ml():
    """E/F (head and tail) start and stay at <= seq_proj_frac_cap fractional bits; the renorm lanes are sized for the cap."""
    spec = QUANT_SPECS["high"]  # weight b0=20, i0=2: f0 = 18 before the cap
    nattn = native_attention("kv_ca", None, spec=spec)
    cap = LayerFactory(spec).quant.seq_proj_frac_cap
    assert cap == 7
    for proj in (nattn.core.E, nattn.core.F):
        for quantizer in (proj.proj.kq, proj.tail_q):
            f = quantizer.quantizer._f  # noqa: SLF001
            assert float(np.max(n(f.value))) == cap
            assert float(np.max(n(f.constraint(np.full(f.shape, 12.0))))) == cap  # an optimizer step cannot pass the cap
    lane_in = nattn.core.lut_k.iq.quantizer
    assert float(np.max(n(lane_in.i))) + float(np.max(n(lane_in.f))) == 1 + 2 * cap + 4
    assert nml.lut_entries(cap) == 2**19 <= 1_000_000
    assert nml.lut_entries(cap + 1) > 1_000_000  # 7 is the largest deployable cap
    # control: the q/k/v projections keep the spec's 18 fractional bits (the cap is local to E and F)
    assert float(np.max(n(nattn.q_proj.kq.quantizer.f))) == 18


# measured max relative error of the rescale at 1 valid key (production E grid, step 2^-6): 3.528e-3; 12-bit lanes 0.1848
RENORM_REL_TOL = 7.06e-3


def renorm_rel_error(lane=None, lut_out=None, n_valid: int = 1) -> float:
    n_keys, queries = 160, 64
    rng = np.random.default_rng(3)
    e = np.round(rng.uniform(-(K**-0.5), K**-0.5, (n_keys, K)) * 64) / 64
    e = e.astype(np.float32)
    factory = LayerFactory(PROD_SPEC)
    with factory.scopes():
        core = nml.AttendCore(factory, 1, 4, queries, n_keys, n_keys, K, name="r")
        if lane is not None:
            kept = factory.mask_einsum("bqn,znk->bqk", name="kept12", table_bits=lane)
            lut = factory.lut(nml.rsqrt_floor, name="lut12", in_bits=lane, out_bits=lut_out)
    core.build()
    core.E.assign_full(e)
    np.testing.assert_array_equal(n(core.E.rows(False)[0]), e)  # E is on the quantizer's grid
    if lane is None:
        kept, lut = core.kept_k, core.lut_k
    else:
        kept.build([(None, queries, n_keys), (1, n_keys, K)])
        lut.build((None, queries * K))
    mask = np.zeros((1, queries, n_keys), bool)
    for row in range(queries):
        mask[0, row, rng.choice(n_keys, n_valid, replace=False)] = True
    got = n(nml.rescale(kept, lut, mask, core.E.energy(False), False))
    ref = ml._rescale(t(e), t(mask), 1e-8).numpy()  # noqa: SLF001
    finite = ref < 1e6  # columns whose kept rows are all exactly 0 never reach an output ((S * M) E is 0 there)
    return float(np.abs(got / ref - 1)[finite].max())


def test_renorm_precision_bound_at_one_valid_key():
    err = renorm_rel_error()
    assert err <= RENORM_REL_TOL
    # control: the prototype's 12-bit lanes (QML 18.5%) exceed the bound by >= 10x
    assert renorm_rel_error(lane=(1, 12), lut_out=(7, 8)) >= 10 * RENORM_REL_TOL


EXPECTED_LEAVES = {
    "attend": {"qk": QEinsum, "E_proj": QEinsumDense, "F_proj": QEinsumDense, "softmax": QSoftmax, "pv": QEinsum}
    | {f"{kind}_{p}": cls for p in "kv" for kind, cls in (("kept", QEinsum), ("rsqrt", QUnaryFunctionLUT), ("mul", QEinsum))},
    "project": {"qk": QEinsum, "E_proj": QEinsumDense, "F_proj": QEinsumDense, "softmax": QSoftmax, "av": QEinsum}
    | {f"{kind}_{p}": cls for p in "kv" for kind, cls in (("kept", QEinsum), ("rsqrt", QUnaryFunctionLUT), ("mul", QEinsum))},
    "project_unmasked": {"qk": QEinsum, "E_proj": QEinsumDense, "F_proj": QEinsumDense, "softmax": QSoftmax, "av": QEinsum},
}
LIVE_MIN_CHANGE = 5e-3  # half the smallest measured change: q_ca 0.134 (rsqrt_k), enc 0.0112 (rsqrt_v), q_sa 0.243 (qk)


@pytest.mark.parametrize("shape", ["q_ca", "enc", "q_sa"])
def test_leaf_inventory_and_every_leaf_is_live(shape):
    """Quantized: every multiply of the core is the listed HGQ2 class (no float twin left), and coarsening any one leaf's
    quantizers to 1 fractional bit moves the output, so no leaf is bypassed or inert.
    """
    kind = SHAPES[shape][3]
    tattn = torch_attention(shape, 20, qkv_norm=True, norm="LayerNorm")
    nattn = native_attention(shape, tattn, spec=QUANT_SPECS["high"], name="a", qkv_norm=True, norm="LayerNorm")
    leaves = qleaves(nattn)
    expected = {f"a_{part}": QDense for part in ("q_proj", "k_proj", "v_proj", "out_proj")}
    expected |= {f"a_{kind}_{part}": cls for part, cls in EXPECTED_LEAVES[kind].items()}
    assert {name: type(leaf) for name, leaf in leaves.items()} == expected
    floats = [
        layer.name
        for layer in nattn._flatten_layers(recursive=True)  # noqa: SLF001
        if type(layer) in {EinsumOp, SoftmaxOp, keras.layers.EinsumDense, keras.layers.Activation}
    ]
    assert floats == []
    x = make_inputs(shape, 21, sort=shape == "q_ca")
    base = n(native_out(nattn, x))
    weak = {}
    for name, leaf in leaves.items():
        with coarsened(leaf) as count:
            assert count > 0, f"{name}: no quantizer bit variable"
            change = np.abs(n(native_out(nattn, x)) - base).max()
        if change <= LIVE_MIN_CHANGE:
            weak[name] = float(change)
    assert not weak, weak
    np.testing.assert_array_equal(n(native_out(nattn, x)), base)  # restored


# measured on q_ca (B=2, perturbed affines, sort): max |native - org float| high 2.503e-2, coarse 0.7552
HIGH_TOL = 5.01e-2


def test_paired_delta_high_vs_coarse():
    tattn = torch_attention("q_ca", 0, qkv_norm=True, norm="LayerNorm")
    x = make_inputs("q_ca", 22, sort=True)
    ref = torch_out(tattn, x)
    high = n(native_out(native_attention("q_ca", tattn, spec=QUANT_SPECS["high"], qkv_norm=True, norm="LayerNorm"), x))
    coarse = n(native_out(native_attention("q_ca", tattn, spec=COARSE_SPEC, qkv_norm=True, norm="LayerNorm"), x))
    err_high, err_coarse = np.abs(high - ref).max(), np.abs(coarse - ref).max()
    assert err_high <= HIGH_TOL
    assert coarse.std(axis=1).mean() > 0.05, "the coarse spec must still depend on the input"
    assert err_coarse >= 10 * err_high, (err_high, err_coarse)


# measured: 0.7727, 0.04861, 0.009735, 0.007950
SWEEP = [(6, 4), (10, 8), (14, 12), (20, 18)]  # (weight b0, datalane/table f0); seq_proj_frac_cap = b0 - i0


def test_error_falls_monotonically_with_bitwidth():
    """Above cap 7 the renorm table exceeds hls4ml's limit: these settings are a precision study, not deployable."""
    tattn = torch_attention("q_ca", 0, qkv_norm=True, norm="LayerNorm")
    x = make_inputs("q_ca", 22, sort=True)
    ref = torch_out(tattn, x)
    errors = []
    for b0, f0 in SWEEP:
        spec = {
            "weight": {"default_q_type": "kbi", "b0": b0, "i0": 2},
            "datalane": {"default_q_type": "kif", "i0": 8, "f0": f0},
            "table": {"default_q_type": "kif", "i0": 2, "f0": f0},
            "seq_proj_frac_cap": b0 - 2,
        }
        errors.append(float(np.abs(n(native_out(native_attention("q_ca", tattn, spec=spec, qkv_norm=True, norm="LayerNorm"), x)) - ref).max()))
    assert all(a > b for a, b in itertools.pairwise(errors)), errors
    assert errors[-1] <= 1.59e-2, errors  # 2x the measured 7.950e-3


@pytest.mark.parametrize("shape", ["q_ca", "kv_ca", "enc", "q_sa"])
def test_trains_under_jit_at_the_production_spec(shape):
    """One compile over 3 calls, finite gradients, a nonzero gradient for every trainable, EBOPs > 0 for every leaf."""
    nattn = native_attention(shape, None, spec=PROD_SPEC, qkv_norm=True, norm="LayerNorm")
    x = make_inputs(shape, 23, sort=shape == "q_ca")

    def loss_fn(tv, ntv, q, kv):
        args = (q,) if shape in SELF else (q, kv, kv)
        out, new_ntv, losses = nattn.stateless_call(
            tv, ntv, *args, kv_mask=x["kv_mask"], attn_mask=x["attn_mask"], kv_sort_idx=x["kv_sort_idx"], return_losses=True, training=True
        )
        qloss = sum(losses)
        return keras.ops.mean(keras.ops.square(out)) + 1e-12 * qloss, (new_ntv, qloss)

    step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    tv = [v.value for v in nattn.trainable_variables]
    ntv = [v.value for v in nattn.non_trainable_variables]
    nonzero = np.zeros(len(tv), bool)
    rng = np.random.default_rng(24)
    for _ in range(3):
        q = (x["q"] + rng.normal(0, 0.1, x["q"].shape)).astype(np.float32)
        kv = q if shape in SELF else (x["kv"] + rng.normal(0, 0.1, x["kv"].shape)).astype(np.float32)
        (loss, (ntv, qloss)), grads = step(tv, ntv, q, kv)
        assert all(np.isfinite(n(g)).all() for g in grads)
        nonzero |= np.array([bool(np.any(n(g) != 0)) for g in grads])
        assert np.isfinite(float(loss)) and float(qloss) > 0
    assert step._cache_size() == 1  # noqa: SLF001
    assert nonzero.all(), [v.path for v, ok in zip(nattn.trainable_variables, nonzero, strict=True) if not ok]

    # EBOPs per leaf: train-mode minus eval-mode losses (beta0 = 1), so the bit regularisers cancel
    def leaf_losses(training):
        native_out(nattn, x, training=training)
        return {name: float(sum(float(v) for v in leaf.losses)) for name, leaf in qleaves(nattn).items()}

    train, evaluation = leaf_losses(True), leaf_losses(False)
    assert {name: train[name] - evaluation[name] for name in train if train[name] - evaluation[name] <= 0} == {}


# ------------------------------------------------------------------------------------------ init and config
def test_in_projection_and_sequence_projection_init_match_org():
    """q/k/v kernels: xavier on the packed (3 dim, dim) matrix; zero bias; E/F (head and tail): U(+-1/sqrt(k))."""
    dim, seq_len, k = 512, 1024, 64
    keras.utils.set_random_seed(0)
    torch.manual_seed(0)
    nattn = NAttention(dim, num_heads=8, attn_type="linformer", linformer_seq_len=seq_len, linformer_k=k, name="init")
    nattn.build((None, 600, dim), (None, 700, dim))  # attend core, 324 tail rows
    tattn = Attention(dim, num_heads=8, attn_type="linformer", linformer_seq_len=seq_len, linformer_k=k)
    bound = math.sqrt(6 / (4 * dim))
    kernels = [n(layer.kernel) for layer in (nattn.q_proj, nattn.k_proj, nattn.v_proj)]
    ref = tattn.in_proj_weight.detach().numpy()
    for sample in [*kernels, ref]:
        assert np.abs(sample).max() <= bound * (1 + 1e-6)
        assert np.abs(sample).max() > 0.99 * bound
        assert sample.std() == pytest.approx(bound / math.sqrt(3), rel=0.02)
    for layer in (nattn.q_proj, nattn.k_proj, nattn.v_proj):
        assert np.all(n(layer.bias) == 0)
    assert not np.array_equal(kernels[0], kernels[1])
    proj_bound = 1 / math.sqrt(k)
    projections = [n(nattn.core.E.full()), n(nattn.core.F.full()), tattn.proj_k.detach().numpy()]
    assert projections[0].shape == projections[2].shape == (seq_len, k)
    for sample in projections:
        assert np.abs(sample).max() <= proj_bound * (1 + 1e-6)
        assert np.abs(sample).max() > 0.99 * proj_bound
        assert sample.std() == pytest.approx(proj_bound / math.sqrt(3), rel=0.02)
    assert not np.array_equal(projections[0], projections[1])


def test_build_attention_honours_linformer_seq_len_and_k():
    attn = build_attention(DIM, attn_type="linformer", num_heads=HEADS, linformer_seq_len=160, linformer_k=64, name="ba")
    assert type(attn) is NAttention
    attn.build((None, 150, DIM), (None, 160, DIM))
    assert tuple(attn.core.E.kernel.shape) == (160, 64)
    # the dense type accepts and ignores the linformer keys, as org's Attention signature does
    dense = build_attention(DIM, attn_type="torch", num_heads=HEADS, linformer_seq_len=160, linformer_k=64, name="bd")
    dense.build((None, 150, DIM), (None, 160, DIM))
    assert dense.core is None


@pytest.mark.parametrize(
    ("kwargs", "err", "match"),
    [
        ({"linformer_seq_len": 160, "linformer_proj_dim": 64}, TypeError, "linformer_k"),
        ({"linformer_seq_len": 160, "linformer_k": 64, "linformer_proj_dim": 64}, TypeError, "linformer_k"),
        ({"linformer_seq_len": 160, "linformer_K": 64}, TypeError, "unknown attention option"),
        ({"linformer_seq_len": 160, "linformer_k": 64, "attn_typ": "torch"}, TypeError, "unknown attention option"),
        ({"linformer_seq_len": 160}, ValueError, "linformer_seq_len and linformer_k"),
        ({"linformer_seq_len": 160, "linformer_k": 64, "linformer_core": "explicit"}, ValueError, "linformer_core"),
    ],
)
def test_build_attention_rejects_unknown_or_missing_options(kwargs, err, match):
    with pytest.raises(err, match=match):
        build_attention(DIM, attn_type="linformer", num_heads=HEADS, **kwargs)


def test_core_choice_and_call_mismatches():
    def build(core="auto", same_shape=False, q_len=150, kv_len=160, seq_len=160):
        attn = NAttention(DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=seq_len, linformer_k=K, linformer_core=core)
        q_shape = (None, q_len, DIM)
        attn.build(q_shape, q_shape, q_shape) if same_shape else attn.build(q_shape, (None, kv_len, DIM))
        return attn

    assert build().core_kind == "attend"
    assert build(same_shape=True, kv_len=150).core_kind == "project"  # (q, q, q): the decoder's q_sa build
    selfbuilt = NAttention(DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=168, linformer_k=K)
    selfbuilt.build((None, 168, DIM))
    assert selfbuilt.core_kind == "project"
    with pytest.raises(ValueError, match="exceeds linformer_seq_len"):
        build(kv_len=170)
    q = np.zeros((1, 150, DIM), np.float32)
    kv = np.zeros((1, 160, DIM), np.float32)
    with pytest.raises(ValueError, match="needs an attn_mask"):
        build()(q, kv, kv)
    with pytest.raises(ValueError, match="takes no attn_mask"):
        build(core="project")(q, kv, kv, attn_mask=np.ones((1, 150, 160), bool))
    with pytest.raises(ValueError, match="without mask support"):
        build(core="project_unmasked", same_shape=True, kv_len=150)(q, kv_mask=np.ones((1, 150), bool))
    with pytest.raises(ValueError, match="built length"):
        build()(q, kv[:, :100], kv[:, :100], attn_mask=np.ones((1, 150, 100), bool))
