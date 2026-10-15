"""S2 float parity: pure-torch hepattn modules vs the keras_native layers on JAX, in one process.

Tolerances are the docs/hgq/PARITY.md rows (dense/attention 1e-6/1e-5, encoder.clic_shaped 2e-6/2e-5,
Linformer 2e-5/2e-4 as tests/keras/test_linformer_parity.py). Biases, norm affines, LayerScale and register
tokens are perturbed away from their default init (``perturb_affine_params``), so a missing port of any of
them fails at O(0.1), far above these tolerances.
"""

import copy

import keras
import numpy as np
import pytest
import torch
from torch import nn

from hepattn.keras_native.attention import Attention as NAttention
from hepattn.keras_native.attention import merge_masks as native_merge_masks
from hepattn.keras_native.dense import Dense as NDense
from hepattn.keras_native.encoder import Encoder as NEncoder
from hepattn.keras_native.input import InputNet as NInputNet
from hepattn.keras_native.linformer import LinformerAttention as NLinformer
from hepattn.models.attention import Attention, merge_masks
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder
from hepattn.models.input import InputNet
from hepattn.models.linformer import LinformerAttention
from hepattn.models.posenc import FourierPositionEncoder

from .parity_utils import assert_parity, make_padded_batch
from .porting import perturb_affine_params, port_attention, port_dense, port_encoder, port_input_net, port_linformer


def t(x) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x))


def n(x) -> np.ndarray:
    return keras.ops.convert_to_numpy(x)


# ----------------------------------------------------------------------------------------------- Dense
DENSE_CONFIGS = {
    "clic_input_net_27_to_256": ({"input_size": 27, "output_size": 256}, {}),
    "clic_encoder_ffn_256": ({"input_size": 256}, {}),
    "clic_classification_head": ({"input_size": 256, "output_size": 6, "hidden_layers": [256, 128, 32]}, {}),
    "clic_regression_head_518": ({"input_size": 518, "output_size": 5, "hidden_layers": [512, 256, 128, 64, 32]}, {}),
    "swiglu_ffn": ({"input_size": 64, "activation": "SwiGLU"}, {}),
    "final_activation_sigmoid": (
        {"input_size": 32, "output_size": 8},
        {"t": {"final_activation": nn.Sigmoid()}, "n": {"final_activation": "sigmoid"}},
    ),
    "relu_activation": ({"input_size": 32, "output_size": 8}, {"t": {"activation": nn.ReLU()}, "n": {"activation": "torch.nn.ReLU"}}),
    "no_bias": ({"input_size": 32, "bias": False}, {}),
    "with_dropout_eval": ({"input_size": 32, "dropout": 0.1}, {}),
}


def dense_pair(name: str, seed: int = 42):
    cfg, extra = DENSE_CONFIGS[name]
    torch.manual_seed(seed)
    tdense = Dense(**cfg, **extra.get("t", {})).eval()
    perturb_affine_params(tdense, seed)
    ndense = NDense(**cfg, **extra.get("n", {}), name="d")
    ndense.build((None, 21, cfg["input_size"]))
    port_dense(tdense, ndense)
    return tdense, ndense


@pytest.mark.parametrize("name", DENSE_CONFIGS)
def test_dense_parity(name):
    tdense, ndense = dense_pair(name)
    x, _ = make_padded_batch(4, 21, DENSE_CONFIGS[name][0]["input_size"], seed=7)
    with torch.no_grad():
        ref = tdense(t(x))
    assert_parity("dense", name, ref, ndense(x, training=False), atol=1e-6, rtol=1e-5)


def test_dense_structure_and_names():
    """SwiGLU doubles the inner projection; names follow the hybrid ({name}_hidden{i}, {name}_final)."""
    gated = NDense(64, activation="SwiGLU", name="ffn")
    gated.build((None, 5, 64))
    assert gated.hidden[0].kernel.shape == (64, 256)
    assert [layer.name for layer in (*gated.hidden, gated.final)] == ["ffn_hidden0", "ffn_final"]
    plain = NDense(64, hidden_layers=2, name="ffn")
    plain.build((None, 5, 64))
    assert [layer.kernel.shape for layer in plain.hidden] == [(64, 128), (128, 128)]


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        ({"activation": "SwiGLU"}, NotImplementedError),
        ({"dropout": 0.1}, NotImplementedError),
        ({"activation": "Mish"}, ValueError),
    ],
)
def test_dense_unsupported_raises(kwargs, err):
    from hepattn.keras_native.factory import LayerFactory  # noqa: PLC0415

    factory = LayerFactory({"weight": {"default_q_type": "kbi"}}) if err is NotImplementedError else None
    with pytest.raises(err):
        NDense(32, factory=factory, **kwargs)


# ------------------------------------------------------------------------------------- merge_masks
@pytest.mark.parametrize("q_given", [False, True])
@pytest.mark.parametrize("kv_given", [False, True])
@pytest.mark.parametrize("attn_given", [False, True])
def test_merge_masks_equals_torch(q_given, kv_given, attn_given):
    rng = np.random.default_rng(3)
    q_mask = rng.random((2, 5)) > 0.3 if q_given else None
    kv_mask = rng.random((2, 7)) > 0.3 if kv_given else None
    attn_mask = rng.random((2, 5, 7)) > 0.5 if attn_given else None
    q_shape, k_shape = (2, 5, 4), (2, 7, 4)
    ref = merge_masks(*(None if m is None else t(m) for m in (q_mask, kv_mask, attn_mask)), torch.Size(q_shape), torch.Size(k_shape), "cpu")
    got = native_merge_masks(q_mask, kv_mask, attn_mask, q_shape, k_shape)
    if ref is None:
        assert got is None
        return
    np.testing.assert_array_equal(n(got), ref.numpy())
    # non-vacuous: the merged mask is neither all True nor all False
    assert 0 < ref.sum() < ref.numel()


# --------------------------------------------------------------------------------------- Attention
DIM, HEADS = 64, 8


def attention_pair(seed: int = 0, q_len: int = 24, kv_len: int | None = None, **kwargs):
    torch.manual_seed(seed)
    tattn = Attention(DIM, num_heads=HEADS, attn_type="torch", **kwargs).eval()
    perturb_affine_params(tattn, seed)
    nattn = NAttention(DIM, num_heads=HEADS, name="attn", **kwargs)
    nattn.build((None, q_len, DIM), None if kv_len is None else (None, kv_len, DIM))
    port_attention(tattn, nattn)
    return tattn, nattn


def test_qkv_projection_split():
    """Each ported projection matches the packed torch projection (localises a wrong chunk order)."""
    tattn, nattn = attention_pair(q_len=10, kv_len=14)
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal(s).astype(np.float32) for s in ((2, 10, DIM), (2, 14, DIM), (2, 14, DIM)))
    with torch.no_grad():
        q_t, k_t, v_t = nn.functional._in_projection_packed(t(q), t(k), t(v), tattn.in_proj_weight, tattn.in_proj_bias)  # noqa: SLF001
    assert_parity("attention.q_proj", "dim64h8", q_t, nattn.q_proj(q), atol=1e-6, rtol=1e-5)
    assert_parity("attention.k_proj", "dim64h8", k_t, nattn.k_proj(k), atol=1e-6, rtol=1e-5)
    assert_parity("attention.v_proj", "dim64h8", v_t, nattn.v_proj(v), atol=1e-6, rtol=1e-5)


def test_self_attention_parity_padded():
    tattn, nattn = attention_pair(qkv_norm=True, norm="LayerNorm")
    x, kv_mask = make_padded_batch(3, 24, DIM, seed=2)
    with torch.no_grad():
        ref = tattn(t(x), kv_mask=t(kv_mask))
    got = nattn(x, kv_mask=kv_mask)
    assert_parity("attention.self_masked", "dim64h8+qkvnorm", ref, got, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_cross_attention_parity_with_attn_mask():
    tattn, nattn = attention_pair(q_len=12, kv_len=20, qkv_norm=True, norm="LayerNorm")
    rng = np.random.default_rng(3)
    q, _ = make_padded_batch(2, 12, DIM, seed=3)
    k, kv_mask = make_padded_batch(2, 20, DIM, seed=4)
    attn_mask = rng.random((2, 12, 20)) > 0.3
    attn_mask[..., 0] = True
    with torch.no_grad():
        ref = tattn(t(q), k=t(k), kv_mask=t(kv_mask), attn_mask=t(attn_mask))
    got = nattn(q, k=k, kv_mask=kv_mask, attn_mask=attn_mask)
    assert_parity("attention.cross_masked", "dim64h8+qkvnorm", ref, got, atol=1e-6, rtol=1e-5)


def test_mask_polarity_probe():
    """Perturbing masked-out keys must not change valid outputs (catches an inverted mask)."""
    _, nattn = attention_pair(q_len=16)
    x, kv_mask = make_padded_batch(3, 16, DIM, seed=5)
    assert not kv_mask.all(), "fixture must contain masked-out slots"
    out_1 = n(nattn(x, kv_mask=kv_mask))
    out_2 = n(nattn(x + np.where(kv_mask[..., None], 0.0, 100.0).astype(np.float32), kv_mask=kv_mask))
    valid = np.broadcast_to(kv_mask[..., None], out_1.shape)
    np.testing.assert_array_equal(out_1[valid], out_2[valid])
    # positive control: perturbing a VALID key changes the valid outputs
    x_valid = x.copy()
    x_valid[:, 0] += 100.0
    assert not np.allclose(n(nattn(x_valid, kv_mask=kv_mask))[valid], out_1[valid], atol=1e-3)


def test_value_residual_two_layers():
    """Layer 2 mixes layer 1's projected values; a broken cache or mix transpose diverges O(1) at layer 2."""
    torch.manual_seed(6)
    t_first = Attention(DIM, num_heads=HEADS, value_residual=True, is_first_layer=True).eval()
    t_second = Attention(DIM, num_heads=HEADS, value_residual=True, is_first_layer=False).eval()
    perturb_affine_params(t_first, 6)
    perturb_affine_params(t_second, 7)
    n_first = NAttention(DIM, num_heads=HEADS, value_residual=True, is_first_layer=True, name="a0")
    n_second = NAttention(DIM, num_heads=HEADS, value_residual=True, is_first_layer=False, name="a1")
    n_first.build((None, 18, DIM))
    n_second.build((None, 18, DIM))
    port_attention(t_first, n_first)
    port_attention(t_second, n_second)

    x, kv_mask = make_padded_batch(2, 18, DIM, seed=7)
    iv_t: dict = {}
    with torch.no_grad():
        h_t = t_first(t(x), kv_mask=t(kv_mask), initial_values=iv_t)
        out_t = t_second(h_t, kv_mask=t(kv_mask), initial_values=iv_t)
    h_n, iv_n = n_first(x, kv_mask=kv_mask)
    out_n, iv_pass = n_second(h_n, kv_mask=kv_mask, initial_v=iv_n)
    assert iv_pass is iv_n, "a later layer must pass the first layer's cache on"
    assert_parity("attention.value_residual_l1", "dim64h8", h_t, h_n, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)
    assert_parity("attention.value_residual_l2", "dim64h8", out_t, out_n, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)
    # control: without the cache, layer 2 differs (the mix is not a no-op)
    no_mix, _ = n_second(h_n, kv_mask=kv_mask, initial_v=None)
    assert np.abs(n(no_mix) - n(out_n)).max() > 1e-3


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        ({"attn_type": "flex"}, ValueError),
        ({"window_size": 4}, ValueError),
        ({"qkv_norm": True}, ValueError),
        ({"qkv_norm": True, "norm": "RMSNorm"}, NotImplementedError),
        ({"qkv_norm": True, "norm": "BatchNorm"}, ValueError),
    ],
)
def test_attention_unsupported_raises(kwargs, err):
    with pytest.raises(err):
        NAttention(DIM, num_heads=HEADS, **kwargs)


# --------------------------------------------------------------------------------------- Linformer
L_DIM, L_HEADS, L_SEQ, L_K = 256, 16, 256, 256


def linformer_pair(seed: int, q_len: int, kv_len: int):
    torch.manual_seed(seed)
    tlin = LinformerAttention(L_DIM, seq_len=L_SEQ, k=L_K, heads=L_HEADS, dim_head=L_DIM // L_HEADS).eval()
    perturb_affine_params(tlin, seed)
    nlin = NLinformer(L_DIM, num_heads=L_HEADS, seq_len=L_SEQ, k=L_K, name="lin")
    nlin.build((None, q_len, L_DIM), (None, kv_len, L_DIM))
    port_linformer(tlin, nlin)
    return tlin, nlin


def test_linformer_self_attention_parity():
    tlin, nlin = linformer_pair(1, 168, 168)
    x, _ = make_padded_batch(2, 168, L_DIM, seed=2)
    with torch.no_grad():
        ref = tlin(t(x), k=t(x), v=t(x))
    assert_parity("linformer.self", "dim256h16k256", ref, nlin(x), atol=2e-5, rtol=2e-4)


def test_linformer_cross_attention_parity_masked():
    tlin, nlin = linformer_pair(3, 150, 168)
    rng = np.random.default_rng(4)
    q, _ = make_padded_batch(2, 150, L_DIM, seed=5)
    kv, _ = make_padded_batch(2, 168, L_DIM, seed=6)
    attn_mask = rng.random((2, 150, 168)) > 0.3
    attn_mask[..., 0] = True
    attn_mask[:, 1, :] = False  # one fully masked query: torch zeroes it, so must we
    with torch.no_grad():
        ref = tlin(t(q), k=t(kv), v=t(kv), attn_mask=t(attn_mask))
    got = nlin(q, k=kv, v=kv, attn_mask=attn_mask)
    assert_parity("linformer.cross_masked", "dim256h16k256", ref, got, atol=2e-5, rtol=2e-4)


def test_linformer_ignores_dense_attention_options():
    """Helen's semantics: kv_mask / q_mask / qkv_norm / value_residual change nothing."""
    _, nlin = linformer_pair(8, 40, 40)
    nlin_opts = NLinformer(L_DIM, num_heads=L_HEADS, seq_len=L_SEQ, k=L_K, qkv_norm=True, norm="LayerNorm", value_residual=True, name="lin2")
    nlin_opts.build((None, 40, L_DIM))
    nlin_opts.set_weights(nlin.get_weights())
    x, mask = make_padded_batch(2, 40, L_DIM, seed=9)
    ref = n(nlin(x))
    np.testing.assert_array_equal(n(nlin(x, kv_mask=mask, q_mask=mask)), ref)
    np.testing.assert_array_equal(n(nlin_opts(x)), ref)


def test_linformer_mask_needs_rank_ge_kv_len():
    nlin = NLinformer(32, num_heads=4, seq_len=64, k=16, name="small")
    x = np.zeros((1, 40, 32), np.float32)
    with pytest.raises(ValueError, match="must be >= kv_len"):
        nlin(x, attn_mask=np.ones((1, 40, 40), bool))


# ----------------------------------------------------------------------------------------- Encoder
def encoder_pair(seed: int, x_len: int, **cfg):
    ncfg = copy.deepcopy(cfg)  # torch Encoder/EncoderLayer write into attn_kwargs
    torch.manual_seed(seed)
    tenc = Encoder(**cfg).eval()
    perturb_affine_params(tenc, seed)
    nenc = NEncoder(**ncfg)
    nenc.build((None, x_len, cfg["dim"]))
    port_encoder(tenc, nenc)
    return tenc, nenc


def test_hybrid_norm_per_layer_parity():
    """Per-layer comparison localises hybrid-norm wiring errors (pre-norm at depth > 0, missing post-norm)."""
    cfg = {"num_layers": 3, "dim": 64, "hybrid_norm": True, "attn_kwargs": {"num_heads": 4}}
    tenc, nenc = encoder_pair(10, 20, **cfg)
    x, kv_mask = make_padded_batch(2, 20, 64, seed=11)
    h_t, h_n, iv_t, iv_n = t(x), x, {}, None
    for i, (tl, nl) in enumerate(zip(tenc.layers, nenc.layers, strict=True)):
        with torch.no_grad():
            h_t = tl(h_t, kv_mask=t(kv_mask), initial_values=iv_t)
        h_n, iv_n = nl(h_n, kv_mask=kv_mask, initial_v=iv_n)
        assert_parity("encoder.hybrid_norm", f"layer{i}", h_t, h_n, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_register_tokens_parity():
    cfg = {"num_layers": 2, "dim": 64, "num_register_tokens": 4, "attn_kwargs": {"num_heads": 4}}
    tenc, nenc = encoder_pair(12, 16, **cfg)
    x, kv_mask = make_padded_batch(2, 16, 64, seed=13)
    with torch.no_grad():
        ref = tenc(t(x), kv_mask=t(kv_mask))
    got = nenc(x, kv_mask=kv_mask)
    assert got.shape == x.shape, "register tokens must be stripped from the output"
    assert_parity("encoder.register_tokens", "2L+4reg", ref, got, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_layer_scale_prenorm_parity():
    """Pre-norm Residual with LayerScale (not covered by the hybrid suite); gamma is perturbed per channel."""
    cfg = {"num_layers": 2, "dim": 64, "layer_scale": 0.5, "attn_kwargs": {"num_heads": 4}}
    tenc, nenc = encoder_pair(16, 16, **cfg)
    x, kv_mask = make_padded_batch(2, 16, 64, seed=17)
    with torch.no_grad():
        ref = tenc(t(x), kv_mask=t(kv_mask))
    assert_parity("encoder.layer_scale", "2L_prenorm_ls0.5", ref, nenc(x, kv_mask=kv_mask), valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_clic_shaped_encoder_parity():
    """CLIC base.yaml encoder: 6 layers, dim 256, 16 heads, hybrid norm, value residual, 8 registers."""
    cfg = {"num_layers": 6, "dim": 256, "hybrid_norm": True, "value_residual": True, "num_register_tokens": 8, "attn_kwargs": {"num_heads": 16}}
    tenc, nenc = encoder_pair(14, 80, **cfg)
    x, kv_mask = make_padded_batch(2, 80, 256, seed=15)
    with torch.no_grad():
        ref = tenc(t(x), kv_mask=t(kv_mask))
    got = nenc(x, kv_mask=kv_mask)
    assert_parity("encoder.clic_shaped", "6L_dim256_h16_hybrid_vres_8reg", ref, got, valid_mask=kv_mask, atol=2e-6, rtol=2e-5)


def test_clic_linformer_encoder_parity():
    """pflow_hgq.yaml encoder: linformer (k=256, seq 256), 6 layers, dim 256, 160 nodes + 8 registers."""
    cfg = {
        "num_layers": 6,
        "dim": 256,
        "attn_type": "linformer",
        "hybrid_norm": True,
        "value_residual": True,
        "num_register_tokens": 8,
        "attn_kwargs": {"num_heads": 16, "linformer_seq_len": 256, "linformer_proj_dim": 256},
    }
    tenc, nenc = encoder_pair(18, 160, **cfg)
    x, kv_mask = make_padded_batch(2, 160, 256, seed=19)
    with torch.no_grad():
        ref = tenc(t(x), kv_mask=t(kv_mask))
    got = nenc(x, kv_mask=kv_mask)
    assert_parity("encoder.clic_linformer", "6L_dim256_h16_lin256_8reg", ref, got, valid_mask=kv_mask, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        ({"attn_type": "flex"}, ValueError),
        ({"window_size": 8}, ValueError),
        ({"score_mod": "relative_position"}, ValueError),
        ({"drop_path": 0.1}, NotImplementedError),
    ],
)
def test_encoder_unsupported_raises(kwargs, err):
    with pytest.raises(err):
        NEncoder(num_layers=1, dim=32, attn_kwargs={"num_heads": 4}, **kwargs)


def test_encoder_rejects_sorting():
    nenc = NEncoder(num_layers=1, dim=32, attn_kwargs={"num_heads": 4})
    with pytest.raises(NotImplementedError):
        nenc(np.zeros((1, 4, 32), np.float32), x_sort_value=np.zeros((1, 4), np.float32))


def test_encoder_layer_names_follow_hybrid():
    nenc = NEncoder(num_layers=2, dim=32, hybrid_norm=True, value_residual=True, num_register_tokens=8, attn_kwargs={"num_heads": 4})
    nenc.build((None, 10, 32))
    names = {layer.name for layer in nenc._flatten_layers(include_self=False)}  # noqa: SLF001
    expected = {
        "encoder_l0_attn_q_proj", "encoder_l0_attn_out_proj", "encoder_l0_attn_scores", "encoder_l0_attn_softmax",
        "encoder_l0_attn_values", "encoder_l0_ffn_hidden0", "encoder_l0_ffn_final", "encoder_l1_attn_value_residual_mix",
        "encoder_l0_attn_norm", "encoder_l0_ffn_norm", "encoder_l1_ffn_norm", "encoder_l1_attn_q_norm",
    }  # fmt: skip
    assert expected <= names, expected - names
    assert "encoder_l1_attn_norm" not in names, "hybrid norm: no pre-attention norm at depth > 0"


# ---------------------------------------------------------------------------------------- InputNet
def test_input_net_fourier_parity():
    """InputNet(Dense 27 -> 256) + FourierPositionEncoder(eta, phi), with the torch buffer B ported."""
    torch.manual_seed(20)
    tnet = InputNet(
        input_name="node",
        fields=["features"],
        net=Dense(input_size=27, output_size=256),
        posenc=FourierPositionEncoder(input_name="node", dim=256, fields=["eta", "phi"], scale=0.1),
    ).eval()
    perturb_affine_params(tnet, 20)
    nnet = NInputNet(
        "node",
        net={"input_size": 27, "output_size": 256},
        fields=["features"],
        posenc={"dim": 256, "fields": ["eta", "phi"], "scale": 0.1},
        name="innet0",
    )
    rng = np.random.default_rng(21)
    inputs = {
        "node_features": rng.standard_normal((2, 160, 27)).astype(np.float32),
        "node_eta": rng.uniform(-3, 3, (2, 160)).astype(np.float32),
        "node_phi": rng.uniform(-np.pi, np.pi, (2, 160)).astype(np.float32),
    }
    nnet(inputs)  # build
    assert not nnet.posenc.B.trainable and nnet.posenc.B.shape == (2, 128)
    port_input_net(tnet, nnet)
    with torch.no_grad():
        ref = tnet({k: t(v) for k, v in inputs.items()})
    assert_parity("input_net.fourier", "27_to_256+fourier256", ref, nnet(inputs), atol=1e-6, rtol=1e-5)
    # control: the posenc term is not negligible (B ported, not left at its own init)
    no_pe = n(nnet.net(inputs["node_features"]))
    assert np.abs(n(nnet(inputs)) - no_pe).max() > 0.5
