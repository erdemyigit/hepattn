"""Float parity: hepattn Attention (torch backend) vs KerasAttention.

The tests are layered so failures localize: the packed-QKV split is checked
per-projection first (a wrong chunk order or missing transpose is O(1) there but
can be partially masked in a full attention output), then full attention under
padding and attention masks, then the mask-polarity probe, then the two-layer
value-residual chain.
"""

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import assert_parity, make_padded_batch  # ty: ignore [unresolved-import]

from hepattn.keras.attention import KerasAttention
from hepattn.keras.porting import port_attention
from hepattn.models.attention import Attention

DIM = 64
HEADS = 8


def build_pair(seed: int = 0, **kwargs) -> tuple[Attention, KerasAttention]:
    torch.manual_seed(seed)
    tattn = Attention(DIM, num_heads=HEADS, attn_type="torch", **kwargs).eval()
    kattn = KerasAttention(
        DIM,
        num_heads=HEADS,
        bias=kwargs.get("bias", True),
        qkv_norm=kwargs.get("qkv_norm", False),
        norm=kwargs.get("norm"),
        value_residual=kwargs.get("value_residual", False),
        is_first_layer=kwargs.get("is_first_layer", False),
    ).eval()
    port_attention(tattn, kattn)
    return tattn, kattn


def test_qkv_projection_split():
    """Each ported projection must individually match the packed torch projection."""
    tattn, kattn = build_pair()
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(2, 10, DIM, generator=gen)
    k = torch.randn(2, 14, DIM, generator=gen)
    v = torch.randn(2, 14, DIM, generator=gen)
    with torch.no_grad():
        q_t, k_t, v_t = F._in_projection_packed(q, k, v, tattn.in_proj_weight, tattn.in_proj_bias)  # noqa: SLF001  # ty: ignore [unresolved-attribute]
    assert_parity("attention.q_proj", "dim64h8", q_t, kattn.q_proj(q), atol=1e-6, rtol=1e-5)
    assert_parity("attention.k_proj", "dim64h8", k_t, kattn.k_proj(k), atol=1e-6, rtol=1e-5)
    assert_parity("attention.v_proj", "dim64h8", v_t, kattn.v_proj(v), atol=1e-6, rtol=1e-5)


def test_self_attention_parity_padded():
    tattn, kattn = build_pair(qkv_norm=True, norm="LayerNorm")
    x, kv_mask = make_padded_batch(3, 24, DIM, seed=2)
    with torch.no_grad():
        out_t = tattn(x, kv_mask=kv_mask)
    out_k = kattn(x, kv_mask=kv_mask)
    assert_parity("attention.self_masked", "dim64h8+qkvnorm", out_t, out_k, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_cross_attention_parity_with_attn_mask():
    tattn, kattn = build_pair(qkv_norm=True, norm="LayerNorm")
    gen = torch.Generator().manual_seed(3)
    q, _ = make_padded_batch(2, 12, DIM, seed=3)
    k, kv_mask = make_padded_batch(2, 20, DIM, seed=4)
    attn_mask = torch.rand(2, 12, 20, generator=gen) > 0.3
    attn_mask[..., 0] = True  # every query attends to at least one key
    with torch.no_grad():
        out_t = tattn(q, k=k, kv_mask=kv_mask, attn_mask=attn_mask)
    out_k = kattn(q, k=k, kv_mask=kv_mask, attn_mask=attn_mask)
    assert_parity("attention.cross_masked", "dim64h8+qkvnorm", out_t, out_k, atol=1e-6, rtol=1e-5)


def test_mask_polarity_probe():
    """Perturbing masked-out keys must not change any output.

    Catches inverted masks that tolerance-based comparisons can miss when fixture
    masks are mostly True.
    """
    _, kattn = build_pair()
    x, kv_mask = make_padded_batch(3, 16, DIM, seed=5)
    assert not bool(kv_mask.all()), "fixture must contain masked-out slots"

    out_1 = kattn(x, kv_mask=kv_mask)
    x_perturbed = x + torch.where(kv_mask.unsqueeze(-1), torch.zeros(()), torch.full((), 100.0))
    out_2 = kattn(x_perturbed, kv_mask=kv_mask)

    valid = kv_mask.unsqueeze(-1).expand_as(out_1)
    assert torch.equal(out_1[valid], out_2[valid]), "outputs at valid slots changed when only masked-out keys were perturbed"


def test_value_residual_two_layers():
    """Layer-2 output depends on layer-1's cached initial values.

    A broken cache or a wrong mix transpose matches layer 1 but diverges O(1) at layer 2.
    """
    torch.manual_seed(6)
    t_first = Attention(DIM, num_heads=HEADS, attn_type="torch", value_residual=True, is_first_layer=True).eval()
    t_second = Attention(DIM, num_heads=HEADS, attn_type="torch", value_residual=True, is_first_layer=False).eval()
    k_first = KerasAttention(DIM, num_heads=HEADS, value_residual=True, is_first_layer=True).eval()
    k_second = KerasAttention(DIM, num_heads=HEADS, value_residual=True, is_first_layer=False).eval()
    port_attention(t_first, k_first)
    port_attention(t_second, k_second)

    x, kv_mask = make_padded_batch(2, 18, DIM, seed=7)
    iv_t: dict = {}
    iv_k: dict = {}
    with torch.no_grad():
        h_t = t_first(x, kv_mask=kv_mask, initial_values=iv_t)
        out_t = t_second(h_t, kv_mask=kv_mask, initial_values=iv_t)
    h_k = k_first(x, kv_mask=kv_mask, initial_values=iv_k)
    out_k = k_second(h_k, kv_mask=kv_mask, initial_values=iv_k)

    assert iv_k, "keras first layer did not cache initial values"
    assert_parity("attention.value_residual_l1", "dim64h8", h_t, h_k, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)
    assert_parity("attention.value_residual_l2", "dim64h8", out_t, out_k, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)
