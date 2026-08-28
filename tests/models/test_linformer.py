"""Linformer attention: the properties that are easy to break and expensive to notice.

Two of these guard behaviour that is NOT obvious from the config:

- `test_masked_path_requires_k_at_least_seq_len` pins a hard constraint. The mask is
  built in the *projected* space (shape ...,k) but is filled from an original-sequence
  mask (shape ...,kv_len), so `k < kv_len` raises. This is why the CLIC config runs
  k=256 against 168 hits -- the "compression" matrix is an EXPANSION, and it is forced,
  not chosen. Anyone who lowers k to get a speedup will hit this.

- `test_fully_masked_row_is_finite_and_attends_to_nothing` covers helenllii's fix
  (e207a434). Softmax over an all -inf row is NaN; the fix routes those rows around the
  softmax. MaskFormer's mask-attention produces such rows whenever a query's predicted
  mask is empty, so without this a run dies partway through training.
"""

import pytest
import torch

from hepattn.models.attention import Attention
from hepattn.models.linformer import LinformerAttention

DIM, HEADS, N = 64, 4, 24


def make(k, seq_len=None):
    torch.manual_seed(0)
    return LinformerAttention(dim=DIM, seq_len=seq_len or max(k, N), k=k, heads=HEADS).eval()


def test_forward_shape_and_projection_receives_gradient():
    """Output shape must be unchanged, and the projection E must actually be trained."""
    m = make(k=N)
    x = torch.randn(2, N, DIM)
    out = m(x, x, x)
    assert out.shape == (2, N, DIM)

    out.sum().backward()
    for name in ("proj_k", "proj_v"):
        g = getattr(m, name).grad
        assert g is not None and g.abs().sum() > 0, f"{name} got no gradient -- E is not being learned"


def test_output_depends_on_the_projection_matrix():
    """Mutating E must change the output; catches a stub that ignores the projection."""
    m = make(k=N)
    x = torch.randn(2, N, DIM)
    with torch.no_grad():
        before = m(x, x, x).clone()
        m.proj_k.add_(1.0)
        after = m(x, x, x)
    assert not torch.allclose(before, after), "output ignores proj_k -- projection is not wired in"


@pytest.mark.parametrize("k", [N // 2, N - 1])
def test_masked_path_requires_k_at_least_seq_len(k):
    """k < kv_len raises on the masked path -- the documented reason CLIC must use k > n."""
    m = make(k=k, seq_len=4 * N)
    x = torch.randn(2, N, DIM)
    mask = torch.ones(2, N, N)
    assert torch.isfinite(m(x, x, x)).all(), "unmasked path should still work for k < n"
    with pytest.raises(RuntimeError):
        m(x, x, x, attn_mask=mask)


def test_fully_masked_row_is_finite_and_attends_to_nothing():
    """A query masked against every key must give finite output, not NaN."""
    m = make(k=N)
    x = torch.randn(2, N, DIM)
    mask = torch.ones(2, N, N)
    mask[0, 3, :] = 0  # query 3 of batch 0 attends to nothing

    out = m(x, x, x, attn_mask=mask)
    assert torch.isfinite(out).all(), "fully-masked row produced NaN/Inf (softmax over all -inf)"

    # The row must be the bias alone: attention output is zero, so only to_out's bias survives.
    assert torch.allclose(out[0, 3], m.to_out.bias, atol=1e-6), "fully-masked row did not attend to nothing"


def test_wired_through_the_attention_layer_without_stray_parameters():
    """attn_type='linformer' must skip the unused qkv/out projections of standard Attention."""
    a = Attention(dim=DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=N, linformer_proj_dim=N)
    x = torch.randn(2, N, DIM)
    assert a(x, x, x).shape == (2, N, DIM)
    assert not hasattr(a, "in_proj_weight"), "linformer allocated the standard in_proj it never uses"


def test_padding_does_not_leak_into_real_tokens():
    """Padded key/value slots must not influence real tokens.

    `linformer` is in VARLEN_ATTN_TYPES, so Attention.forward accepts kv_mask -- but the
    Linformer call originally ignored it, and padded hits were attended to as real ones
    (measured leak 1.3e+01 where torch gives exactly 0). Score-space masking cannot fix
    this: K/V are projected along the SEQUENCE axis first, so every projected column is
    already a mixture of the padded rows. They must be zeroed pre-projection.
    """
    B, n, real = 2, 24, 16
    kv_mask = torch.zeros(B, n, dtype=torch.bool)
    kv_mask[:, :real] = True

    torch.manual_seed(0)
    a = Attention(dim=DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=n, linformer_proj_dim=n).eval()
    torch.manual_seed(1)
    x = torch.randn(B, n, DIM)
    y = x.clone()
    y[:, real:] = torch.randn(B, n - real, DIM) * 50  # garbage in PAD slots only

    with torch.no_grad():
        o1 = a(x, x, x, kv_mask=kv_mask)[:, :real]
        o2 = a(y, y, y, kv_mask=kv_mask)[:, :real]
    assert torch.allclose(o1, o2, atol=1e-6), f"padding leaked into real tokens (max diff {(o1 - o2).abs().max():.3e})"


def test_cross_attention_with_padding_runs():
    """Decoder shape: fewer queries than keys, with a padded key axis."""
    B, nq, nkv, real = 2, 20, 24, 16
    kv_mask = torch.zeros(B, nkv, dtype=torch.bool)
    kv_mask[:, :real] = True
    a = Attention(dim=DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=nkv, linformer_proj_dim=nkv).eval()
    with torch.no_grad():
        out = a(torch.randn(B, nq, DIM), torch.randn(B, nkv, DIM), torch.randn(B, nkv, DIM), kv_mask=kv_mask)
    assert out.shape == (B, nq, DIM)
    assert torch.isfinite(out).all()


def test_qkv_norm_is_applied_and_trained():
    """qkv-norm must reach the Linformer path.

    The branch skips `_prepare_qkv`, where q/k/v_norm are applied in the standard path.
    Before the fix these 27,648 parameters (3 norms x 2 x dim x 18 modules on the CLIC
    model) received no gradient at all: the Linformer trained without a normalization both
    baselines have, and DDP aborted on the unused parameters.
    """
    a = Attention(dim=DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=N, linformer_proj_dim=N, qkv_norm=True, norm="LayerNorm")
    x = torch.randn(2, N, DIM)
    a(x, x, x).sum().backward()

    for which in ("q_norm", "k_norm", "v_norm"):
        for pname, p in getattr(a, which).named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"{which}.{pname} got no gradient"


def test_no_dead_parameters_in_the_shipped_configuration():
    """value_residual=False + qkv_norm=True must leave nothing unused.

    Dead parameters are not cosmetic here: DDP raises on them, which is why the config
    needed ddp_find_unused_parameters_true. value_residual stays off because its blend is
    per-input-position while v has been compressed to length k -- the tensors do not
    broadcast, so it cannot be applied to a Linformer without a redesign.
    """
    a = Attention(
        dim=DIM,
        num_heads=HEADS,
        attn_type="linformer",
        linformer_seq_len=N,
        linformer_proj_dim=N,
        qkv_norm=True,
        norm="LayerNorm",
        value_residual=False,
        is_first_layer=False,
    )
    x = torch.randn(2, N, DIM)
    a(x, x, x).sum().backward()
    dead = [n for n, p in a.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"unused parameters would abort DDP: {dead}"
