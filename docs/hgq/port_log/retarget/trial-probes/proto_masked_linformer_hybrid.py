"""Feasibility prototype: org PR #9 masked-linformer as a branch of the hybrid KerasAttention.

Measures float parity against the merged torch Attention(attn_type="linformer") on the two paths the CLIC
model uses (encoder self-attention with kv_mask; decoder cross-attention with a per-query attn_mask, kv_mask and
kv_sort_idx), shows the parity is discriminating, and runs the quantized branch at a high and a low bitwidth.
"""

import sys

import torch
from torch import nn

from hepattn.keras import set_keras_default_device

set_keras_default_device("cpu")
from masked_linformer import _rescale  # noqa: E402

from hepattn.keras.attention import KerasAttention  # noqa: E402
from hepattn.keras.factory import LayerFactory, apply_softmax  # noqa: E402
from hepattn.keras.porting import port_attention  # noqa: E402
from hepattn.models.attention import Attention  # noqa: E402

HIGH = {"weight": {"default_q_type": "kbi", "b0": 20, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 8, "f0": 16},
        "table": {"default_q_type": "kif", "i0": 2, "f0": 16}, "ebops": {"beta0": 0.0}}
LOW = {"weight": {"default_q_type": "kbi", "b0": 4, "i0": 1}, "datalane": {"default_q_type": "kif", "i0": 2, "f0": 2},
       "table": {"default_q_type": "kif", "i0": 1, "f0": 3}, "ebops": {"beta0": 0.0}}


class KerasMaskedLinformer(KerasAttention):
    """KerasAttention projections + masked_linformer.project/attend, with every matmul a factory einsum."""

    def __init__(self, dim, num_heads, seq_len, k, factory, renorm_k_bug=False, **kw):
        super().__init__(dim, num_heads=num_heads, factory=factory, **kw)
        self.proj_k = nn.Parameter(torch.empty(seq_len, k))
        self.proj_v = nn.Parameter(torch.empty(seq_len, k))
        self.renorm_k_bug = renorm_k_bug  # discrimination switch: drop the key-column renorm
        self.seq_k = factory.einsum("bhnd,nk->bhkd", name="seq_k")
        self.seq_v = factory.einsum("bhnd,nk->bhkd", name="seq_v")
        self.s_e = factory.einsum("bhqn,nk->bhqk", name="s_e")
        self.a_f = factory.einsum("bhqk,nk->bhqn", name="a_f")

    def forward(self, q, k=None, v=None, kv_mask=None, attn_mask=None, kv_sort_idx=None, **_):
        t = self.training
        if k is None:
            k = v = q
        qp = self.separate_heads(self.q_proj(q, training=t))
        kp = self.separate_heads(self.k_proj(k, training=t))
        vp = self.separate_heads(self.v_proj(v, training=t))
        if kv_sort_idx is not None:
            idx = kv_sort_idx[:, None, :, None]
            kp, vp = (x.gather(-2, idx.expand_as(x)) for x in (kp, vp))
            kv_mask = None if kv_mask is None else kv_mask.gather(-1, kv_sort_idx)
            attn_mask = None if attn_mask is None else attn_mask.gather(-1, kv_sort_idx[:, None].expand_as(attn_mask))
        n = kp.shape[-2]
        ek, ev = self.proj_k[:n], self.proj_v[:n]
        scale = self.head_dim**-0.5
        if attn_mask is None:  # masked_linformer.project, then plain attention
            if kv_mask is not None:
                keep = kv_mask[:, None, :, None]
                kp, vp = torch.where(keep, kp, 0.0), torch.where(keep, vp, 0.0)
            kp, vp = self.seq_k([kp, ek], training=t), self.seq_v([vp, ev], training=t)
            if kv_mask is not None:
                kp = kp * _rescale(self.proj_k, kv_mask, 1e-8)[:, None, :, None]
                vp = vp * _rescale(self.proj_v, kv_mask, 1e-8)[:, None, :, None]
            attn = apply_softmax(self.attn_softmax, self.scores_einsum([qp, kp], training=t) * scale, training=t)
            out = self.values_einsum([attn, vp], training=t)
        else:  # masked_linformer.attend
            if kv_mask is not None:
                attn_mask = attn_mask & kv_mask[:, None]
            m = attn_mask[:, None]
            seen = m.any(-2)[..., None]
            kp, vp = torch.where(seen, kp, 0.0), torch.where(seen, vp, 0.0)
            s = torch.where(m, self.scores_einsum([qp, kp], training=t), 0.0) * scale
            s = self.s_e([s, ek], training=t)
            if not self.renorm_k_bug:
                s = s * _rescale(self.proj_k, attn_mask, 1e-8)[:, None]
            a = apply_softmax(self.attn_softmax, s, training=t)
            a = a * _rescale(self.proj_v, attn_mask, 1e-8)[:, None]
            out = self.values_einsum([torch.where(m, self.a_f([a, ev], training=t), 0.0), vp], training=t)
        return self.out_proj(self.recombine_heads(out), training=t)


def pair(quant=None, seq_len=168, k=64, bug=False):
    torch.manual_seed(0)
    ref = Attention(32, num_heads=4, attn_type="linformer", linformer_seq_len=seq_len, linformer_k=k).eval()
    f = LayerFactory(quant)
    with f.scopes():  # HGQ2 config scopes must be active at construction (as in KerasMaskFormer)
        kat = KerasMaskedLinformer(32, 4, seq_len, k, f, renorm_k_bug=bug).eval()
    if quant is None:
        port_attention(ref, kat)
    kat.proj_k.data.copy_(ref.proj_k.data)
    kat.proj_v.data.copy_(ref.proj_v.data)
    return ref, kat


g = torch.Generator().manual_seed(1)
x = torch.randn(2, 168, 32, generator=g)
kvm = torch.ones(2, 168, dtype=torch.bool)
kvm[0, 120:] = False
kvm[1, 60:] = False
q = torch.randn(2, 150, 32, generator=g)
kv = torch.randn(2, 160, 32, generator=g)
kvm2 = torch.ones(2, 160, dtype=torch.bool)
kvm2[:, 140:] = False
am = torch.rand(2, 150, 160, generator=g) > 0.7
sort = torch.argsort(torch.rand(2, 160, generator=g).masked_fill(~kvm2, float("inf")), dim=-1)


def diffs(bug=False):
    ref, kat = pair(bug=bug)
    with torch.no_grad():
        enc = (ref(x, kv_mask=kvm) - kat(x, kv_mask=kvm))[kvm].abs().max().item()
        ca = (ref(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort) - kat(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort)).abs().max().item()
    return enc, ca


print("float parity  encoder SA max|d| = %.2e   decoder CA (attend+sort) max|d| = %.2e" % diffs())
print("discrimination (no key renorm)  encoder %.2e   decoder CA %.2e" % diffs(bug=True))
for tag, spec in (("high", HIGH), ("low", LOW)):
    ref, kat = pair(quant=spec)
    try:
        with torch.no_grad():
            kat(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort)  # materialize the lazy quantized layers
        port_attention(ref, kat)
        with torch.no_grad():
            d = (ref(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort) - kat(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort)).abs().max().item()
        kat.train()
        out = kat(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort)
        out.sum().backward()
        print(f"quant {tag}: max|quant - torch float| = {d:.2e}; train fwd+bwd finite={bool(torch.isfinite(out).all())}, proj_k.grad finite={bool(torch.isfinite(kat.proj_k.grad).all())}")
    except Exception as e:  # noqa: BLE001
        print(f"quant {tag}: FAIL {type(e).__name__}: {str(e)[:200]}")
