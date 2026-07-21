"""Keras/HGQ2 twin of hepattn.models.linformer.LinformerAttention.

Mirrors Helen's LinformerAttention (lucidrains-style low-rank attention) op-for-op,
with the parameterized/quantizable pieces built through the LayerFactory:
q/k/v/out projections (QDense), the two attention contractions (QEinsum), and the
softmax (QSoftmax). The two sequence-length projection matrices proj_k/proj_v remain
float nn.Parameters — a documented conversion boundary (like the norms), to be
refined to QEinsumDense for full FPGA synthesis.

Interface matches KerasAttention.forward so it drops into the Residual/encoder/decoder
wrappers unchanged. Masking follows Helen's exact projected-space convention, which
requires the projection rank k >= key sequence length.
"""

import math

import torch
from torch import Tensor, nn

from hepattn.keras.factory import LayerFactory, apply_softmax


class KerasLinformerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        seq_len: int = 256,
        k: int = 256,
        bias: bool = True,
        attn_type: str = "linformer",
        factory: LayerFactory | None = None,
        name: str | None = None,
        **_ignored,
    ) -> None:
        """Args mirror the LinformerAttention/Attention surface used by the CLIC config.

        Extra Attention kwargs (qkv_norm, value_residual, ...) are accepted and ignored,
        exactly as Helen's linformer ignores them (it bypasses _prepare_qkv).

        Raises:
            ValueError: If attn_type is not 'linformer'.
        """
        super().__init__()
        factory = factory or LayerFactory()
        if attn_type != "linformer":
            raise ValueError(f"KerasLinformerAttention only supports attn_type='linformer', got '{attn_type}'")
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.k = k
        self.attn_type = "linformer"

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        # q/k/v projections are bias-free (as in LinformerAttention); out projection has bias
        self.to_q = factory.dense(dim, use_bias=False, name=sub("to_q"))
        self.to_k = factory.dense(dim, use_bias=False, name=sub("to_k"))
        self.to_v = factory.dense(dim, use_bias=False, name=sub("to_v"))
        self.to_out = factory.dense(dim, use_bias=True, name=sub("to_out"))
        for layer in (self.to_q, self.to_k, self.to_v, self.to_out):
            if not factory.quantize:
                layer.build((None, dim))

        # sequence-length projection matrices (float boundary; see module docstring)
        self.proj_k = nn.Parameter(self._init_proj(torch.zeros(seq_len, k)))
        self.proj_v = nn.Parameter(self._init_proj(torch.zeros(seq_len, k)))

        # quantizable attention core
        self.seq_proj_k = factory.einsum("bnd,nk->bkd", name=sub("seqproj_k"))
        self.seq_proj_v = factory.einsum("bnd,nk->bkd", name=sub("seqproj_v"))
        self.scores_einsum = factory.einsum("bhnd,bhkd->bhnk", name=sub("scores"))
        self.attn_softmax = factory.softmax(axis=-1, name=sub("softmax"))
        self.values_einsum = factory.einsum("bhnk,bhkd->bhnd", name=sub("values"))

    @staticmethod
    def _init_proj(t: Tensor) -> Tensor:
        std = 1.0 / math.sqrt(t.shape[-1])
        return t.uniform_(-std, std)

    def set_backend(self, attn_type: str, **kwargs) -> str:
        if attn_type != "linformer":
            raise ValueError(f"KerasLinformerAttention cannot switch to backend '{attn_type}'")
        return self.attn_type

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        attn_mask: Tensor | None = None,
        **_ignored,
    ) -> Tensor:
        """Low-rank attention mirroring hepattn.models.linformer.LinformerAttention.

        With attn_mask, the projection rank k must be >= the key sequence length (the mask
        is applied in projected space, as in the torch reference).
        """
        if k is None and v is None:  # self-attention
            k = v = q
        assert k is not None and v is not None
        b, n, _d = q.shape
        kv_len = k.shape[1]
        assert kv_len <= self.seq_len, f"kv_len {kv_len} exceeds seq_len {self.seq_len}"

        queries = self.to_q(q, training=self.training)
        keys = self.to_k(k, training=self.training)
        values = self.to_v(v, training=self.training)

        # project keys/values along the sequence axis to rank k (slicing proj for kv_len < seq_len)
        proj_k = self.proj_k[:kv_len]
        proj_v = self.proj_v[:kv_len]
        keys = self.seq_proj_k([keys, proj_k], training=self.training)
        values = self.seq_proj_v([values, proj_v], training=self.training)

        # split heads: queries (b,h,n,dh); keys/values (b,h,k,dh)
        queries = queries.reshape(b, n, self.num_heads, -1).transpose(1, 2)
        keys = keys.reshape(b, self.k, self.num_heads, -1).transpose(1, 2)
        values = values.reshape(b, self.k, self.num_heads, -1).transpose(1, 2)

        scores = self.scores_einsum([queries, keys], training=self.training) * (self.head_dim**-0.5)

        keep = None
        if attn_mask is not None:
            # mask is applied in the projected (rank-k) space, matching the torch reference,
            # which requires k >= kv_len. keep[..., :kv_len] = attend mask; [kv_len:] never attended.
            assert self.k >= kv_len, f"linformer projection rank k={self.k} must be >= kv_len={kv_len} for masking"
            keep = torch.zeros(b, 1, n, self.k, dtype=torch.bool, device=scores.device)
            keep[..., :kv_len] = attn_mask.to(torch.bool)[:, None, ...]

        attn = apply_softmax(self.attn_softmax, scores, attn_mask=keep, training=self.training)
        out = self.values_einsum([attn, values], training=self.training)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out, training=self.training)
