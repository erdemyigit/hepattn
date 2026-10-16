"""Keras/HGQ2 twin of hepattn.models.attention.Attention (torch-SDPA and masked-Linformer semantics).

The torch Attention packs Q/K/V into one (3*dim, dim) projection. This twin uses three separate factory-built
projections (so each can be quantized independently) and an explicit attention core whose scores and weights are
quantizable tensors:
- ``attn_type="torch"``: einsum -> masked softmax -> einsum. In float mode this reproduces the torch backend
  numerically within fp32 tolerance (docs/hgq/PARITY.md).
- ``attn_type="linformer"``: org ``Attention._linformer_attention`` (PR #9). The projections, ``qkv_norm`` and the
  value residual are the ones of the dense path; the core is ``hepattn.keras_native.masked_linformer`` (``AttendCore``
  or ``ProjectCore``), the same keras code the native port runs, here under the torch backend
  (docs/hgq/RETARGET_ORG_MAIN.md section 4, gate H0).

flash/flex/varlen backends, attention biases and score mods are training-time performance features of the torch
model with no keras/HGQ2 counterpart.
"""

import math

import torch
from torch import Tensor, nn

from hepattn.keras.factory import LayerFactory, apply_softmax
from hepattn.keras_native.masked_linformer import AttendCore, ProjectCore
from hepattn.models.attention import merge_masks
from hepattn.models.norm import NORM_TYPES

ATTN_TYPES = ("torch", "linformer")
LINFORMER_CORES = ("auto", "attend", "project", "project_unmasked")


class KerasAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        torch_compile: bool = False,
        window_size: int | None = None,
        qkv_norm: bool = False,
        norm: str | None = None,
        value_residual: bool = False,
        is_first_layer: bool = False,
        linformer_seq_len: int | None = None,
        linformer_k: int | None = None,
        linformer_core: str = "auto",
        factory: LayerFactory | None = None,
        name: str | None = None,
    ) -> None:
        """Multi-head attention from factory-built (quantizable) leaves.

        Accepts the org/main hepattn Attention arguments, so torch attn_kwargs from the YAML configs pass through.

        ``linformer_core`` (hybrid only) chooses the org path of a linformer layer: ``"attend"`` (called with an
        ``attn_mask``), ``"project"`` (no ``attn_mask``; ``kv_mask`` optional), ``"project_unmasked"`` (no mask at
        all: no renorm leaves; the projection rows past the key length are kept only as buffers, for the torch
        state_dict layout), or ``"auto"``: ``"attend"`` if the first call has an ``attn_mask``, else ``"project"``,
        as org decides per call. The core has static shapes, so it is built at the first forward, in float and in
        quantized mode; later calls must match it.

        Raises:
            ValueError: If an attn_type other than torch/linformer, a backend-specific option, or linformer without
                ``linformer_seq_len`` and ``linformer_k`` is requested.
        """
        super().__init__()
        factory = factory or LayerFactory()

        assert dim % num_heads == 0, "num_heads must divide dim."
        if attn_type not in ATTN_TYPES:
            raise ValueError(f"KerasAttention implements attn_type {ATTN_TYPES}, got attn_type='{attn_type}'")
        if torch_compile or window_size is not None:
            raise ValueError("torch_compile / window_size are not supported by KerasAttention")
        if qkv_norm and not norm:
            raise ValueError("norm must be provided when qkv_norm is True")
        if norm is not None and norm not in NORM_TYPES:
            raise ValueError(f"Unsupported norm: {norm}. Must be one of {list(NORM_TYPES.keys())}")
        if attn_type == "linformer" and (linformer_seq_len is None or linformer_k is None):
            raise ValueError("linformer attention needs linformer_seq_len and linformer_k")
        if linformer_core not in LINFORMER_CORES:
            raise ValueError(f"linformer_core must be one of {LINFORMER_CORES}, got '{linformer_core}'")

        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_type = attn_type
        self.qkv_norm = qkv_norm
        self.value_residual = value_residual
        self.is_first_layer = is_first_layer
        self.name = name

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        # float: eager build; quantized: lazy build on first forward (see KerasDense)
        self.q_proj = factory.dense(dim, use_bias=bias, name=sub("q_proj"))
        self.k_proj = factory.dense(dim, use_bias=bias, name=sub("k_proj"))
        self.v_proj = factory.dense(dim, use_bias=bias, name=sub("v_proj"))
        self.out_proj = factory.dense(dim, use_bias=bias, name=sub("out_proj"))
        if not factory.quantize:
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                layer.build((None, dim))

        if self.value_residual and not self.is_first_layer:
            self.value_residual_mix = factory.dense(num_heads, activation="sigmoid", use_bias=True, name=sub("value_residual_mix"))
            if not factory.quantize:
                self.value_residual_mix.build((None, dim))

        if self.qkv_norm:
            assert norm is not None
            norm_cls = NORM_TYPES[norm]
            self.q_norm = norm_cls(dim)
            self.k_norm = norm_cls(dim)
            self.v_norm = norm_cls(dim)

        if attn_type == "linformer":
            self.linformer_seq_len = linformer_seq_len
            self.linformer_k = linformer_k
            self.linformer_core = linformer_core
            self.core = None
            self.core_kind = None
            # Kept out of the module tree (a registered module would join the state_dict and the optimizer): the
            # factory builds the core at the first forward, and ported projections wait here until it exists.
            self.__dict__["_factory"] = factory
            self.__dict__["_pending_proj"] = None
        else:
            # explicit attention core: quantizable scores einsum, softmax, values einsum
            self.scores_einsum = factory.einsum("bhqd,bhkd->bhqk", name=sub("scores"))
            self.attn_softmax = factory.softmax(axis=-1, name=sub("softmax"))
            self.values_einsum = factory.einsum("bhqk,bhkd->bhqd", name=sub("values"))

    def set_backend(self, attn_type: str, **kwargs) -> str:
        """Duck-typed counterpart of Attention.set_backend; the backend is fixed at construction.

        Raises:
            ValueError: If another backend is requested.
        """
        if attn_type != self.attn_type:
            raise ValueError(f"KerasAttention ({self.attn_type}) cannot switch to backend '{attn_type}'")
        return self.attn_type

    def separate_heads(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (self.num_heads, -1)).transpose(-3, -2)  # B S D -> B H S Dh

    def recombine_heads(self, x: Tensor) -> Tensor:
        return x.transpose(-3, -2).flatten(-2)  # B H S Dh -> B S D

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
        score_mod=None,
        initial_values: dict | None = None,
        kv_sort_idx: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Mirror of hepattn Attention.forward for the torch and linformer backends.

        Masks follow the hepattn convention: True = participates in attention. ``kv_sort_idx`` (B, M) is the order in
        which the keys enter the linformer sequence projection; dense attention ignores it, as org. Linformer ignores
        ``q_mask``, as org.

        Raises:
            NotImplementedError: If attn_bias or score_mod are requested.
        """
        if attn_bias is not None:
            raise NotImplementedError("attn_bias is not supported by KerasAttention")
        if score_mod is not None:
            raise NotImplementedError("score_mod is a flex-attention feature, not supported by KerasAttention")

        q_shape = q.shape
        if k is None and v is None:  # self-attention
            k = v = q
            kv_shape = q.shape
        else:  # cross-attention
            assert k is not None, "k must be provided for cross-attention"
            if v is None:
                v = k
            kv_shape = k.shape
            assert k.shape == v.shape, f"Shape mismatch: k.shape={k.shape} vs v.shape={v.shape}"

        # value-residual mix is computed from the raw (pre-projection) queries, as in torch
        mix = None
        if self.value_residual and not self.is_first_layer:
            mix = self.value_residual_mix(q, training=self.training)  # (B, S, H)
            mix = mix.unsqueeze(-1).transpose(-2, -3)  # -> (B, H, S, 1)

        qp = self.q_proj(q, training=self.training)
        kp = self.k_proj(k, training=self.training)
        vp = self.v_proj(v, training=self.training)

        if self.qkv_norm:
            qp = self.q_norm(qp)
            kp = self.k_norm(kp)
            vp = self.v_norm(vp)

        qp = self.separate_heads(qp)
        kp = self.separate_heads(kp)
        vp = self.separate_heads(vp)

        if self.value_residual and initial_values is not None:
            if self.is_first_layer:
                initial_values["v"] = vp
            else:
                assert mix is not None
                vp = vp * mix + initial_values["v"] * (1.0 - mix)

        if self.attn_type == "linformer":
            out = self._linformer(qp, kp, vp, kv_mask, attn_mask, kv_sort_idx)
        else:
            mask = merge_masks(q_mask, kv_mask, attn_mask, q_shape, kv_shape, qp.device)
            if mask is not None and mask.dim() == 3:
                mask = mask.unsqueeze(-3)  # broadcast over heads

            # scale queries before the matmul, matching the order of operations in torch SDPA
            qp = qp * (self.head_dim**-0.5)
            scores = self.scores_einsum([qp, kp], training=self.training)
            attn = apply_softmax(self.attn_softmax, scores, attn_mask=mask, training=self.training)
            out = self.values_einsum([attn, vp], training=self.training)

        out = self.recombine_heads(out)
        return self.out_proj(out, training=self.training)

    # ------------------------------------------------------------------------------------------------ linformer
    def _linformer(self, q: Tensor, k: Tensor, v: Tensor, kv_mask, attn_mask, kv_sort_idx) -> Tensor:
        """Org ``Attention._linformer_attention`` on (B, H, S, Dh) heads, with the keras_native core.

        Raises:
            ValueError: If a call does not match the built core (shapes, or the masks of another org path).
        """
        if kv_sort_idx is not None:  # the projections index key slots, so the keys enter in this order
            k, v = (x.gather(-2, kv_sort_idx[:, None, :, None].expand_as(x)) for x in (k, v))
            kv_mask = None if kv_mask is None else kv_mask.gather(-1, kv_sort_idx)
            attn_mask = None if attn_mask is None else attn_mask.gather(-1, kv_sort_idx[:, None].expand_as(attn_mask))
        if self.core is None:
            kind = self.linformer_core
            if kind == "auto":
                kind = "project" if attn_mask is None else "attend"
            self.build_core(q.shape[-2], k.shape[-2], kind)
        if (q.shape[-2], k.shape[-2]) != (self.core.nq, self.core.n):
            raise ValueError(f"{self.name}: (q_len, kv_len) {(q.shape[-2], k.shape[-2])} != the built {(self.core.nq, self.core.n)}")
        if self.core_kind == "attend":
            if attn_mask is None:
                raise ValueError(f"{self.name}: the attend core needs an attn_mask; set linformer_core='project' for calls without one")
            if kv_mask is not None:
                attn_mask = attn_mask & kv_mask[:, None]
            return self.core(q, k, v, attn_mask, training=self.training)
        if attn_mask is not None:
            raise ValueError(f"{self.name}: the {self.core_kind} core takes no attn_mask; set linformer_core='attend'")
        return self.core(q, k, v, kv_mask, training=self.training)

    def build_core(self, q_len: int, kv_len: int, kind: str) -> None:
        """Create the keras_native core for (q_len, kv_len), its leaves under the factory scopes (as native).

        Raises:
            ValueError: If the core exists, ``kind`` names no core, or the key length exceeds ``linformer_seq_len``.
        """
        if self.core is not None:
            raise ValueError(f"{self.name}: the linformer core is already built")
        if kind not in LINFORMER_CORES[1:]:
            raise ValueError(f"{self.name}: unknown linformer core '{kind}'")
        if kv_len > self.linformer_seq_len:
            raise ValueError(f"{self.name}: key/value length {kv_len} exceeds linformer_seq_len {self.linformer_seq_len}")
        factory = self._factory
        dims = (self.num_heads, self.head_dim, q_len, kv_len, self.linformer_seq_len, self.linformer_k)
        core_name = f"{self.name}_{kind}" if self.name else None
        with factory.scopes():
            if kind == "attend":
                core = AttendCore(factory, *dims, name=core_name)
            else:
                core = ProjectCore(factory, *dims, masked=kind == "project", name=core_name)
        core.build()
        self.core = core
        self.core_kind = kind
        # org project(t, E, None) reads only E[:n], so the rows past n of an unmasked core get no gradient. They are
        # buffers here: in the state_dict with torch's (seq_len, k) layout, and outside the optimizer.
        if core.E.tail is None and kv_len < self.linformer_seq_len:
            bound = 1.0 / math.sqrt(self.linformer_k)  # masked_linformer.init_
            device = core.E.kernel.value.device
            for part in ("k", "v"):
                dead = torch.empty(self.linformer_seq_len - kv_len, self.linformer_k, device=device).uniform_(-bound, bound)
                self.register_buffer(f"proj_{part}_dead", dead)
        pending = self._pending_proj
        if pending is not None:
            self.__dict__["_pending_proj"] = None
            self.load_projections(*pending)

    @torch.no_grad()
    def load_projections(self, proj_k: Tensor, proj_v: Tensor) -> None:
        """Load org's (seq_len, k) ``proj_k`` / ``proj_v``; held until the first forward builds the core.

        Raises:
            ValueError: If a projection is not (linformer_seq_len, linformer_k).
        """
        for proj in (proj_k, proj_v):
            if tuple(proj.shape) != (self.linformer_seq_len, self.linformer_k):
                raise ValueError(f"{self.name}: projection {tuple(proj.shape)} != {(self.linformer_seq_len, self.linformer_k)}")
        if self.core is None:
            self.__dict__["_pending_proj"] = (proj_k.detach().clone(), proj_v.detach().clone())
            return
        for part, proj, seq_proj in (("k", proj_k, self.core.E), ("v", proj_v, self.core.F)):
            seq_proj.assign_full(proj.detach().cpu().numpy())
            if hasattr(self, f"proj_{part}_dead"):
                getattr(self, f"proj_{part}_dead").copy_(proj[self.core.n :])

    def full_projections(self) -> tuple[Tensor, Tensor]:
        """The float (seq_len, k) ``proj_k`` / ``proj_v`` this layer holds, in org's layout.

        Raises:
            RuntimeError: If the core is not built yet.
        """
        if self.core is None:
            raise RuntimeError(f"{self.name}: the linformer core builds at the first forward")
        out = []
        for part, seq_proj in (("k", self.core.E), ("v", self.core.F)):
            rows = [seq_proj.kernel.value]
            if seq_proj.tail is not None:
                rows.append(seq_proj.tail.value)
            elif hasattr(self, f"proj_{part}_dead"):
                rows.append(getattr(self, f"proj_{part}_dead"))
            out.append(torch.cat([row.detach() for row in rows], 0))
        return out[0], out[1]


ATTENTION_KEYS = frozenset({
    "num_heads",
    "bias",
    "torch_compile",
    "window_size",
    "qkv_norm",
    "norm",
    "value_residual",
    "is_first_layer",
    "linformer_seq_len",
    "linformer_k",
    "linformer_core",
})


def build_keras_attention(dim: int, attn_type: str = "torch", *, factory: "LayerFactory | None" = None, name: str | None = None, **attn_kwargs):
    """KerasAttention for torch/flash/flash-varlen (dense-SDPA semantics) and linformer (masked Linformer, org PR #9).

    Only org/main Attention options (plus ``linformer_core``) are accepted. An unknown one raises instead of being
    dropped, so a stale config cannot silently build another model (``linformer_proj_dim`` used to give k=256).

    Raises:
        TypeError: For an attention option that org/main Attention does not have.
    """
    if "linformer_proj_dim" in attn_kwargs:
        raise TypeError("linformer_proj_dim is not an org/main Attention option; the projected length is linformer_k (org PR #9)")
    unknown = sorted(set(attn_kwargs) - ATTENTION_KEYS)
    if unknown:
        raise TypeError(f"unknown attention option(s) {unknown}; accepted: {sorted(ATTENTION_KEYS)}")
    return KerasAttention(dim, attn_type=attn_type, factory=factory, name=name, **attn_kwargs)
