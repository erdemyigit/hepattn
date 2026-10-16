"""Input sorting in the hybrid (org PR #9, ``input_sort_field: phi``; docs/hgq/RETARGET_ORG_MAIN.md sections 1.2 and 5.5).

Dense attention is permutation-equivariant, so a full-model comparison cannot see the sort (the no-sort control
of test_org_model_parity.py measures 0 difference). These tests check the mechanics directly:
- the encoder round trip unsort(sort(x)) is exact, on a fixture whose sort is not the identity;
- with a stub that marks the sequence position, the hybrid encoder equals the torch encoder only when both sort;
- the model hands the torch sort index to the decoder's query cross-attention only, as org does.
"""

import copy

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import org_model_utils as U  # ty: ignore [unresolved-import]

from hepattn.keras.encoder import KerasEncoder
from hepattn.keras.porting import port_encoder
from hepattn.models.encoder import Encoder, sort_index

DIM = 32
ENCODER = {
    "num_layers": 2,
    "dim": DIM,
    "norm": "FastLayerNorm",
    "hybrid_norm": True,
    "dense_norm_placement": "legacy",
    "value_residual": True,
    "num_register_tokens": 4,
    "attn_kwargs": {"num_heads": 4},
    "dense_kwargs": {"activation": "SwiGLU"},
}
# Measured (R4a, macOS arm64, torch 2.10.0, CPU): max |torch - hybrid| on valid slots 7.15e-7; 2x.
STUB_TOL = 1.43e-6


def fixture(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(3, 40, DIM, generator=g)
    phi = torch.randn(3, 40, generator=g)
    valid = torch.arange(40) < torch.tensor([[33], [21], [39]])
    return x, phi, valid


def mark_positions(_module, args, kwargs):
    """Forward pre-hook: add a bias that depends on the token's position in the sequence the layer sees."""
    x = args[0]
    pos = torch.arange(x.shape[-2], dtype=x.dtype).unsqueeze(-1) / x.shape[-2]
    return (x + torch.sin(7 * pos + torch.arange(x.shape[-1], dtype=x.dtype)),), kwargs


def test_sort_fixture_is_not_the_identity():
    _, phi, valid = fixture()
    idx = sort_index(phi, valid)
    assert not torch.equal(idx, torch.arange(40).expand_as(idx)), "vacuous: the sort keeps the order"
    for b in range(3):  # padding goes last
        assert valid[b, idx[b]].int().diff().le(0).all()


def test_encoder_sort_round_trip_is_exact():
    """No layers: the encoder output is the input after sort, register add/strip and unsort, bit for bit."""
    torch.manual_seed(0)
    kenc = KerasEncoder(**{**ENCODER, "num_layers": 0})
    x, phi, valid = fixture()
    assert torch.equal(kenc(x, x_sort_value=phi, kv_mask=valid), x)


def stub_outputs(sort_hybrid: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    tenc = Encoder(attn_type="torch", **ENCODER).eval()
    kenc = KerasEncoder(**ENCODER).eval()
    port_encoder(tenc, kenc)
    tenc.layers[0].register_forward_pre_hook(mark_positions, with_kwargs=True)
    kenc.layers[0].register_forward_pre_hook(mark_positions, with_kwargs=True)
    x, phi, valid = fixture()
    with torch.no_grad():
        y_t = tenc(x, x_sort_value=phi, kv_mask=valid)
        y_k = kenc(x, x_sort_value=phi if sort_hybrid else None, kv_mask=valid)
    return y_t, y_k, valid


def test_encoder_sort_matches_torch_with_position_stub():
    y_t, y_k, valid = stub_outputs()
    diff = (y_t - y_k)[valid].abs().max().item()
    assert diff <= STUB_TOL, diff


def test_control_unsorted_hybrid_differs_with_position_stub():
    y_t, y_k, valid = stub_outputs(sort_hybrid=False)
    diff = (y_t - y_k)[valid].abs().max().item()
    assert diff >= 10 * STUB_TOL, diff  # measured 2.54 (R4a)


def attention_kwargs(model, inputs) -> dict[str, list]:
    """kv_sort_idx as each decoder attention receives it, per layer."""
    seen: dict[str, list] = {"q_ca": [], "q_sa": [], "kv_ca": []}
    hooks = []
    for layer in model.decoder.decoder_layers:
        for part, record in seen.items():

            def hook(_module, _args, kwargs, record=record):
                record.append(kwargs.get("kv_sort_idx"))

            hooks.append(getattr(layer, part).fn.register_forward_pre_hook(hook, with_kwargs=True))
    with torch.no_grad():
        model(inputs)
    for h in hooks:
        h.remove()
    return seen


@pytest.fixture(scope="module")
def dense_configs(tmp_path_factory):
    gen = U.to_dense_attention(U.generate_polaris_config(tmp_path_factory.mktemp("gen") / "polaris_hgq.yaml"))
    org = U.torch_reference_config(U.to_dense_attention(U.merge_org_configs()))
    return gen, org


def test_model_passes_the_sort_index_to_query_cross_attention_only(dense_configs):
    gen, org = dense_configs
    inputs, _ = U.clic_dummy_batch()
    expected = sort_index(inputs["node_phi"], inputs["node_valid"])
    seen_k = attention_kwargs(U.instantiate(gen, quantized=False), inputs)
    seen_t = attention_kwargs(U.instantiate(org), inputs)
    assert len(seen_k["q_ca"]) == 4
    for part in ("q_ca", "q_sa", "kv_ca"):
        for got_k, got_t in zip(seen_k[part], seen_t[part], strict=True):
            if part == "q_ca":
                assert torch.equal(got_k, expected) and torch.equal(got_t, expected)
            else:
                assert got_k is None and got_t is None


def test_control_no_sort_field_passes_no_sort_index(dense_configs):
    gen, _ = dense_configs
    gen = copy.deepcopy(gen)
    gen["model"]["model"]["init_args"].pop("input_sort_field")
    inputs, _ = U.clic_dummy_batch()
    seen = attention_kwargs(U.instantiate(gen, quantized=False), inputs)
    assert seen["q_ca"] and all(idx is None for idx in seen["q_ca"])
