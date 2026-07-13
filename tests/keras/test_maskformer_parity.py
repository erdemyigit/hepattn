"""End-to-end float parity: torch MaskFormer vs KerasMaskFormer on the CLIC test config.

The model pair mirrors tests/experiments/clic/test_clic.yaml (dim 32, 2+2 layers,
150 queries, the four CLIC tasks) and runs on genuine CLIC dummy-data batches.
Mask attention makes the graph self-referential (task outputs become the next
layer's attention mask), so the E2E test also asserts BOOLEAN equality of the
per-layer attention masks — a threshold-flip immune criterion that float
tolerances alone cannot provide.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import assert_parity

import hepattn
from hepattn.experiments.clic.pflow_data import CLICDataset
from hepattn.keras.decoder import KerasMaskFormerDecoderLayer
from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.keras.porting import port_attention, port_dense, port_maskformer, port_residual
from hepattn.models.decoder import MaskFormerDecoder, MaskFormerDecoderLayer
from hepattn.models.dense import Dense
from hepattn.models.encoder import Encoder
from hepattn.models.input import InputNet
from hepattn.models.maskformer import MaskFormer
from hepattn.models.matcher import Matcher
from hepattn.models.posenc import FourierPositionEncoder
from hepattn.models.task import (
    IncidenceBasedRegressionTask,
    IncidenceRegressionTask,
    ObjectClassificationTask,
    ObjectHitMaskTask,
)

DIM = 32
NUM_QUERIES = 150
MAX_NODES = 160
SCALE_DICT = str(Path(hepattn.__file__).parent / "experiments/clic/configs/clic_var_transform.yaml")

ENCODER_CFG = {"num_layers": 2, "hybrid_norm": True, "value_residual": True, "num_register_tokens": 8, "attn_kwargs": {"num_heads": 16}}
DECODER_CFG = {
    "num_decoder_layers": 2,
    "num_queries": NUM_QUERIES,
    "mask_attention": True,
    "use_query_masks": False,
    "decoder_layer_config": {"dim": DIM, "hybrid_norm": True, "attn_kwargs": {"num_heads": 16}},
}


def make_input_nets() -> nn.ModuleList:
    return nn.ModuleList([
        InputNet(
            input_name="node",
            fields=["features"],
            net=Dense(input_size=27, output_size=DIM),
            posenc=FourierPositionEncoder(input_name="node", dim=DIM, fields=["eta", "phi"], scale=0.1),
        )
    ])


def make_tasks() -> nn.ModuleList:
    return nn.ModuleList([
        ObjectClassificationTask(
            name="classification",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            num_classes=5,
            losses={"object_ce": 2},
            costs={"object_ce": 2},
            net=Dense(input_size=DIM, output_size=6, hidden_layers=[256, 128, 32], activation=nn.SiLU()),
            null_weight=0.5,
            class_weights=[1.0, 3.0, 8.0, 1.5, 1.0],
            mask_queries=False,
            has_intermediate_loss=True,
        ),
        ObjectHitMaskTask(
            name="mask",
            input_constituent="node",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            pred_threshold=0.1,
            logit_scale=4,
            losses={"mask_bce": 5.0, "mask_dice": 1.0},
            costs={"mask_dice": 1.0},
            dim=DIM,
            null_weight=1.0,
            has_intermediate_loss=True,
        ),
        IncidenceRegressionTask(
            name="incidence",
            input_constituent="node",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            losses={"kl_div": 1.0},
            costs={"kl_div": 1.0},
            net=Dense(input_size=DIM, hidden_layers=2, activation=nn.SiLU()),
            node_net=Dense(input_size=DIM, hidden_layers=1),
            has_intermediate_loss=False,
        ),
        IncidenceBasedRegressionTask(
            name="regression",
            fields=["e", "pt", "eta", "sinphi", "cosphi"],
            input_constituent="node",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            loss="l1",
            loss_weight=10.0,
            cost_weight=10.0,
            use_incidence=True,
            use_nodes=True,
            cost="new",
            mode="scale",
            scale_dict_path=SCALE_DICT,
            net=Dense(input_size=70, output_size=5, hidden_layers=[512, 256, 128, 64, 32], activation=nn.SiLU()),
            has_intermediate_loss=False,
        ),
    ])


def make_matcher() -> Matcher:
    return Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)


def make_torch_model(seed: int = 0) -> MaskFormer:
    torch.manual_seed(seed)
    return MaskFormer(
        input_nets=make_input_nets(),
        encoder=Encoder(
            num_layers=ENCODER_CFG["num_layers"], dim=DIM, attn_type="torch", **{k: v for k, v in ENCODER_CFG.items() if k != "num_layers"}
        ),
        decoder=MaskFormerDecoder(**DECODER_CFG),
        tasks=make_tasks(),
        dim=DIM,
        matcher=make_matcher(),
    ).eval()


def make_keras_model(seed: int = 0, quant: dict | None = None) -> KerasMaskFormer:
    torch.manual_seed(seed)
    return KerasMaskFormer(
        input_nets=make_input_nets(),
        encoder=ENCODER_CFG,
        decoder=DECODER_CFG,
        tasks=make_tasks(),
        dim=DIM,
        matcher=make_matcher(),
        quant=quant,
    ).eval()


def make_ported_pair(seed: int = 0) -> tuple[MaskFormer, KerasMaskFormer]:
    tmodel = make_torch_model(seed)
    kmodel = make_keras_model(seed)
    port_maskformer(tmodel, kmodel)
    return tmodel, kmodel


def clic_dummy_batch(num_events: int = 2) -> tuple[dict, dict]:
    ds = CLICDataset(
        filepath="dummy",
        inputs={"node": ["features"]},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=SCALE_DICT,
        num_events=num_events,
        num_objects=NUM_QUERIES,
        max_nodes=MAX_NODES,
        dummy_data=True,
    )
    events = [ds[i] for i in range(num_events)]
    inputs = {k: torch.stack([e[0][k] for e in events]) for k in events[0][0]}
    targets = {k: torch.stack([e[1][k] for e in events]) for k in events[0][1]}
    return inputs, targets


def flatten_outputs(outputs: dict, prefix: str = "") -> dict[str, torch.Tensor]:
    flat = {}
    for key, value in outputs.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_outputs(value, prefix=f"{path}."))
        elif isinstance(value, torch.Tensor):
            flat[path] = value
    return flat


def assert_tensor_tree_parity(tag: str, tree_t: dict, tree_k: dict, atol: float, rtol: float) -> None:
    flat_t = flatten_outputs(tree_t)
    flat_k = flatten_outputs(tree_k)
    assert set(flat_t) == set(flat_k), f"[{tag}] output keys differ: {set(flat_t) ^ set(flat_k)}"
    for key in sorted(flat_t):
        t, k = flat_t[key], flat_k[key]
        assert t.shape == k.shape, f"[{tag}:{key}] shape mismatch {t.shape} vs {k.shape}"
        if t.dtype == torch.bool:
            assert torch.equal(t, k), f"[{tag}:{key}] boolean outputs differ"
            continue
        nan_t, nan_k = torch.isnan(t), torch.isnan(k)
        assert torch.equal(nan_t, nan_k), f"[{tag}:{key}] NaN patterns differ"
        finite = ~nan_t
        assert_parity(tag, key, t[finite], k[finite], atol=atol, rtol=rtol)


def test_decoder_layer_parity():
    """Bidirectional decoder layer: both updated queries AND keys must match.

    A missing kv update or missing attn-mask transpose diverges O(1) on kv while q still matches.
    """
    torch.manual_seed(20)
    tlayer = MaskFormerDecoderLayer(**DECODER_CFG["decoder_layer_config"]).eval()
    klayer = KerasMaskFormerDecoderLayer(**DECODER_CFG["decoder_layer_config"]).eval()
    port_residual(tlayer.q_ca, klayer.q_ca, port_attention)
    port_residual(tlayer.q_sa, klayer.q_sa, port_attention)
    port_residual(tlayer.q_dense, klayer.q_dense, port_dense)
    port_residual(tlayer.kv_ca, klayer.kv_ca, port_attention)
    port_residual(tlayer.kv_dense, klayer.kv_dense, port_dense)

    gen = torch.Generator().manual_seed(21)
    q = torch.randn(2, NUM_QUERIES, DIM, generator=gen)
    kv = torch.randn(2, 40, DIM, generator=gen)
    attn_mask = torch.rand(2, NUM_QUERIES, 40, generator=gen) > 0.5
    attn_mask[..., 0] = True  # decoder applies unmask_all_false upstream; mimic its postcondition
    kv_mask = torch.ones(2, 40, dtype=torch.bool)
    kv_mask[:, 35:] = False

    with torch.no_grad():
        q_t, kv_t = tlayer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)
    q_k, kv_k = klayer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

    assert_parity("decoder_layer.q", "dim32h16_bidir", q_t, q_k, atol=1e-6, rtol=1e-5)
    assert_parity("decoder_layer.kv", "dim32h16_bidir", kv_t, kv_k, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


def test_maskformer_e2e_forward_parity():
    tmodel, kmodel = make_ported_pair(seed=30)
    inputs, _ = clic_dummy_batch()

    with torch.no_grad():
        out_t = tmodel(inputs)
        out_k = kmodel(inputs)

    # boolean attn-mask equality per decoder layer: immune to threshold-flip amplification
    for layer in ("layer_0", "layer_1"):
        assert torch.equal(out_t[layer]["attn_mask"], out_k[layer]["attn_mask"]), f"{layer} attention masks differ"

    assert_tensor_tree_parity("maskformer.e2e", out_t, out_k, atol=5e-5, rtol=1e-4)


def test_maskformer_loss_and_predict_parity():
    tmodel, kmodel = make_ported_pair(seed=31)
    inputs, targets = clic_dummy_batch()

    with torch.no_grad():
        out_t = tmodel(inputs)
        out_k = kmodel(inputs)
        _, _, losses_t = tmodel.loss(out_t, dict(targets))
        _, _, losses_k = kmodel.loss(out_k, dict(targets))
        preds_t = tmodel.predict(out_t)
        preds_k = kmodel.predict(out_k)

    flat_lt = flatten_outputs(losses_t)
    flat_lk = flatten_outputs(losses_k)
    assert set(flat_lt) == set(flat_lk), f"loss keys differ: {set(flat_lt) ^ set(flat_lk)}"
    assert flat_lt, "loss dict is empty — vacuous comparison"
    for key in sorted(flat_lt):
        loss_t, loss_k = flat_lt[key], flat_lk[key]
        if not bool(torch.isfinite(loss_t).all()):
            # e.g. mask_bce = inf on random-init dummy data (finfo.min-padded logits in BCE);
            # the torch reference itself is non-finite here and keras must match it exactly
            assert torch.equal(loss_t, loss_k), f"non-finite loss '{key}' differs: {loss_t} vs {loss_k}"
        else:
            assert_parity("maskformer.loss", key, loss_t, loss_k, atol=1e-4, rtol=1e-3)

    # thresholded boolean predictions must agree exactly
    flat_pt = flatten_outputs(preds_t)
    flat_pk = flatten_outputs(preds_k)
    assert set(flat_pt) == set(flat_pk)
    bool_keys = [k for k in flat_pt if flat_pt[k].dtype == torch.bool]
    assert bool_keys, "no boolean predictions found — vacuous comparison"
    for key in bool_keys:
        assert torch.equal(flat_pt[key], flat_pk[key]), f"boolean prediction '{key}' differs"


def test_keras_state_dict_roundtrip():
    """Port -> save -> fresh build -> load -> bit-identical outputs.

    Catches keras variables that escape torch state_dict tracking (lazy build) —
    the failure mode that would silently break Lightning checkpointing.
    """
    _, kmodel = make_ported_pair(seed=32)
    inputs, _ = clic_dummy_batch()

    with torch.no_grad():
        out_before = kmodel(inputs)

    state = kmodel.state_dict()
    assert state, "state_dict is empty"
    kmodel_2 = make_keras_model(seed=99)  # different init on purpose
    kmodel_2.load_state_dict(state)

    with torch.no_grad():
        out_after = kmodel_2(inputs)

    flat_a = flatten_outputs(out_before)
    flat_b = flatten_outputs(out_after)
    assert set(flat_a) == set(flat_b)
    for key in sorted(flat_a):
        a, b = flat_a[key], flat_b[key]
        if a.dtype == torch.bool:
            assert torch.equal(a, b), f"{key} differs after roundtrip"
        else:
            nan_a = torch.isnan(a)
            assert torch.equal(nan_a, torch.isnan(b)), f"{key} NaN pattern differs after roundtrip"
            assert torch.equal(a[~nan_a], b[~nan_a]), f"{key} differs after roundtrip"


def test_incidence_softmax_axis_probe():
    """The incidence task softmax runs over the QUERY axis (dim=1), not the node axis.

    On a non-square (num_queries != num_nodes) fixture the incidence output must sum
    to 1 across queries. A wrong axis port passes on square inputs but fails here.
    """
    _, kmodel = make_ported_pair(seed=33)
    inputs, _ = clic_dummy_batch()
    with torch.no_grad():
        out_k = kmodel(inputs)
    inc = out_k["final"]["incidence"]["pflow_incidence"]
    assert inc.shape[1] == NUM_QUERIES and inc.shape[2] == MAX_NODES
    node_valid = inputs["node_valid"]
    sums = inc.sum(dim=1)[node_valid]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "incidence does not sum to 1 over the query axis"
