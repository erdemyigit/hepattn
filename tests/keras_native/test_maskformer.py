"""S3 float parity and behaviour of the native decoder / tasks / MaskFormer against the pure-torch hepattn model.

The pure-torch model runs in this process (only the keras backend is process-global). Configs: ``golden/mf_cases.py``
(the CLIC test config of ``tests/keras/test_maskformer_parity.py``; ``torch`` = dense attention, ``lin`` = Linformer in
encoder and decoder as ``pflow_hgq.yaml``), on genuine CLIC dummy events. Tolerances: ``maskformer.e2e`` 5e-5 / 1e-4
(PARITY.md), decoder layer 1e-6 / 1e-5 (``tests/keras/test_maskformer_parity.py``), and boolean attention masks
EXACTLY equal per layer.
"""

import copy
import os
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest
import torch
import yaml

from hepattn.keras_native.decoder import MaskFormerDecoder as NDecoder
from hepattn.keras_native.decoder import MaskFormerDecoderLayer as NDecoderLayer
from hepattn.keras_native.maskformer import MaskFormer as NMaskFormer
from hepattn.keras_native.maskformer import from_config
from hepattn.keras_native.tasks import IncidenceBasedRegressionTask, ObjectClassificationTask, ObjectHitMaskTask, topk_mask
from hepattn.models.decoder import MaskFormerDecoderLayer
from hepattn.models.task import ObjectHitMaskTask as TorchObjectHitMaskTask
from hepattn.utils.masks import topk_attn

from .golden.mf_cases import DIM, NUM_QUERIES, VARIANTS, clic_dummy_batch, native_kwargs, torch_model
from .mf_golden_utils import build_mf, flatten
from .parity_utils import assert_parity
from .porting import perturb_affine_params, port_any_attention, port_dense, port_maskformer, port_residual

# R0 xfail (docs/hgq/RETARGET_ORG_MAIN.md section 2, item 4); strict, so the marker fails once the named lane fixes the test.
R3_XFAIL = pytest.mark.xfail(
    strict=True, reason="R3 (after the R2 merge): the torch reference passes linformer_proj_dim, which org Attention rejects since PR #9"
)
R3_SWIGLU_XFAIL = pytest.mark.xfail(strict=True, reason="R3: quantized SwiGLU raises NotImplementedError (RETARGET_ORG_MAIN.md section 5.4)")

REPO = Path(__file__).resolve().parents[2]
E2E = {"atol": 5e-5, "rtol": 1e-4}


@pytest.fixture(scope="module")
def batch():
    inputs, _ = clic_dummy_batch()
    return inputs


def tin(inputs: dict) -> dict:
    return {k: torch.from_numpy(v) for k, v in inputs.items()}


def ported_pair(variant: str, inputs: dict, seed: int = 30, **overrides):
    tmodel = torch_model(variant, seed=seed, **overrides)
    nmodel = build_mf(variant, None, inputs, **overrides)
    port_maskformer(tmodel, nmodel)
    return tmodel, nmodel


def run_pair(tmodel, nmodel, inputs):
    with torch.no_grad():
        out_t = {k: v.numpy() for k, v in flatten_torch(tmodel(tin(inputs))).items()}
    out_n = flatten(nmodel(inputs, training=False))
    return out_t, out_n


def flatten_torch(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(flatten_torch(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def assert_tree_parity(tag: str, out_t: dict, out_n: dict) -> dict:
    """Keys equal; booleans exactly equal; NaN patterns equal; floats at the e2e tolerance. Returns the boolean leaves."""
    assert set(out_t) == set(out_n), f"[{tag}] output keys differ: {set(out_t) ^ set(out_n)}"
    bools = {}
    for key in sorted(out_t):
        t, n = out_t[key], np.asarray(out_n[key])
        assert t.shape == n.shape, f"[{tag}:{key}] shape {t.shape} vs {n.shape}"
        if t.dtype == bool:
            assert n.dtype == bool and np.array_equal(t, n), f"[{tag}:{key}] boolean outputs differ in {int((t != n).sum())} elements"
            bools[key] = t
            continue
        assert np.array_equal(np.isnan(t), np.isnan(n)), f"[{tag}:{key}] NaN patterns differ"
        finite = ~np.isnan(t)
        assert_parity(tag, key, t[finite], n[finite], **E2E)
    return bools


# ------------------------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("variant", [pytest.param(v, marks=R3_XFAIL) if v == "lin" else v for v in VARIANTS])
def test_maskformer_e2e_float_parity(variant, batch):
    """maskformer.e2e rows at 5e-5 / 1e-4 and per-layer attention masks exactly equal (incl. key sets)."""
    out_t, out_n = run_pair(*ported_pair(variant, batch), batch)
    expected = {"layer_0.mask.pflow_node_logit", "layer_0.attn_mask", "layer_1.classification.pflow_logit", "layer_1.mask.pflow_node_logit"}
    assert expected <= set(out_t), "classification must skip layer 0 and the mask task must run at every layer"
    assert not any(k.startswith("layer_0.classification") for k in out_t)
    bools = assert_tree_parity(f"maskformer.e2e.{variant}", out_t, out_n)
    for key, mask in bools.items():  # non-vacuous: the masks restrict attention, and not to nothing
        assert 0.05 < mask.mean() < 0.95, f"{key}: attention mask fraction {mask.mean():.3f}"


@R3_XFAIL
def test_mask_logit_sentinel_and_incidence_axis(batch):
    """Padded nodes carry the finfo.min sentinel (as torch); incidence sums to 1 over the QUERY axis on valid nodes."""
    _, nmodel = ported_pair("lin", batch)
    out = flatten(nmodel(batch, training=False))
    valid = batch["node_valid"]
    logit = out["final.mask.pflow_node_logit"]
    assert (logit[np.broadcast_to(~valid[:, None, :], logit.shape)] == np.finfo(np.float32).min).all()
    assert (logit[np.broadcast_to(valid[:, None, :], logit.shape)] > -1e30).all()
    inc = out["final.incidence.pflow_incidence"]
    assert inc.shape == (2, NUM_QUERIES, 160)
    np.testing.assert_allclose(inc.sum(axis=1)[valid], 1.0, atol=1e-5)
    assert (inc.sum(axis=1)[~valid] == 0).all()


@pytest.mark.parametrize("threshold", [None, 0.5])
def test_mask_attention_threshold_is_inclusive_as_torch(threshold):
    """``sigmoid(logit) == threshold`` keeps the node, as torch's ``>=`` (``hepattn/models/task.py:571``).

    sigmoid(0) is exactly 0.5 in both frameworks, so a ``>`` port differs at logit 0. The e2e and golden tests
    cannot see this: their logits never sit exactly on the threshold. None uses the task's threshold (0.5 here).
    """
    finfo_min = np.finfo(np.float32).min
    logits = np.array([[[0.0, -1.0, 1.0, finfo_min, 0.0, -1e-7]]], dtype=np.float32)
    kwargs = {
        "name": "mask",
        "input_constituent": "node",
        "input_object": "query",
        "output_object": "pflow",
        "target_object": "particle",
        "losses": {"mask_bce": 1.0},
        "costs": {"mask_dice": 1.0},
        "dim": 8,
        "pred_threshold": 0.5,
    }
    key = "pflow_node_logit"
    ref = TorchObjectHitMaskTask(**kwargs).attn_mask({key: torch.from_numpy(logits)}, threshold=threshold)["node"].numpy()
    got = np.asarray(ObjectHitMaskTask(**kwargs).attn_mask({key: logits}, threshold=threshold)["node"])
    assert ref[0, 0, 0] and ref[0, 0, 4], "control: torch keeps a node whose sigmoid equals the threshold"
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("variant", [pytest.param(v, marks=R3_XFAIL) if v == "lin" else v for v in VARIANTS])
def test_decoder_layer_parity(variant):
    """Bidirectional decoder layer: updated queries AND keys (valid keys) at 1e-6 / 1e-5, with a padded key mask."""
    cfg = {"dim": DIM, "hybrid_norm": True, "depth": 1, "attn_kwargs": copy.deepcopy(VARIANTS[variant]["decoder_attn"])}
    torch.manual_seed(20)
    tlayer = MaskFormerDecoderLayer(**cfg).eval()
    perturb_affine_params(tlayer, 20)
    nlayer = NDecoderLayer(**{**cfg, "attn_kwargs": copy.deepcopy(cfg["attn_kwargs"])}, name="decoder_l1")
    rng = np.random.default_rng(21)
    q = rng.standard_normal((2, NUM_QUERIES, DIM)).astype(np.float32)
    kv = rng.standard_normal((2, 160, DIM)).astype(np.float32)
    attn_mask = rng.random((2, NUM_QUERIES, 160)) > 0.5
    attn_mask[..., 0] = True  # the decoder restores all-False rows upstream
    kv_mask = np.ones((2, 160), dtype=bool)
    kv_mask[:, 140:] = False
    nlayer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)  # build
    for part in ("q_ca", "q_sa", "kv_ca"):
        port_residual(getattr(tlayer, part), getattr(nlayer, part), port_any_attention)
    for part in ("q_dense", "kv_dense"):
        port_residual(getattr(tlayer, part), getattr(nlayer, part), port_dense)
    with torch.no_grad():
        q_t, kv_t = tlayer(torch.from_numpy(q), torch.from_numpy(kv), attn_mask=torch.from_numpy(attn_mask), kv_mask=torch.from_numpy(kv_mask))
    q_n, kv_n = nlayer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)
    assert_parity(f"decoder_layer.q.{variant}", "dim32h16_bidir", q_t, q_n, atol=1e-6, rtol=1e-5)
    assert_parity(f"decoder_layer.kv.{variant}", "dim32h16_bidir", kv_t, kv_n, valid_mask=kv_mask, atol=1e-6, rtol=1e-5)


# ------------------------------------------------------------------------------------- all-False rows
def raw_layer_masks(out: dict, threshold: float) -> dict:
    """The task masks before the restore, recomputed from the recorded mask logits (sigmoid >= threshold)."""
    return {layer: 1 / (1 + np.exp(-out[f"{layer}.mask.pflow_node_logit"].astype(np.float64))) >= threshold for layer in ("layer_0", "layer_1")}


@pytest.mark.parametrize("variant", [pytest.param(v, marks=R3_XFAIL) if v == "lin" else v for v in VARIANTS])
def test_all_false_rows_attend_everywhere(variant, batch):
    """A query whose mask row is all False attends everywhere; turning the restore off changes the output.

    The mask threshold is raised to 0.9 so that a fraction of the rows is all False (counted, must be > 0 and < all).
    Both settings are checked against torch (masks exactly equal, values at the e2e tolerance), and the native mask
    rows are checked directly: restored rows all True, other rows the thresholded logits. Control: restore off.
    """
    threshold = 0.9
    overrides = {"task_overrides": {"mask": {"pred_threshold": threshold}}}
    outs = {}
    for restore in (True, False):
        tmodel, nmodel = ported_pair(variant, batch, decoder_overrides={"unmask_all_false": restore}, **overrides)
        out_t, out_n = run_pair(tmodel, nmodel, batch)
        assert_tree_parity(f"maskformer.all_false.{variant}.restore_{restore}", out_t, out_n)
        outs[restore] = out_n

    for layer, raw in raw_layer_masks(outs[True], threshold).items():
        all_false = ~raw.any(-1)
        n_rows = int(all_false.sum())
        assert 0 < n_rows < all_false.size, f"{layer}: {n_rows} all-False rows; the test needs some, not all"
        recorded = outs[True][f"{layer}.attn_mask"]
        assert recorded[all_false].all(), f"{layer}: restored rows must attend everywhere"
        assert np.array_equal(recorded[~all_false], raw[~all_false]), f"{layer}: rows with a True entry must be kept"
        off = outs[False][f"{layer}.attn_mask"]
        raw_off = raw_layer_masks(outs[False], threshold)[layer]
        assert not off[~raw_off.any(-1)].any(), f"{layer}: with the restore off, all-False rows stay all False"
    # the restore changes what the queries see: layer-1 and final outputs move by >= 10x the e2e parity tolerance
    for key in ("layer_1.mask.pflow_node_logit", "final.mask.pflow_node_logit", "final.classification.pflow_logit"):
        on, off = outs[True][key].astype(np.float64), outs[False][key].astype(np.float64)
        keep = np.abs(on) < 1e30  # not the finfo.min padding
        diff = np.abs(on - off)[keep]
        tol = E2E["atol"] + E2E["rtol"] * np.abs(on[keep]).max()
        assert diff.max() > 10 * tol, f"{key}: unmask_all_false changes it by {diff.max():.2e}, not above 10x the tolerance {tol:.2e}"


# ---------------------------------------------------------------------------------------- task pieces
def test_topk_mask_equals_torch_topk_attn_without_ties():
    rng = np.random.default_rng(3)
    scores = rng.random((3, 150, 160)).astype(np.float32)  # continuous: no ties
    for axis, k in ((-2, 2), (-1, 3), (1, 1)):
        ref = topk_attn(torch.from_numpy(scores), k, dim=axis).numpy()
        got = keras.ops.convert_to_numpy(topk_mask(scores, k, axis))
        assert np.array_equal(ref, got), (axis, k)
        assert (got.sum(axis=axis) == k).all()


def mixed_regression_inputs(batch: dict, seed: int = 11) -> tuple[dict, np.ndarray, np.ndarray]:
    """Regression-task inputs that exercise every per-object branch of ``get_proxy_feats`` in EVERY event.

    The e2e and golden cases do not: their class argmax is all-charged (torch variant) or all-neutral (lin variant).
    Here the class argmax cycles through 0..5 over the queries (charged 0-2 next to neutral 3-5 in one event, and
    class 3 at the ``< 3`` edge). The incidence is continuous (no top-k ties) and normalised over the query axis on
    valid nodes, so the ~60 track nodes per event have one max particle each and the other particles take the
    top-2 (zero-track) branch.
    """
    rng = np.random.default_rng(seed)
    b, n = batch["node_valid"].shape
    x = {
        "inputs": batch,
        "query_embed": rng.standard_normal((b, NUM_QUERIES, DIM)).astype(np.float32),
        "node_embed": rng.standard_normal((b, n, DIM)).astype(np.float32),
        "node_valid": batch["node_valid"],
    }
    inc = rng.uniform(0.05, 1.0, (b, NUM_QUERIES, n))
    inc = (inc / inc.sum(1, keepdims=True) * batch["node_valid"][:, None, :]).astype(np.float32)
    probs = rng.uniform(0.0, 0.5, (b, NUM_QUERIES, 6))
    probs[:, np.arange(NUM_QUERIES), np.arange(NUM_QUERIES) % 6] += 1.0
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    return x, inc, probs


@R3_XFAIL
def test_regression_task_parity_on_mixed_charged_and_neutral_objects(batch):
    """IncidenceBasedRegressionTask vs torch at the e2e tolerance, with charged and neutral objects in the same event
    and both charged proxy branches (max track, top-2 tracks) present; the branch counts are asserted, not assumed.

    The task does not depend on the encoder / decoder attention variant, so one variant suffices.
    """
    tmodel, nmodel = ported_pair("lin", batch)
    treg, nreg = tmodel.tasks[3], nmodel.tasks[3]
    assert isinstance(nreg, IncidenceBasedRegressionTask)
    x, inc, probs = mixed_regression_inputs(batch)

    # coverage, from the inputs alone (numpy, independent of both implementations)
    is_charged = probs.argmax(-1) < 3
    assert (probs.argmax(-1) == 3).any()
    for event in range(is_charged.shape[0]):
        assert 0.3 < is_charged[event].mean() < 0.7, f"event {event}: charged fraction {is_charged[event].mean():.2f}"
    charged_inc = inc * batch["node_is_track"][:, None, :]
    is_max = (charged_inc == charged_inc.max(1, keepdims=True)) & (charged_inc > 0)
    zero_track = ~is_max.any(-1)
    rank = np.argsort(np.argsort(-charged_inc, axis=1), axis=1)  # continuous values: no ties among positive entries
    top2_nonempty = ((rank < 2) & (charged_inc > 0)).any(-1)  # a zero-track particle that still gets a proxy track
    for event in range(is_charged.shape[0]):
        n_max = int((is_charged[event] & ~zero_track[event]).sum())
        n_top2 = int((is_charged[event] & zero_track[event] & top2_nonempty[event]).sum())
        assert n_max >= 10 and n_top2 >= 10, f"event {event}: charged objects on the max branch {n_max}, top-2 branch {n_top2}"

    outputs = {"incidence": {"pflow_incidence": inc}, "classification": {"pflow_class_prob": probs}}
    out_n = {k: np.asarray(keras.ops.convert_to_numpy(v)) for k, v in nreg(x, outputs=outputs).items()}
    tx = {**{k: torch.from_numpy(v) for k, v in x.items() if k != "inputs"}, "inputs": tin(batch)}
    toutputs = {task: {key: torch.from_numpy(v) for key, v in d.items()} for task, d in outputs.items()}
    with torch.no_grad():
        out_t = {k: v.numpy() for k, v in treg(tx, toutputs).items()}
    assert set(out_t) == set(out_n) == {"pflow_regr", "pflow_proxy_regr"}
    proxy = out_t["pflow_proxy_regr"]
    assert np.isfinite(proxy).all()
    # non-vacuous: eta is nonzero wherever a proxy has a constituent (charged particles with no track in their
    # max / top-2 set get a constant proxy, as in torch)
    eta_nonzero = proxy[..., 2] != 0
    assert np.array_equal(eta_nonzero[is_charged], (~zero_track | top2_nonempty)[is_charged])
    assert eta_nonzero[~is_charged].all()
    for key in sorted(out_t):
        assert_parity("regression.mixed", key, out_t[key], out_n[key], **E2E)


@R3_XFAIL
def test_regression_proxy_uses_detached_incidence_and_class_probs(batch):
    """Gradients reach the regression net only through its input embedding, never through the incidence.

    Only the incidence assertion discriminates (mutation T4, stop_gradient removed, fails it). The class probs enter
    only through argmax, whose gradient is 0 with or without stop_gradient, so their assertion is structural.
    """
    import jax  # noqa: PLC0415

    _, nmodel = ported_pair("lin", batch)
    reg = nmodel.tasks[3]
    assert isinstance(reg, IncidenceBasedRegressionTask)
    rng = np.random.default_rng(5)
    x = {
        "inputs": batch,
        "query_embed": rng.standard_normal((2, NUM_QUERIES, DIM)).astype(np.float32),
        "node_embed": rng.standard_normal((2, 160, DIM)).astype(np.float32),
        "node_valid": batch["node_valid"],
    }
    inc = rng.random((2, NUM_QUERIES, 160)).astype(np.float32)
    probs = rng.random((2, NUM_QUERIES, 6)).astype(np.float32)

    def f(inc, probs):
        out = reg(x, outputs={"incidence": {"pflow_incidence": inc}, "classification": {"pflow_class_prob": probs}})
        return keras.ops.sum(out["pflow_regr"])

    g_inc, g_probs = jax.grad(f, argnums=(0, 1))(inc, probs)
    assert not np.any(np.asarray(g_inc)) and not np.any(np.asarray(g_probs))
    assert np.isfinite(float(f(inc, probs)))


# -------------------------------------------------------------------------------------- unsupported
def kwargs_with(**changes):
    kw = native_kwargs("lin")
    for path, value in changes.items():
        node = kw
        *parents, leaf = path.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return kw


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"decoder__dynamic_queries": True}, NotImplementedError),
        ({"decoder__decoder_layer_config__cross_attn_mode": "kmeans"}, NotImplementedError),
        ({"decoder__local_strided_attn": True}, NotImplementedError),
        ({"decoder__fast_local_ca": True}, NotImplementedError),
        ({"decoder__posenc": {"alpha": 1.0, "base": 2.0}}, NotImplementedError),
        ({"decoder__use_query_masks": True}, NotImplementedError),
        ({"decoder__decoder_layer_config__attn_kwargs": {"num_heads": 16, "attn_type": "flex"}}, ValueError),
        ({"encoder__attn_type": "flex"}, ValueError),
        ({"unified_decoding": True}, NotImplementedError),
        ({"encoder_tasks": [{"class_path": "x"}]}, NotImplementedError),
        ({"input_sort_field": "phi"}, NotImplementedError),
        ({"sorter": {"class_path": "hepattn.utils.sorter.Sorter"}}, NotImplementedError),
    ],
    ids=lambda v: "-".join(v) if isinstance(v, dict) else v.__name__,
)
def test_unsupported_configs_raise(changes, error):
    with pytest.raises(error):
        NMaskFormer(**kwargs_with(**changes))


def test_unsupported_task_options_raise():
    with pytest.raises(NotImplementedError, match="mask_queries"):
        ObjectClassificationTask("c", "query", "pflow", "particle", {}, {}, dim=DIM, num_classes=5, mask_queries=True)
    kw = native_kwargs("lin")
    kw["tasks"].append({"class_path": "hepattn.models.task.ObjectRegressionTask", "init_args": {}})
    with pytest.raises(NotImplementedError, match="not ported"):
        NMaskFormer(**kw)
    kw = native_kwargs("lin")
    kw["input_nets"].append(copy.deepcopy(kw["input_nets"][0]))
    with pytest.raises(NotImplementedError, match="input types"):
        NMaskFormer(**kw)


def clic_yaml_model_section(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["model"]


def test_mtl_raises():
    section = clic_yaml_model_section(REPO / "src/hepattn/experiments/clic/configs/pflow_hgq.yaml")
    section["mtl"] = True
    with pytest.raises(NotImplementedError, match="mtl"):
        from_config(section)


def check_clic_structure(model: NMaskFormer, quant_beta0: float) -> None:
    assert [type(t) for t in model.tasks][:2] == [ObjectClassificationTask, ObjectHitMaskTask]
    assert [t.task_name for t in model.tasks] == ["classification", "mask", "incidence", "regression"]
    assert [t.name for t in model.tasks] == ["task0", "task1", "task2", "task3"]
    assert len(model.encoder.layers) == 6 and len(model.decoder.layers) == 4
    assert model.encoder.attn_type == "linformer" and model.decoder.layers[0].attn_type == "linformer"
    assert model.decoder.num_queries == 150 and model.factory.quantize
    assert model.factory.quant.ebops == {"beta0": quant_beta0}
    assert model.tasks[3].net.input_size == 518 and model.tasks[3].mode == "scale" and model.tasks[3].use_nodes


def test_from_config_reads_pflow_hgq_yaml(monkeypatch):
    """The committed CLIC HGQ2 config constructs unchanged (scale_dict_path is relative to the experiment dir)."""
    monkeypatch.chdir(REPO / "src/hepattn/experiments/clic")
    model = from_config(clic_yaml_model_section(Path("configs/pflow_hgq.yaml")))
    check_clic_structure(model, 1.0e-12)


@R3_SWIGLU_XFAIL
def test_from_config_reads_generated_polaris_yaml(tmp_path):
    """The config polaris/03_make_config.py generates (what Polaris trains) constructs unchanged."""
    out = tmp_path / "polaris_hgq.yaml"
    env = {**os.environ, "HGQ_CONFIG_OUT": str(out)}
    subprocess.run([sys.executable, str(REPO / "polaris/03_make_config.py")], env=env, check=True, capture_output=True)
    model = from_config(clic_yaml_model_section(out))
    check_clic_structure(model, 0.0)


def test_decoder_does_not_track_tasks(batch):
    """The model owns the task variables once; the decoder holds the tasks untracked (no duplicate variables)."""
    nmodel = build_mf("lin", None, batch)
    paths = [v.path for v in nmodel.variables]
    assert len(paths) == len(set(paths)), "duplicate variable paths"
    assert not any(isinstance(layer, ObjectHitMaskTask) for layer in nmodel.decoder._flatten_layers(include_self=False))  # noqa: SLF001
    assert isinstance(nmodel.decoder, NDecoder) and len(nmodel.decoder.tasks) == 4
    assert any(p.startswith("maskformer/task1/") for p in paths)
    assert "maskformer/decoder/initial_queries" in paths


# ------------------------------------------------------------------------------------------ init (no port)
def leaf_stats(layer) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.asarray(keras.ops.convert_to_numpy(layer.kernel), dtype=np.float64)
    bias = np.asarray(keras.ops.convert_to_numpy(layer.bias), dtype=np.float64)
    return kernel, bias


@R3_XFAIL
def test_task_heads_decoder_and_queries_start_from_the_hybrid_init(batch):
    """Carried gate (S2 -> S3): task heads start from torch nn.Linear init (the hybrid ports them from torch Dense),
    decoder FFNs from Keras glorot / zero bias (the hybrid builds them from the config), initial queries ~ N(0, 1).

    Bounds and the torch side are checked on the same layer shapes (torch reference model, seed 0).
    """
    keras.utils.set_random_seed(0)
    nmodel = build_mf("lin", None, batch)
    tmodel = torch_model("lin", seed=0, perturb=False)
    checked = 0
    for ntask, ttask in zip(nmodel.tasks, tmodel.tasks, strict=True):
        for attr in ("net", "object_net", "node_net"):
            ndense = getattr(ntask, attr, None)
            if ndense is None:
                continue
            tlinears = [m for m in getattr(ttask, attr).net if isinstance(m, torch.nn.Linear)]
            for layer, lin in zip([*ndense.hidden, ndense.final], tlinears, strict=True):
                kernel, bias = leaf_stats(layer)
                fan_in = kernel.shape[0]
                bound = 1 / np.sqrt(fan_in)
                t_w = lin.weight.detach().numpy().astype(np.float64)
                assert np.abs(t_w).max() <= bound * (1 + 1e-6)  # the torch side has this bound too
                assert np.abs(kernel).max() <= bound * (1 + 1e-6), f"{layer.name}: kernel outside U(+-1/sqrt(fan_in))"
                assert np.abs(bias).max() <= bound * (1 + 1e-6) and np.any(bias != 0), f"{layer.name}: bias not U(+-1/sqrt(fan_in))"
                if kernel.size >= 2000:
                    assert np.std(kernel) == pytest.approx(bound / np.sqrt(3), rel=0.05), layer.name
                    assert np.std(kernel) == pytest.approx(np.std(t_w), rel=0.05), layer.name
                checked += 1
    assert checked == 4 + 2 + 3 + 2 + 6  # classification, mask object net, incidence net + node net, regression
    for layer in nmodel.decoder.layers:
        for res in (layer.q_dense, layer.kv_dense):
            for leaf in [*res.fn.hidden, res.fn.final]:
                kernel, bias = leaf_stats(leaf)
                limit = np.sqrt(6 / sum(kernel.shape))
                assert not np.any(bias), f"{leaf.name}: decoder FFN bias must start at zero (Keras default)"
                assert np.abs(kernel).max() <= limit * (1 + 1e-6) and np.abs(kernel).max() > 0.9 * limit, f"{leaf.name}: not glorot-uniform"
    queries = np.asarray(keras.ops.convert_to_numpy(nmodel.decoder.initial_queries), dtype=np.float64)
    assert nmodel.decoder.initial_queries.trainable
    assert queries.shape == (NUM_QUERIES, DIM)
    assert np.std(queries) == pytest.approx(1.0, rel=0.05) and abs(np.mean(queries)) < 0.05
    assert np.std(queries) == pytest.approx(float(tmodel.decoder.initial_queries.detach().std()), rel=0.05)
