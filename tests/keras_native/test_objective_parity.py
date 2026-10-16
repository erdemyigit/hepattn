"""S4: keras_native costs and losses equal the torch task.cost / task.loss on random CLIC-shaped tensors."""

import logging
import warnings

import keras
import numpy as np
import pytest
import torch
import yaml

from hepattn.keras_native import losses as L
from hepattn.keras_native import objective as O
from hepattn.models import loss as torch_loss
from hepattn.models.matcher import Matcher

from .objective_utils import (
    CLIC_YAML,
    LAYER_TASKS,
    ORG_BASE_YAML,
    assert_close_same_nonfinite,
    max_errors,
    random_outputs,
    random_targets,
    to_torch,
    torch_tasks,
)

COST_TOL = {"atol": 1e-5, "rtol": 1e-5}  # plan gate S4: each task cost to 1e-5
LOSS_TOL = {"atol": 1e-4, "rtol": 1e-3}  # PARITY.md maskformer.loss rows


@pytest.fixture(scope="module")
def tasks():
    return {t.name: t for t in torch_tasks()}


@pytest.fixture(scope="module")
def specs():
    return {s.name: s for s in O.clic_specs()}


def _np(x):
    return keras.ops.convert_to_numpy(x)


@pytest.mark.parametrize(("seed", "batch"), [(0, 1), (1, 3), (2, 4)])
def test_task_costs_equal_torch(tasks, specs, seed, batch):
    rng = np.random.default_rng(seed)
    targets = random_targets(rng, batch)
    outputs = random_outputs(rng, targets["node_valid"])
    targets_t = to_torch(targets)
    n_compared = 0
    for layer, layer_out in outputs.items():
        for name in LAYER_TASKS[layer]:
            ref = tasks[name].cost(to_torch(layer_out[name]), targets_t)
            got = specs[name].cost(layer_out[name], targets)
            assert set(ref) == set(got), f"{layer}.{name}: cost keys {set(ref)} vs {set(got)}"
            for key in ref:
                ref_np = ref[key].numpy()
                abs_err, rel_err = assert_close_same_nonfinite(ref_np, _np(got[key]), **COST_TOL, what=f"{layer}.{name}.{key}")
                assert np.isfinite(ref_np).any()
                print(f"cost {layer}.{name}.{key} B={batch}: max abs {abs_err:.2e}, max rel {rel_err:.2e}")
                n_compared += 1
    assert n_compared == 7


@pytest.mark.parametrize("padded_node_targets", [False, True], ids=["real-like", "dummy-like"])
def test_task_losses_equal_torch(tasks, specs, padded_node_targets):
    """Losses on unpermuted outputs (the matching is tested separately).

    ``dummy-like`` puts target 1 at padded nodes. Before org ``a4263c1`` that made torch and native mask_bce
    inf at the 3 mask layers; with the per-event normalisation every loss is finite.
    """
    rng = np.random.default_rng(10)
    targets = random_targets(rng, 4, padded_node_targets=padded_node_targets)
    outputs = random_outputs(rng, targets["node_valid"])
    targets_t = to_torch(targets)
    n_compared = 0
    for layer, layer_out in outputs.items():
        for name in LAYER_TASKS[layer]:
            ref = tasks[name].loss(to_torch(layer_out[name]), targets_t)
            got = specs[name].loss(layer_out[name], targets)
            assert set(ref) == set(got)
            for key in ref:
                r, g = ref[key].numpy(), _np(got[key])
                assert np.isfinite(r), f"{layer}.{name}.{key}: torch {r}"
                abs_err, rel_err = assert_close_same_nonfinite(r, g, **LOSS_TOL, what=f"{layer}.{name}.{key}")
                print(f"loss {layer}.{name}.{key} ({'dummy' if padded_node_targets else 'real'}): {r:.6f}, abs {abs_err:.2e}, rel {rel_err:.2e}")
                n_compared += 1
    assert n_compared == 10


def _per_event_mask_bce(logits, targets, object_valid, pad):
    """The a4263c1 formula in numpy: sum over the event's valid nodes / count, mean over the valid objects."""
    bce = L.bce_with_logits(logits, targets)
    per = np.sum(np.where(pad[:, None, :], np.asarray(bce), 0.0), -1) / np.maximum(pad.sum(-1), 1)[:, None]
    return per[object_valid].mean()


def _cross_event_mask_bce(logits, targets, object_valid, pad):
    """The formula before a4263c1: every valid object weighted by EVERY event's node mask (control)."""
    bce = np.asarray(L.bce_with_logits(logits, targets), dtype=np.float64)
    term = np.einsum("qpn,bn->bqp", bce, pad.astype(np.float64)) / np.maximum(pad.sum(-1), 1)[:, None, None]
    return term[np.broadcast_to(object_valid[None], term.shape)].mean()


def test_mask_losses_are_normalised_per_event():
    """Torch (a4263c1) and native reduce each object over its own event's nodes. The old cross-event
    formula differs (control); with one event both formulas agree, so the difference is the coupling only.
    """
    rng = np.random.default_rng(3)
    batch = 3
    targets = random_targets(rng, batch, padded_node_targets=False)
    logits = rng.normal(size=(batch, 150, 160)).astype(np.float32)
    tgt = targets["particle_node_valid"].astype(np.float32)
    ov, pad = targets["particle_valid"], targets["node_valid"]
    args_t = (torch.from_numpy(logits), torch.from_numpy(tgt))
    kw_t = {"object_valid_mask": torch.from_numpy(ov), "input_pad_mask": torch.from_numpy(pad)}

    bce_t = torch_loss.mask_bce_loss(*args_t, **kw_t).item()
    bce_n = float(L.mask_bce_loss(logits, tgt, ov, pad))
    dice_t = torch_loss.mask_dice_loss(*args_t, **kw_t).item()
    dice_n = float(L.mask_dice_loss(logits, tgt, ov, pad))
    np.testing.assert_allclose(bce_n, bce_t, rtol=1e-6)
    np.testing.assert_allclose(dice_n, dice_t, rtol=1e-6)

    np.testing.assert_allclose(_per_event_mask_bce(logits, tgt, ov, pad), bce_t, rtol=1e-6)
    cross = _cross_event_mask_bce(logits, tgt, ov, pad)
    print(f"mask_bce: torch {bce_t:.6f}, native {bce_n:.6f}, cross-event formula {cross:.6f}")
    assert abs(cross - bce_t) > 1e-3 * abs(bce_t), (cross, bce_t)  # measured 1.3e-3 relative; tolerance above 1e-6
    one = slice(0, 1)
    np.testing.assert_allclose(
        _cross_event_mask_bce(logits[one], tgt[one], ov[one], pad[one]), float(L.mask_bce_loss(logits[one], tgt[one], ov[one], pad[one])), rtol=1e-6
    )


def test_weighted_ce_equals_torch_and_is_not_the_plain_mean():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 6, size=(4, 150))
    # confident logits, but class 2 (weight 8) is always predicted wrong: its loss dominates the weighted mean
    logits = (4.0 * np.eye(6)[labels] + rng.normal(size=(4, 150, 6))).astype(np.float32)
    logits[labels == 2] = (4.0 * np.eye(6)[0] + rng.normal(size=(int((labels == 2).sum()), 6))).astype(np.float32)
    weight = np.array([1.0, 3.0, 8.0, 1.5, 1.0, 0.5], dtype=np.float32)
    ref = torch_loss.object_ce_loss(torch.from_numpy(logits), torch.from_numpy(labels), weight=torch.from_numpy(weight)).item()
    got = float(L.object_ce_loss(logits, labels, weight=weight))
    np.testing.assert_allclose(got, ref, rtol=1e-6)

    plain = float(L.object_ce_loss(logits, labels))
    ref_plain = torch_loss.object_ce_loss(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    np.testing.assert_allclose(plain, ref_plain, rtol=1e-6)
    # the weighted value differs from both wrong normalisations: the plain mean, and mean(w_i l_i)
    nll = -np.take_along_axis(np.asarray(keras.ops.log_softmax(logits, -1)), labels[..., None], -1)[..., 0]
    wrong_norm = float(np.mean(weight[labels] * nll))
    print(f"weighted CE {got:.5f} (torch {ref:.5f}), plain mean {plain:.5f}, mean(w*l) {wrong_norm:.5f}")
    assert abs(got - plain) > 0.5 * plain, (got, plain)
    assert abs(got - wrong_norm) > 0.1 * got, (got, wrong_norm)


def test_no_valid_object_gives_zero_and_torch_nan_only_for_kl_div_and_l1(specs):
    """No valid object: the mask losses give 0 in torch too (``clamp_min(1)``, a4263c1). The documented
    deviation remains for kl_div and the regression l1: torch's mean over an empty selection is NaN, the
    static-shape port returns 0.
    """
    rng = np.random.default_rng(5)
    targets = random_targets(rng, 2, padded_node_targets=False, n_particles=[0, 0])
    outputs = random_outputs(rng, targets["node_valid"])["final"]
    tasks = {t.name: t for t in torch_tasks()}
    targets_t = to_torch(targets)
    torch_nan = {"mask": False, "incidence": True, "regression": True}
    for name, nan in torch_nan.items():
        ref = tasks[name].loss(to_torch(outputs[name]), targets_t)
        got = specs[name].loss(outputs[name], targets)
        assert set(ref) == set(got)
        for key in ref:
            assert np.isnan(ref[key].item()) == nan, f"{name}.{key}: torch {ref[key].item()}"
            if not nan:
                assert ref[key].item() == 0.0, f"{name}.{key}: torch {ref[key].item()}"
            assert float(got[key]) == 0.0, f"{name}.{key}: native {float(got[key])}"


def test_costs_and_losses_are_float32_for_bfloat16_outputs(specs):
    rng = np.random.default_rng(6)
    targets = random_targets(rng, 2, padded_node_targets=False)
    outputs = random_outputs(rng, targets["node_valid"])["final"]
    bf16 = {name: {k: keras.ops.cast(v, "bfloat16") for k, v in out.items()} for name, out in outputs.items()}
    for name, spec in specs.items():
        for value in [*spec.cost(bf16[name], targets).values(), *spec.loss(bf16[name], targets).values()]:
            assert keras.backend.standardize_dtype(value.dtype) == "float32", name


@pytest.mark.parametrize("config", [CLIC_YAML, ORG_BASE_YAML], ids=["pflow_hgq", "org_base"])
def test_specs_from_yaml_equal_clic_specs_and_torch_tasks(tasks, config):
    cfg = yaml.safe_load(config.read_text())
    entries = cfg["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]
    parsed = O.specs_from_config(entries)
    assert parsed == O.clic_specs()
    by_name = {s.name: s for s in parsed}
    assert [s.name for s in parsed] == ["classification", "mask", "incidence", "regression"]
    np.testing.assert_array_equal(by_name["classification"].loss_weights, tasks["classification"].loss_weights.numpy())
    for name in ("classification", "mask", "incidence"):
        assert by_name[name].losses == tasks[name].losses
        assert by_name[name].costs == tasks[name].costs
        assert by_name[name].outputs == tasks[name].outputs
    reg, reg_t = by_name["regression"], tasks["regression"]
    assert (list(reg.fields), reg.loss_fn_name, reg.loss_weight, reg.cost_weight) == (
        reg_t.fields,
        reg_t.loss_fn_name,
        reg_t.loss_weight,
        reg_t.cost_weight,
    )
    assert reg.outputs == reg_t.outputs
    assert by_name["mask"].null_weight == tasks["mask"].null_weight


_CLS = "hepattn.models.task.ObjectClassificationTask"
_MASK = "hepattn.models.task.ObjectHitMaskTask"
_INC = "hepattn.models.task.IncidenceRegressionTask"
_REG = "hepattn.models.task.IncidenceBasedRegressionTask"
_OBJ = {"name": "t", "output_object": "pflow", "target_object": "particle"}
_REG_OK = {**_OBJ, "fields": ["e"], "loss_weight": 1.0, "cost_weight": 1.0, "loss": "l1", "cost": "new"}


def test_spec_from_config_uses_torch_defaults():
    spec = O.spec_from_config(_CLS, {**_OBJ, "num_classes": 3, "losses": {"object_ce": 1}, "costs": {"object_ce": 1}})
    np.testing.assert_array_equal(spec.loss_weights, np.ones(4, dtype=np.float32))  # torch: no class weights, null_weight 1.0
    assert O.spec_from_config(_REG, _REG_OK).loss_fn_name == "l1"


@pytest.mark.parametrize(
    ("class_path", "init_args"),
    [
        (_CLS, {**_OBJ, "losses": {"object_bce": 1}, "costs": {"object_bce": 1}}),  # torch default num_classes=1
        (_CLS, {**_OBJ, "num_classes": 5, "losses": {"object_ce": 1}, "costs": {"object_ce": 1}, "mask_queries": True}),
        (_MASK, {**_OBJ, "input_constituent": "node", "losses": {"mask_focal": 1.0}, "costs": {"mask_dice": 1.0}}),
        (_MASK, {**_OBJ, "input_constituent": "node", "losses": {"mask_bce": 1.0}, "costs": {"mask_bce": 1.0}}),
        (_INC, {**_OBJ, "input_constituent": "node", "losses": {"mask_kl_div": 1.0}, "costs": {"kl_div": 1.0}}),
        (_REG, {**_REG_OK, "loss": "smooth_l1"}),
        (_REG, {**_REG_OK, "cost": "old"}),
        (_REG, {k: v for k, v in _REG_OK.items() if k not in {"loss", "cost"}}),  # torch defaults smooth_l1 / old
        ("hepattn.models.task.HitFilterTask", {}),
    ],
)
def test_unsupported_task_configs_raise(class_path, init_args):
    with pytest.raises(NotImplementedError):
        O.spec_from_config(class_path, init_args)


def test_query_mask_raises(specs):
    rng = np.random.default_rng(7)
    targets = random_targets(rng, 1)
    outputs = random_outputs(rng, targets["node_valid"])
    targets["query_mask"] = np.ones((1, 150), dtype=bool)
    with pytest.raises(NotImplementedError, match="query_mask"):
        O.model_costs(outputs, targets, list(specs.values()))
    with pytest.raises(NotImplementedError, match="query_mask"):
        O.model_losses(outputs, targets, list(specs.values()))


def test_max_errors_helper_discriminates():
    """The parity helper itself fails on a 2e-5 perturbation at 1e-5 tolerance."""
    ref = np.linspace(-1, 1, 100, dtype=np.float32)
    assert max_errors(ref, ref) == (0.0, 0.0)
    with pytest.raises(AssertionError):
        assert_close_same_nonfinite(ref, ref + 2e-5, atol=1e-5, rtol=0.0)
    nan_ref = ref.copy()
    nan_ref[3] = np.nan
    with pytest.raises(AssertionError, match="NaN pattern"):
        assert_close_same_nonfinite(nan_ref, ref, atol=1, rtol=1)
    inf_ref = ref.copy()
    inf_ref[3] = np.inf
    with pytest.raises(AssertionError, match="inf"):
        assert_close_same_nonfinite(inf_ref, np.where(np.arange(100) == 3, -np.inf, ref), atol=1, rtol=1)


def test_total_loss_is_the_sum_of_every_leaf():
    """Distinct powers of two: dropping, doubling or zeroing any leaf changes the total exactly."""
    losses = {
        "layer_0": {"mask": {"mask_bce": 1.0, "mask_dice": 2.0}},
        "layer_1": {"classification": {"object_ce": 4.0}, "mask": {"mask_bce": 8.0, "mask_dice": 16.0}},
        "final": {"classification": {"object_ce": 32.0}, "regression": {"l1": 64.0}},
    }
    leaves = keras.tree.map_structure(lambda v: keras.ops.convert_to_tensor(v, "float32"), losses)
    assert float(O.total_loss(leaves)) == 127.0


def test_mask_bce_sample_weight_with_non_unit_null_weight_equals_torch():
    """null_weight != 1 weights the BCE of target-0 entries; CLIC uses 1.0, so this path needs its own case."""
    rng = np.random.default_rng(11)
    targets = random_targets(rng, 3, padded_node_targets=False)
    outputs = random_outputs(rng, targets["node_valid"])["final"]["mask"]
    task = torch_tasks()[1]
    assert task.name == "mask"
    values = {}
    for null_weight in (0.5, 1.0):
        task.null_weight = null_weight
        spec = O.ObjectHitMaskSpec(
            name="mask",
            input_constituent="node",
            output_object="pflow",
            target_object="particle",
            losses={"mask_bce": 5.0, "mask_dice": 1.0},
            costs={"mask_dice": 1.0},
            null_weight=null_weight,
        )
        ref = task.loss(to_torch(outputs), to_torch(targets))
        got = spec.loss(outputs, targets)
        for key in ref:
            assert_close_same_nonfinite(ref[key].numpy(), _np(got[key]), **LOSS_TOL, what=f"null_weight={null_weight} {key}")
        values[null_weight] = float(got["mask_bce"])
    print(f"mask_bce: null_weight 0.5 -> {values[0.5]:.6f}, 1.0 -> {values[1.0]:.6f}")
    assert abs(values[0.5] - values[1.0]) > 0.1 * values[1.0], values  # control: the weight matters


def test_mask_dice_cost_with_finite_logits_at_padded_nodes_equals_torch(tasks, specs):
    """The model sets finfo.min at padded nodes; here they are finite, so only the pad multiply removes them."""
    rng = np.random.default_rng(12)
    targets = random_targets(rng, 3, padded_node_targets=False)
    logits = (3.0 * rng.normal(size=(3, 150, 160))).astype(np.float32)
    ref = tasks["mask"].cost({"pflow_node_logit": torch.from_numpy(logits)}, to_torch(targets))["mask_dice"].numpy()
    got = _np(specs["mask"].cost({"pflow_node_logit": logits}, targets)["mask_dice"])
    assert_close_same_nonfinite(ref, got, **COST_TOL, what="mask_dice cost, finite padded logits")
    unmasked = torch_loss.mask_dice_cost(torch.from_numpy(logits), torch.from_numpy(targets["particle_node_valid"]).float()).numpy()
    assert np.abs(unmasked - ref).max() > 1e-2  # control: the padded nodes would change the cost


def test_kl_div_with_zero_prediction_at_a_target_entry_equals_torch(tasks, specs):
    """Pred == 0 where true > 0: torch adds eps and stays finite (-true * log(1e-8)); without eps it is inf."""
    rng = np.random.default_rng(13)
    targets = random_targets(rng, 2, padded_node_targets=False)
    outputs = random_outputs(rng, targets["node_valid"])["final"]["incidence"]
    pred = outputs["pflow_incidence"].copy()
    pred[:, :5, :20] = 0.0  # valid objects (>= 10 per event) and valid nodes (>= 50 per event)
    assert (targets["particle_incidence"][:, :5, :20] > 0).all()
    out = {"pflow_incidence": pred}
    targets_t = to_torch(targets)
    for kind in ("cost", "loss"):
        ref = getattr(tasks["incidence"], kind)(to_torch(out), targets_t)["kl_div"].numpy()
        got = _np(getattr(specs["incidence"], kind)(out, targets)["kl_div"])
        assert np.isfinite(ref).all(), kind
        tol = COST_TOL if kind == "cost" else LOSS_TOL
        abs_err, rel_err = assert_close_same_nonfinite(ref, got, **tol, what=f"kl_div {kind} with zero predictions")
        print(f"kl_div {kind} with pred=0 entries: max abs {abs_err:.2e}, max rel {rel_err:.2e}")


def test_non_empty_encoder_outputs_raise(specs):
    rng = np.random.default_rng(14)
    targets = random_targets(rng, 1, padded_node_targets=False)
    outputs = random_outputs(rng, targets["node_valid"])
    spec_list = list(specs.values())
    O.model_costs({**outputs, "encoder": {}}, targets, spec_list)  # torch always emits an empty one
    for encoder in ({"query_mask": np.ones((1, 150), dtype=bool)}, {"hit_filter": {"node_logit": np.zeros((1, 160), np.float32)}}):
        with pytest.raises(NotImplementedError, match="encoder"):
            O.model_costs({**outputs, "encoder": encoder}, targets, spec_list)
        with pytest.raises(NotImplementedError, match="encoder"):
            O.model_losses({**outputs, "encoder": encoder}, targets, spec_list)


def _torch_matcher(**init_args) -> Matcher:
    with warnings.catch_warnings():  # lap1015 here is not the GIL-releasing build (R0.md); the warning is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return Matcher(**init_args)


@pytest.mark.parametrize(("config", "solver"), [(CLIC_YAML, "scipy"), (ORG_BASE_YAML, "lap1015_late")], ids=["pflow_hgq", "org_base"])
def test_matcher_n_jobs_from_yaml_equals_torch_matcher(caplog, config, solver):
    """pflow_hgq.yaml (scipy) and org/main base.yaml (lap1015_late), both 16 threads. Native solves
    lap1015_late with scipy and logs that once; scipy logs nothing.
    """
    entry = yaml.safe_load(config.read_text())["model"]["model"]["init_args"]["matcher"]
    assert entry["init_args"]["default_solver"] == solver
    torch_matcher = _torch_matcher(**entry["init_args"])
    assert torch_matcher.parallel_solver
    O._log_lap1015_fallback.cache_clear()  # noqa: SLF001
    with caplog.at_level(logging.WARNING, logger=O.__name__):
        assert O.matcher_n_jobs_from_config(entry) == torch_matcher.n_jobs == 16
        assert O.matcher_n_jobs_from_config(entry) == 16
    expected = ["keras_native solves default_solver=lap1015_late with scipy"] if solver == "lap1015_late" else []
    assert [r.getMessage().split(":")[0] for r in caplog.records] == expected


def test_matcher_n_jobs_defaults_follow_torch_matcher():
    entry = yaml.safe_load(ORG_BASE_YAML.read_text())["model"]["model"]["init_args"]["matcher"]
    base = {"class_path": entry["class_path"], "init_args": {"adaptive_solver": False}}
    assert O.matcher_n_jobs_from_config(base) == 1  # torch default parallel_solver=False
    parallel = {**base, "init_args": {"adaptive_solver": False, "parallel_solver": True}}
    assert O.matcher_n_jobs_from_config(parallel) == Matcher().n_jobs
    explicit_off = {**base, "init_args": {"adaptive_solver": False, "device_solver": None}}
    assert O.matcher_n_jobs_from_config(explicit_off) == 1


_MATCHER = "hepattn.models.matcher.Matcher"


@pytest.mark.parametrize(
    "entry",
    [
        {"class_path": _MATCHER, "init_args": {}},  # torch default adaptive_solver=True
        {"class_path": _MATCHER, "init_args": {"adaptive_solver": False, "default_solver": "lap1015"}},
        {"class_path": _MATCHER, "init_args": {"adaptive_solver": False, "default_solver": "lap1015_early"}},
        {"class_path": _MATCHER, "init_args": {"adaptive_solver": True, "default_solver": "lap1015_late"}},
        {"class_path": _MATCHER, "init_args": {"adaptive_solver": False, "device_solver": "jv"}},
        {"class_path": _MATCHER, "init_args": {"adaptive_solver": False, "unknown": 1}},
        {"class_path": "my.OtherMatcher", "init_args": {"adaptive_solver": False}},
    ],
    ids=["adaptive-default", "lap1015", "lap1015_early", "lap1015_late-adaptive", "device-jv", "unknown-option", "other-class"],
)
def test_unsupported_matcher_configs_raise(entry):
    with pytest.raises(NotImplementedError):
        O.matcher_n_jobs_from_config(entry)
