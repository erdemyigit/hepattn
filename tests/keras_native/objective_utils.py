"""Random CLIC-shaped outputs/targets and the torch twins of the CLIC tasks, for the S4 parity tests."""

import numpy as np
import torch
from torch import nn

from hepattn.models.dense import Dense
from hepattn.models.task import IncidenceBasedRegressionTask, IncidenceRegressionTask, ObjectClassificationTask, ObjectHitMaskTask

from .conftest import REPO_ROOT

P = 150  # queries = target particles
N = 160  # nodes
C = 6  # 5 classes + null
FIELDS = ("e", "pt", "eta", "sinphi", "cosphi")
SCALE_DICT = str(REPO_ROOT / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml")
CLIC_YAML = REPO_ROOT / "src/hepattn/experiments/clic/configs/pflow_hgq.yaml"
# org/main ac0da8f model config: the target of the retarget (matcher lap1015_late, same task losses/costs).
ORG_BASE_YAML = REPO_ROOT / "src/hepattn/experiments/clic/configs/base.yaml"
# Which tasks the decoder emits at each layer (pflow_hgq.yaml flags, 2 decoder layers here).
LAYER_TASKS = {
    "layer_0": ("mask",),
    "layer_1": ("classification", "mask"),
    "final": ("classification", "mask", "incidence", "regression"),
}


def torch_tasks(dim: int = 32) -> nn.ModuleList:
    """The four CLIC tasks with the pflow_hgq.yaml loss/cost settings (nets at a small dim; unused by cost/loss)."""
    return nn.ModuleList([
        ObjectClassificationTask(
            name="classification",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            num_classes=5,
            losses={"object_ce": 2},
            costs={"object_ce": 2},
            net=Dense(input_size=dim, output_size=6, hidden_layers=[256, 128, 32], activation=nn.SiLU()),
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
            dim=dim,
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
            net=Dense(input_size=dim, hidden_layers=2, activation=nn.SiLU()),
            node_net=Dense(input_size=dim, hidden_layers=1),
            has_intermediate_loss=False,
        ),
        IncidenceBasedRegressionTask(
            name="regression",
            fields=list(FIELDS),
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
            net=Dense(input_size=dim + 6 + dim, output_size=5, hidden_layers=[512, 256, 128, 64, 32], activation=nn.SiLU()),
            has_intermediate_loss=False,
        ),
    ])


def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def random_targets(rng: np.random.Generator, batch: int, *, padded_node_targets: bool = True, n_particles=None) -> dict:
    """CLIC-like targets. ``padded_node_targets`` keeps random True mask targets at padded nodes (like the
    dummy dataset; before org a4263c1 this made torch mask_bce inf); False zeroes them (like real data).
    """
    n_nodes = rng.integers(50, N, size=batch)
    n_part = rng.integers(10, P, size=batch) if n_particles is None else np.asarray(n_particles)
    node_valid = np.arange(N)[None] < n_nodes[:, None]
    particle_valid = np.arange(P)[None] < n_part[:, None]
    particle_class = rng.integers(0, 5, size=(batch, P))
    particle_class[~particle_valid] = 5
    node_targets = rng.random((batch, P, N)) > 0.8
    if not padded_node_targets:
        node_targets &= node_valid[:, None, :] & particle_valid[:, :, None]
    targets = {
        "node_valid": node_valid,
        "particle_valid": particle_valid,
        "particle_class": particle_class.astype(np.int64),
        "particle_node_valid": node_targets,
        "particle_incidence": rng.random((batch, P, N)).astype(np.float32),
    }
    for f in FIELDS:
        values = rng.normal(size=(batch, P)).astype(np.float32)
        values[~particle_valid] = np.nan
        targets[f"particle_{f}"] = values
    return targets


def random_layer_outputs(rng: np.random.Generator, node_valid: np.ndarray, tasks=LAYER_TASKS["final"]) -> dict:
    batch = node_valid.shape[0]
    out = {}
    if "classification" in tasks:
        logits = (2.0 * rng.normal(size=(batch, P, C))).astype(np.float32)
        out["classification"] = {"pflow_logit": logits, "pflow_class_prob": _softmax(logits, -1).astype(np.float32)}
    if "mask" in tasks:
        logit = (3.0 * rng.normal(size=(batch, P, N))).astype(np.float32)
        logit[~np.broadcast_to(node_valid[:, None, :], logit.shape)] = np.finfo(np.float32).min
        out["mask"] = {"pflow_node_logit": logit}
    if "incidence" in tasks:
        inc = _softmax(rng.normal(size=(batch, P, N)), 1) * node_valid[:, None, :]
        out["incidence"] = {"pflow_incidence": inc.astype(np.float32)}
    if "regression" in tasks:
        out["regression"] = {
            "pflow_regr": rng.normal(size=(batch, P, len(FIELDS))).astype(np.float32),
            "pflow_proxy_regr": rng.normal(size=(batch, P, len(FIELDS))).astype(np.float32),
        }
    return out


def random_outputs(rng: np.random.Generator, node_valid: np.ndarray, layer_tasks=None) -> dict:
    layer_tasks = LAYER_TASKS if layer_tasks is None else layer_tasks
    return {layer: random_layer_outputs(rng, node_valid, tasks) for layer, tasks in layer_tasks.items()}


def to_torch(tree):
    if isinstance(tree, dict):
        return {k: to_torch(v) for k, v in tree.items()}
    return torch.from_numpy(np.array(tree))


def to_numpy(tree):
    if isinstance(tree, dict):
        return {k: to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy().copy()
    return np.asarray(tree)


def flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def max_errors(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    finite = np.isfinite(ref)
    if not finite.any():
        return 0.0, 0.0
    diff = np.abs(ref[finite] - got[finite])
    return float(diff.max()), float((diff / np.maximum(np.abs(ref[finite]), 1e-30)).max())


def assert_close_same_nonfinite(ref, got, *, atol: float, rtol: float, what: str = "") -> tuple[float, float]:
    """Same NaN/inf pattern (and the same inf values), finite entries within atol + rtol * |ref|."""
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    assert ref.shape == got.shape, f"{what}: shape {ref.shape} vs {got.shape}"
    np.testing.assert_array_equal(np.isnan(ref), np.isnan(got), err_msg=f"{what}: NaN pattern")
    np.testing.assert_array_equal(np.where(np.isinf(ref), ref, 0), np.where(np.isinf(got), got, 0), err_msg=f"{what}: inf")
    finite = np.isfinite(ref)
    np.testing.assert_allclose(got[finite], ref[finite], atol=atol, rtol=rtol, err_msg=what)
    return max_errors(ref, got)
