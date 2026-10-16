"""``port_to_keras.build_keras_maskformer_from_config`` builds the org/main model from a torch config.

``eval_keras.py``, ``evaluate_hgq.py`` and ``port_to_keras.py`` build the hybrid with this function. For a torch
``MaskFormer`` section it must keep ``input_sort_field`` (``linformer.yaml``: phi, org PR #9): with the masked
Linformer the sort order selects the E/F row of each token, so an unsorted hybrid is another model (control below).
Tolerances: the hybrid org-model rows of ``test_org_model_parity.py`` (2x measured).
"""

import copy

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import org_model_utils as U  # ty: ignore [unresolved-import]
from test_maskformer_parity import flatten_outputs  # ty: ignore [unresolved-import]
from test_org_model_parity import TOLS, scaled_err  # ty: ignore [unresolved-import]

from hepattn.experiments.clic.port_to_keras import build_keras_maskformer_from_config
from hepattn.keras.porting import port_maskformer

OUT_TOL = TOLS["linformer"][0]


@pytest.fixture(scope="module")
def org() -> dict:
    cfg = U.torch_reference_config(U.merge_org_configs())
    for task in cfg["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task.get("init_args", {}):
            task["init_args"]["scale_dict_path"] = U.SCALE_DICT
    return cfg


def errors(org: dict, model_cfg: dict, seed: int = 0) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    tmodel = U.instantiate(org)
    kmodel = build_keras_maskformer_from_config(model_cfg).eval()
    port_maskformer(tmodel, kmodel)
    inputs, _ = U.clic_dummy_batch(2, seed=seed)
    with torch.no_grad():
        flat_t, flat_k = flatten_outputs(tmodel(inputs)), flatten_outputs(kmodel(inputs))
    assert set(flat_t) == set(flat_k), set(flat_t) ^ set(flat_k)
    bools = {key: torch.equal(flat_t[key], flat_k[key]) for key in flat_t if flat_t[key].dtype == torch.bool}
    outs = {key: scaled_err(flat_t[key], flat_k[key]) for key in flat_t if flat_t[key].dtype != torch.bool}
    return bools, outs


def test_torch_config_builds_the_sorted_org_model(org):
    model_cfg = org["model"]["model"]
    assert model_cfg["init_args"]["input_sort_field"] == "phi"
    bools, outs = errors(org, model_cfg)
    assert bools and all(bools.values()), [key for key, ok in bools.items() if not ok]
    assert max(outs.values()) <= OUT_TOL, max(outs, key=outs.get)


def test_control_without_sorting_the_hybrid_leaves_the_org_model(org):
    """Measured: outputs move by >> 10x the tolerance when the hybrid does not sort (what the function built before)."""
    model_cfg = copy.deepcopy(org["model"]["model"])
    model_cfg["init_args"]["input_sort_field"] = None
    _, outs = errors(org, model_cfg)
    assert max(outs.values()) >= 10 * OUT_TOL, max(outs.values())
