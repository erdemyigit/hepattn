"""Saving and loading a hybrid with linformer attention before its cores exist.

The masked-Linformer core of a ``KerasAttention`` takes its shapes from the first call, so it is built at the first
forward, in float mode too, and ``port_maskformer`` holds the ported proj_k / proj_v until then. Before the R4b fix
round, ``state_dict()`` of a model that had not run a forward silently left out all 132 core tensors (the 36 trained
projections among them), and a strict load of that state into a fresh model reported all keys matched and then built
the cores from random projections (2.6 away from torch). Both calls now raise until ``materialize`` runs a forward.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import org_model_utils as U  # ty: ignore [unresolved-import]
from test_maskformer_parity import flatten_outputs  # ty: ignore [unresolved-import]
from test_org_model_parity import TOLS, scaled_err  # ty: ignore [unresolved-import]

from hepattn.keras.porting import port_maskformer

OUT_TOL = TOLS["linformer"][0]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    gen = U.generate_polaris_config(tmp_path_factory.mktemp("gen") / "polaris_hgq.yaml")
    org = U.torch_reference_config(U.merge_org_configs())
    torch.manual_seed(0)
    tmodel = U.instantiate(org)
    inputs, _ = U.clic_dummy_batch(2, seed=0)
    with torch.no_grad():
        ref = flatten_outputs(tmodel(inputs))
    return gen, tmodel, inputs, ref


def fresh(gen: dict, seed: int):
    torch.manual_seed(seed)
    return U.instantiate(gen, quantized=False)


def max_err(model, inputs, ref) -> float:
    with torch.no_grad():
        out = flatten_outputs(model(inputs))
    return max(scaled_err(ref[key], out[key]) for key in ref if ref[key].dtype != torch.bool)


def core_keys(state: dict) -> list[str]:
    return [key for key in state if ".core." in key or key.endswith(("proj_k_dead", "proj_v_dead"))]


def test_state_dict_before_the_first_forward_raises(setup):
    gen, tmodel, _, _ = setup
    model = fresh(gen, 1)
    port_maskformer(tmodel, model)  # the projections wait for the core
    with pytest.raises(RuntimeError, match=r"materialize"):
        model.state_dict()


@pytest.mark.parametrize("strict", [True, False])
def test_load_before_the_first_forward_raises(setup, strict):
    gen, _, inputs, _ = setup
    state = fresh(gen, 1).materialize(inputs).state_dict()
    with pytest.raises(RuntimeError, match=r"materialize"):
        fresh(gen, 2).load_state_dict(state, strict=strict)


def test_materialized_state_round_trips_the_ported_projections(setup):
    """Port -> materialize -> save -> fresh model -> materialize -> strict load: the reloaded model is torch's.
    Control: the fresh model before the load (its own random projections) is far from torch. Measured
    ($S/r4b_fix/logs/roundtrip.out): reloaded 4.489e-6 (tolerance: the full-model one, 1.02e-5), control 57.8.
    """
    gen, tmodel, inputs, ref = setup
    model = fresh(gen, 1)
    port_maskformer(tmodel, model)
    state = model.materialize(inputs).state_dict()
    assert len(core_keys(state)) == 132, len(core_keys(state))  # measured: 469 keys after the forward, 337 before
    target = fresh(gen, 2).materialize(inputs)
    control = max_err(target, inputs, ref)
    target.load_state_dict(state)
    err = max_err(target, inputs, ref)
    assert err <= OUT_TOL, err
    assert control >= 10 * OUT_TOL, control


def test_materialize_keeps_the_training_mode(setup):
    gen, _, inputs, _ = setup
    model = fresh(gen, 3)
    model.train()
    assert model.materialize(inputs).training
    model.eval()
    assert not model.materialize(inputs).training
