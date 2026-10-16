"""S5: ``KerasMaskFormer.quant_losses()`` counts each keras layer once.

The task modules are registered twice in the torch module tree (``model.tasks`` and ``model.decoder.tasks``), so
``keras_layers()`` yields every task-head layer twice. Before S5 ``quant_losses()`` summed over that walk: Polaris G2
(job 7753080, real CLIC events, beta0 1e-12) recorded 211 walked vs 183 unique layers and a penalty of 3.418 vs 3.137
(+8.9%) on the generated org model.

- On the generated Polaris org model (production dims, beta0 1e-12 as G2, CLIC dummy events): 211 walked, 183 unique;
  ``quant_losses()`` equals the per-unique-layer sum (3.13701 here, G2's JAX arm 3.137437 on real events); the old walk
  (control) is larger by the task heads' terms, by the same fraction as in G2: +8.945% here, +8.94% in G2.
- On the golden models (``tests/keras_native/golden/mf_{torch,org}.npz``, rebuilt with the generator's own code and
  weights): ``quant_losses()`` equals the stored per-unique-layer terms, which ``tests/keras_native/test_mf_golden.py``
  asserts the native ``stateless_call(return_losses=True)`` terms equal; the old walk reproduces the stored
  double-counted ``sum_raw`` (control).
"""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import keras
import org_model_utils as U  # ty: ignore [unresolved-import]

from hepattn.keras import callbacks as C

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "keras_native" / "golden"


def make_config_constant(name: str):
    import ast  # noqa: PLC0415

    for node in ast.parse((REPO / "polaris/03_make_config.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def walked_sum(model) -> float:
    """The pre-S5 quant_losses(): every layer of the keras_layers() walk, duplicates included."""
    return float(torch.stack([torch.as_tensor(t) for layer in model.keras_layers() for t in layer.losses]).sum())


def unique_sum(model) -> float:
    return float(torch.stack([torch.as_tensor(t) for layer in model.unique_keras_layers() for t in layer.losses]).sum())


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> dict:
    return U.generate_polaris_config(tmp_path_factory.mktemp("cfg") / "polaris_hgq.yaml")


def test_generated_org_model_counts_each_layer_once(generated):
    cfg = copy.deepcopy(generated)
    cfg["model"]["model"]["init_args"]["quant"]["ebops"] = {"beta0": 1.0e-12}  # as polaris/15_g2_forward.py
    keras.utils.set_random_seed(0)  # the WRAP datalanes fit their integer bits to the activations: EBOPs depend on the init
    model = U.instantiate(cfg, quantized=True)
    inputs, _ = U.clic_dummy_batch(num_events=2)
    with torch.no_grad():
        model.materialize(inputs)
        model.train()(inputs)
    walked, unique = list(model.keras_layers()), model.unique_keras_layers()
    assert (len(walked), len(unique)) == (211, 183)
    task_layers = {id(layer) for task in model.tasks for layer in task.modules() if layer in walked}
    assert len(walked) - len(unique) == len(task_layers) > 0, "the duplicates are not exactly the task-head layers"

    new = float(model.quant_losses())
    assert new == pytest.approx(unique_sum(model), rel=1e-6)
    old = walked_sum(model)
    # measured on CLIC dummy events (offsets 0, 2, 4): new 3.13701 / 3.13733, old 3.41763 / 3.41794, +8.9445% to +8.9454%
    # (spread 1e-5; tolerance 2e-5). G2 on real events: 3.418 / 3.137437 - 1 = +8.94%
    assert new == pytest.approx(3.1370, abs=7e-4)

    # EBOPs at init of the production spec, the input of the beta setpoints (docs/hgq/BETA_SETPOINTS.md): every leaf's exact
    # (int64) EBOPs after one training forward of the seeded model: 250912214913 (unseeded builds read 2.50911e11 to
    # 2.50913e11: the WRAP datalanes fit their integer bits to the activations; docs/hgq/port_log/S5_grad_ratio.py reads
    # 250913099895 after three training forwards); 03_make_config.MODEL_EBOPS is the mean over 3 batches (2.511e11)
    total = C.total_ebops(model)
    assert total == pytest.approx(250912214913, rel=1e-6)
    assert make_config_constant("MODEL_EBOPS") == pytest.approx(total, rel=2e-3)
    assert (old - new) / new == pytest.approx(0.08945, abs=2e-5), (old, new)


def load_generator():
    spec = importlib.util.spec_from_file_location("make_golden_maskformer", GOLDEN / "make_golden_maskformer.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(GOLDEN))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(GOLDEN))
    return module


@pytest.mark.parametrize("case", ["mf_torch", "mf_org"])
@pytest.mark.parametrize("mode", ["high", "prod"])
def test_quant_losses_equals_the_golden_per_layer_terms(case, mode):
    """The generator's sequence: build, port the seeded torch weights, eval forward, training forward."""
    gen = load_generator()
    golden = dict(np.load(GOLDEN / f"{case}.npz"))
    meta = __import__("json").loads(str(golden["meta"]))
    variant = gen.CASES[case]
    tmodel = gen.torch_model(variant, seed=meta["seed"])
    assert gen.torch_state_digest(tmodel) == meta["torch_state_sha256"]
    inputs = {k.removeprefix("in::"): torch.from_numpy(v) for k, v in golden.items() if k.startswith("in::")}
    kmodel = gen.hybrid_model(variant, gen.MODES[mode])
    with torch.no_grad():
        kmodel(inputs)
    gen.port_maskformer(tmodel, kmodel)
    with torch.no_grad():
        kmodel.eval()(inputs)
        kmodel.train()(inputs)
    stored = golden[f"{mode}::qlosses"]
    assert float(kmodel.quant_losses()) == pytest.approx(float(stored.sum()), rel=1e-6)
    raw = meta[mode]["quant_losses"]["sum_raw"]
    assert walked_sum(kmodel) == pytest.approx(raw, rel=1e-6), "control: the old walk no longer reproduces the stored sum"
    assert raw > float(stored.sum()) * (1 + 1e-6) or mode == "high", (raw, float(stored.sum()))
