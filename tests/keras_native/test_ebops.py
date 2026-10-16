"""S5 gates for ``keras_native.ebops`` (plan KERAS_JAX_PLAN.md 3.6).

- Read-out: ``penalty / beta`` on JAX equals the exact EBOPs the Keras TORCH backend stores (uint32 is int64 there), on the
  G1 spike graph (``polaris/14_backend_spike.py``, identical weights and inputs in both backends), in total and per leaf.
  Control: the JAX ``_ebops`` variables of the softmax exp tables read 4294967295, and their sum misses most of the total.
- The loss list of ``stateless_call``: every term before the regularizer tail is an EBOPs term (with EBOPs off exactly the
  regularizer terms remain, equal to the tail of the EBOPs-on list), so ``split_losses`` is exact.
- PeriodicEBOPs reproduces ``tests/keras/test_periodic_ebops.py`` on the hybrid's own golden model (``mf_torch``, prod spec,
  beta 1e-12): active 804.9105 vs skipped 0.33496 (the regularizer sum). The plan's 804.96 / 0.381 are the hybrid's
  pre-S5 double count; the deduplicated hybrid ``quant_losses()`` measures 804.9104 / 0.33496 on the same model.
- ``read_beta`` refuses mixed betas; ``ebops_readout`` refuses beta <= 0.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from hepattn.keras_native.ebops import UINT32_MAX, EbopsLeaves, ebops_readout

from .golden.cases import QUANT_SPECS
from .golden.mf_cases import clic_dummy_batch
from .mf_golden_utils import build_mf, build_mf_from_golden, mf_inputs, mf_meta

REPO = Path(__file__).resolve().parents[2]
SPIKE = REPO / "polaris" / "14_backend_spike.py"
BETA = 2.0**-40  # a power of two: ebops * beta and the division by beta are exact in float32

PROBE = r"""
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("spike", sys.argv[1])
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)
args = spike.parse_args(["--batch", "2"])
import keras
from keras import ops
keras.utils.set_random_seed(0)
x, m, y = spike.make_data(args, np)
model = spike.build_model(args, keras, ops)
model((x, m), training=False)
rng = np.random.default_rng(1)  # identical weights in both backends; quantizer bitwidths keep their init
for v in model.trainable_variables:
    if v.path.rsplit("/", 1)[-1] not in {"b", "i", "f"}:
        v.assign(rng.normal(scale=0.2, size=tuple(v.shape)).astype("float32"))
leaves = [l for l in model._flatten_layers(include_self=False) if getattr(l, "_beta", None) is not None]
for leaf in leaves:
    leaf._beta.assign(float(sys.argv[2]))
out = {"backend": keras.backend.backend(), "names": [l.name for l in leaves]}
if out["backend"] == "torch":
    model((x, m), training=True)
    out["stored"] = [int(ops.convert_to_numpy(l._ebops)) for l in leaves]
else:
    import jax
    from hepattn.keras_native.ebops import EbopsLeaves
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    _, ntv, losses = jax.jit(lambda tv, ntv: model.stateless_call(tv, ntv, (x, m), training=True, return_losses=True))(tv, ntv)
    el = EbopsLeaves(model)
    pen, reg = el.split_losses(losses)
    out["terms"] = [float(t) for t in pen]
    out["penalty_f32"] = float(sum(pen[1:], pen[0]))
    out["n_reg"] = len(reg)
    out["stored"] = [int(v) for v in el.stored_ebops(ntv)]
print("PROBE " + json.dumps(out))
"""


def run_probe(backend: str) -> dict:
    env = {**os.environ, "KERAS_BACKEND": backend, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run([sys.executable, "-c", PROBE, str(SPIKE), repr(BETA)], env=env, capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("PROBE "))
    return json.loads(line.removeprefix("PROBE "))


@pytest.fixture(scope="module")
def probes():
    return run_probe("jax"), run_probe("torch")


# measured on macOS arm64 (S5.md): 86 leaves, total 2.123e14; read-out rel 5.9e-7 (float32 sum), per-leaf max |diff| 0.96
READOUT_RTOL = 1.2e-6  # 2x measured


def test_readout_equals_the_torch_backend_value(probes):
    """JAX ``penalty / beta`` vs the sum of the torch backend's exact per-leaf ``_ebops`` (int64).

    The penalty is summed in float32 as in the train step, so the total carries float32 rounding; per leaf the read-out
    is exact (beta is a power of two) up to the int cast of the float32 EBOPs (< 1).
    """
    jx, th = probes
    assert jx["names"] == th["names"]
    exact = np.array(th["stored"], dtype=np.float64)
    readout = ebops_readout(jx["penalty_f32"], BETA)
    assert abs(readout - exact.sum()) <= READOUT_RTOL * exact.sum(), (readout, exact.sum())
    per_leaf = np.sort(np.array(jx["terms"], dtype=np.float64) / BETA)
    assert per_leaf.size == exact.size
    assert np.abs(per_leaf - np.sort(exact)).max() < 1.0


def test_jax_stored_ebops_saturate_uint32_control(probes):
    jx, th = probes
    exact = np.array(th["stored"], dtype=np.float64)
    stored = np.array(jx["stored"], dtype=np.float64)
    saturated = stored == UINT32_MAX
    assert saturated.sum() >= 1 and exact.max() > UINT32_MAX, "no leaf above the uint32 range: the control is vacuous"
    assert np.array_equal(saturated, exact > UINT32_MAX), "saturation is not exactly the leaves above 2**32 - 1"
    assert stored.sum() < 0.5 * exact.sum(), "reading _ebops would have been close to the truth"
    assert np.array_equal(stored[~saturated], exact[~saturated]), "the unsaturated leaves differ across backends"


@pytest.fixture(scope="module")
def org_model():
    inputs, _ = clic_dummy_batch(num_events=2)
    return build_mf("org", QUANT_SPECS["high"], inputs), inputs


def test_loss_list_is_ebops_terms_then_regularizers(org_model):
    model, inputs = org_model
    el = EbopsLeaves(model)
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]

    def losses(on):
        with el.enabled(on):
            return jax.jit(lambda tv, ntv: model.stateless_call(tv, ntv, inputs, training=True, return_losses=True)[2])(tv, ntv)

    on, off = losses(True), losses(False)
    assert el.n_regularizers > 0, "no regularized weights: the tail split is untested"
    assert len(off) == el.n_regularizers
    pen, reg = el.split_losses(on)
    np.testing.assert_array_equal(np.array(reg), np.array(off))
    assert len(pen) == len(el) == len({id(leaf) for leaf in el.leaves}), (len(pen), len(el))
    assert all(float(t) > 0 for t in pen), "a term in front of the regularizers is not an EBOPs term"
    assert [leaf._enable_ebops for leaf in el.leaves] == [True] * len(el), "enabled() did not restore the flags"  # noqa: SLF001


def test_periodic_ebops_reproduces_the_hybrid_test(org_model):
    """``test_periodic_ebops.py`` (hybrid dim-32 model, prod spec, beta 1e-12) on its golden twin (``mf_torch``/prod, the
    hybrid's weights and quantizer state). Measured with the hybrid (S5.md): the old double-counting walk 804.9605 active /
    0.38085 skipped (the plan's 804.96 / 0.381); the deduplicated ``quant_losses()`` 804.9104 / 0.33496. Native: 804.9105 /
    0.33496 (relative differences 1.0e-7 and 4e-8; tolerances 2e-7 and 1e-7).
    """
    del org_model
    model, golden = build_mf_from_golden("mf_torch", "prod")
    el = EbopsLeaves(model)
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]

    def total(on):
        with el.enabled(on):
            _, _, losses = model.stateless_call(tv, ntv, mf_inputs(golden), training=True, return_losses=True)
        pen, reg = el.split_losses(losses)
        return float(sum(float(x) for x in pen)), float(sum(float(x) for x in reg))

    pen_on, reg_on = total(True)
    pen_off, reg_off = total(False)
    unique = float(np.sum(golden["prod::qlosses"]))
    assert pen_on + reg_on == pytest.approx(unique, rel=1e-5), (pen_on + reg_on, unique)
    assert mf_meta(golden)["prod"]["quant_losses"]["sum_raw"] - unique > 0.01, "the golden no longer records the double count"
    assert pen_off == 0.0 and reg_off == pytest.approx(reg_on, rel=1e-6)
    assert pen_on + reg_on == pytest.approx(804.910400390625, rel=2e-7)  # the deduplicated hybrid value
    assert reg_off == pytest.approx(0.33495965600013733, rel=1e-7)
    assert reg_off < 0.01 * (pen_on + reg_on)  # the hybrid test's own criterion (skipped < 1% of active)


def test_beta_helpers_refuse_invalid_states(org_model):
    model, _ = org_model
    el = EbopsLeaves(model)
    ntv = el.set_beta([v.value for v in model.non_trainable_variables], 3e-9)
    assert el.read_beta(ntv) == pytest.approx(3e-9, rel=1e-6)
    mixed = list(ntv)
    mixed[el.beta_index[0]] = mixed[el.beta_index[0]] * 2
    with pytest.raises(ValueError, match="different betas"):
        el.read_beta(mixed)
    for beta in (0.0, -1e-9):
        with pytest.raises(ValueError, match="beta > 0"):
            ebops_readout(1.0, beta)
