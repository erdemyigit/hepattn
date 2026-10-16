"""The torch-free helper copies in keras_native equal their torch originals (run in the same process)."""

import itertools
import json
import os
import subprocess
import sys

import keras
import numpy as np
import pytest
import torch

from hepattn.keras_native import factory as native_factory
from hepattn.keras_native import norm as native_norm
from hepattn.keras_native import scaling as native_scaling
from hepattn.models import norm as torch_norm
from hepattn.utils import scaling as torch_scaling

from .conftest import REPO_ROOT

CLIC_SCALE_DICT = REPO_ROOT / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml"
TRANSFORM_CONFIGS = [
    {"type": t, "fn": fn, "mean": 0.3, "std": 1.7, "min": 0.2, "max": 9.0}
    for t, fn in itertools.product(["std", "min_max", "min_max_sym"], [None, "log", "log1p", "sqrt"])
]


@pytest.mark.parametrize(
    ("norm", "depth", "hybrid_norm", "qkv_norm", "placement"),
    itertools.product([None, "LayerNorm", "FastLayerNorm"], [0, 1, 3], [False, True], [False, True], ["hybridnorm", "legacy"]),
)
def test_hybrid_norm_config_equals_torch(norm, depth, hybrid_norm, qkv_norm, placement):
    args = (norm, depth, hybrid_norm, qkv_norm, placement)
    assert native_norm.get_hybrid_norm_config(*args) == torch_norm.get_hybrid_norm_config(*args)


def test_dense_norm_placement_default_and_unknown():
    """The default is "hybridnorm" on both sides, and the placements differ where they should (non-vacuous rows)."""
    assert native_norm.get_hybrid_norm_config("LayerNorm", 0, True, False) == torch_norm.get_hybrid_norm_config("LayerNorm", 0, True, False)
    legacy = [native_norm.get_hybrid_norm_config("LayerNorm", d, True, False, "legacy")[1] for d in (0, 1)]
    hybrid = [native_norm.get_hybrid_norm_config("LayerNorm", d, True, False, "hybridnorm")[1] for d in (0, 1)]
    assert legacy == [True, False] and hybrid == [False, True]
    for impl in (native_norm, torch_norm):
        with pytest.raises(ValueError, match="dense_norm_placement"):
            impl.get_hybrid_norm_config("LayerNorm", 0, True, False, "pre")


@pytest.mark.parametrize("config", TRANSFORM_CONFIGS, ids=lambda c: f"{c['type']}-{c['fn']}")
def test_var_transform_equals_torch(config):
    x = np.random.default_rng(0).uniform(0.5, 50.0, size=(4, 160)).astype(np.float32)
    ref = torch_scaling.VarTransform("v", config)
    nat = native_scaling.VarTransform("v", config)
    fwd = keras.ops.convert_to_numpy(nat.transform(x))
    np.testing.assert_allclose(fwd, ref.transform(torch.from_numpy(x)).numpy(), rtol=1e-6, atol=1e-6)
    inv = keras.ops.convert_to_numpy(nat.inverse_transform(fwd))
    np.testing.assert_allclose(inv, ref.inverse_transform(torch.from_numpy(fwd)).numpy(), rtol=1e-6, atol=1e-6)
    # Round trip; also proves the transform is not the identity, so the comparisons above discriminate.
    np.testing.assert_allclose(inv, x, rtol=1e-4)
    assert not np.allclose(fwd, x)


def test_feature_scaler_clic_dict_equals_torch():
    ref = torch_scaling.FeatureScaler(str(CLIC_SCALE_DICT))
    nat = native_scaling.FeatureScaler(CLIC_SCALE_DICT)
    assert set(nat.transforms) == set(ref.transforms)
    rng = np.random.default_rng(1)
    feats = {name: rng.uniform(0.5, 50.0, size=(2, 160)).astype(np.float32) for name in nat.transforms}
    feats["not_scaled"] = rng.normal(size=(2, 160)).astype(np.float32)
    original = {k: v.copy() for k, v in feats.items()}
    out = nat.transform(feats)
    ref_out = ref.transform({k: torch.from_numpy(v.copy()) for k, v in feats.items()})
    for name, value in out.items():
        np.testing.assert_allclose(keras.ops.convert_to_numpy(value), ref_out[name].numpy(), rtol=1e-6, atol=1e-6, err_msg=name)
    # The native scaler is functional: the input dict is untouched.
    for name, value in feats.items():
        np.testing.assert_array_equal(value, original[name])
    back = nat.inverse_transform(out)
    for name, value in back.items():
        np.testing.assert_allclose(keras.ops.convert_to_numpy(value), original[name], rtol=1e-4, err_msg=name)


@pytest.mark.parametrize("bad", [{"type": "robust"}, {"type": "std", "fn": "exp"}])
def test_unsupported_transform_raises(bad):
    with pytest.raises(ValueError, match="Unknown scaling"):
        native_scaling.VarTransform("v", bad)


SHIM_CHECK = """
import json, hepattn.keras.factory as shim, hepattn.keras_native.factory as nat, keras
public = [n for n, o in vars(nat).items() if not n.startswith("_") and getattr(o, "__module__", None) == nat.__name__]
print(json.dumps({"backend": keras.backend.backend(), "public": public,
                  "missing": [n for n in public if getattr(shim, n, None) is not getattr(nat, n)]}))
"""


def test_hybrid_factory_shim_reexports_every_public_name():
    """Under the torch backend, hepattn.keras.factory exposes the very objects of keras_native.factory."""
    env = {**os.environ, "KERAS_BACKEND": "torch", "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run([sys.executable, "-c", SHIM_CHECK], env=env, capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["backend"] == "torch"
    assert {"EinsumOp", "LayerFactory", "QuantSpec", "SoftmaxOp", "apply_softmax"} <= set(out["public"])
    assert out["missing"] == []


@pytest.mark.parametrize("quant", [None, {"ebops": {"beta0": 1e-5}}], ids=["float", "hgq"])
def test_factory_builds_and_runs_under_jax(quant):
    factory = native_factory.LayerFactory(quant)
    with factory.scopes():
        layer = factory.dense(8, name="probe_dense")
    x = np.random.default_rng(2).normal(size=(2, 5, 4)).astype(np.float32)
    y = layer(x, training=True)
    assert tuple(y.shape) == (2, 5, 8)
    assert type(layer).__module__.startswith("hgq." if quant else "keras.")
    if quant:
        # EBOPs x beta plus HGQ2's bitwidth regularizers; the float twin has none.
        assert sum(float(keras.ops.convert_to_numpy(v)) for v in layer.losses) > 0
    else:
        assert layer.losses == []
