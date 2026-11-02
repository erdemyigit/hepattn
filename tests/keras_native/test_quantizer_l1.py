"""``lrs_config.quantizer_l1``: the strength of HGQ2's bitwidth regularizer (``QuantSpec.bit_l1``), tiny Polaris config.

HGQ2 0.1.9 puts ``MonoL1(1e-8)`` (loss ``1e-8 x sum(bits)``) on the KBI ``b``, KIF ``i`` / ``f`` and float ``m`` / ``e``
of every quantizer through its config defaults (``br`` / ``ir`` / ``fr`` / ``mr`` / ``er``). The knob replaces exactly
these: the same variables carry ``MonoL1(l1)`` (none at 0), the regularization loss scales by ``l1 / 1e-8`` at equal
weights, the forward and the model hash do not change.
"""

import copy
import dataclasses

import numpy as np
import pytest
import yaml

from hepattn.experiments.clic.convert_to_native import sample_batch
from hepattn.keras_native.config import ConfigError, load_config

from .s6b_utils import TINY

HGQ2_DEFAULT_L1 = 1e-8


def config_with(tmp_path, l1):
    raw = yaml.safe_load(TINY.read_text())
    if l1 is not None:
        raw["model"]["lrs_config"]["quantizer_l1"] = l1
    path = tmp_path / f"l1_{l1}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config([str(path)])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("l1")
    out = {}
    inputs = None
    for l1 in (None, 3e-7, 0.0):
        rc = config_with(tmp, l1)
        inputs = inputs if inputs is not None else sample_batch(rc, 2)[0]
        model = rc.build_model()
        model(inputs, training=False)
        out[l1] = (rc, model)
    return out, inputs


def regularized(model) -> dict[int, float]:
    """Position in ``trainable_weights`` (the variable paths carry global name counters) -> MonoL1 strength."""
    return {n: w.regularizer.l1 for n, w in enumerate(model.trainable_weights) if w.regularizer is not None}


def reg_loss(model) -> float:
    return float(sum(np.asarray(w.regularizer(w.value)) for w in model.trainable_weights if w.regularizer is not None))


def test_the_knob_rescales_hgq2_bit_regularizers_only(built):
    models, inputs = built
    _, m0 = models[None]
    default = regularized(m0)
    # what HGQ2 regularizes by default: kbi b of the weights, kif i and f of the datalanes
    assert len(default) > 50 and set(default.values()) == {HGQ2_DEFAULT_L1}
    paths = [m0.trainable_weights[n].path for n in default]
    assert all(any(p.endswith(f"/{bit}") for p in paths) for bit in ("b", "i", "f"))
    _, m1 = models[3e-7]
    scaled = regularized(m1)
    assert scaled.keys() == default.keys()  # no regularizer added (e.g. on KBI i) or dropped
    assert set(scaled.values()) == {3e-7}
    m1.set_weights(m0.get_weights())
    assert reg_loss(m1) == pytest.approx(30 * reg_loss(m0), rel=1e-5)  # float32 sums (measured 1.3e-6)
    assert reg_loss(m0) > 0
    a, b = m0(inputs, training=False), m1(inputs, training=False)
    flat_a, flat_b = [np.asarray(x) for x in _leaves(a)], [np.asarray(x) for x in _leaves(b)]
    assert len(flat_a) == len(flat_b) > 5
    for x, y in zip(flat_a, flat_b, strict=True):
        np.testing.assert_array_equal(x, y)
    _, m_off = models[0.0]
    assert regularized(m_off) == {}


def test_the_knob_keeps_the_model_hash(built):
    models, _ = built
    hashes = {l1: rc.model_hash for l1, (rc, _) in models.items()}
    assert len(set(hashes.values())) == 1, hashes
    assert models[3e-7][0].optimizer.quantizer_l1 == 3e-7 and models[None][0].optimizer.quantizer_l1 is None


def test_the_knob_refuses_a_float_model_and_a_double_setting(built):
    models, _ = built
    rc, _ = models[3e-7]
    section = copy.deepcopy(rc.model_section)
    section["model"]["init_args"].pop("quant")
    with pytest.raises(ConfigError, match="quantized model"):
        dataclasses.replace(rc, model_section=section).build_model()
    both = copy.deepcopy(rc.model_section)
    both["model"]["init_args"]["quant"]["bit_l1"] = 1e-6
    with pytest.raises(ConfigError, match="once"):
        dataclasses.replace(rc, model_section=both).build_model()


def _leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _leaves(value)
    else:
        yield tree
