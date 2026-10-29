"""Calibrated validation (``Trainer.calibrated_ntv``, ``val_cal/*``) on the tiny generated Polaris config, dummy events.

In training mode HGQ2 sets each WRAP quantizer's integer bits to a running maximum of the recent batches (decaying
``i_decay_speed`` per call), and eval mode uses the stored ones. ``calibrated_ntv`` traces them over the first N train
micro-batches instead, as ``hgq.utils.trace_minmax`` fixes them for a deployment:

- equal to HGQ2's own eager ``trace_minmax`` steps on the Keras model holding the same state, with the stored integer
  bits set to 20 and every lane signed first (what a huge recent training batch would leave): the trace must ignore them;
  both batches matter (each one alone gives other bits) and some lane comes out unsigned; ``self.state`` is unchanged;
  the returned EBOPs is HGQ2's stored per-leaf sum at the traced bits;
- a fit with ``val_calibration`` logs ``val_cal/*`` next to a ``val/*`` and a final state identical to a fit without it,
  and its last ``val_cal/loss`` is the validation of the final state at the traced bits (which differs from ``val/loss``).
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from hgq.utils.minmax_trace import TrainingFlagWrapper, _reset_minmax  # noqa: PLC2701  (what trace_minmax runs)

from hepattn.keras_native import loop

from .s6b_utils import differing, sources, tiny_config, trainer

N_CAL = 2


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    loop.enable_compilation_cache(tmp_path_factory.mktemp("jax_cache"))
    rc = tiny_config()
    train, val = sources(rc)
    t = trainer(rc, tmp_path_factory.mktemp("cal"), train)
    t._set_beta(1.0)  # noqa: SLF001  (penalty = EBOPs)
    i_pos, k_pos = t._wrap_index  # noqa: SLF001
    ntv = list(t.state.ntv)
    for p in i_pos:
        ntv[p] = jnp.full(ntv[p].shape, 20.0, ntv[p].dtype)
    for p in k_pos:
        ntv[p] = jnp.ones(ntv[p].shape, ntv[p].dtype)
    t.state = t.state._replace(ntv=ntv)
    return {"rc": rc, "train": train, "val": val, "t": t}


def hgq2_trace(t, batches) -> list[np.ndarray]:
    """The non-trainable variables after HGQ2's eager trace (reset, then one tracing call per batch) of ``t.state``."""
    model = t.steps.model
    t.steps.assign_to_model(t.state)
    _reset_minmax(model)
    for batch in batches:
        model(batch.inputs, training=TrainingFlagWrapper("tracing"))
    return [np.asarray(v.value) for v in model.non_trainable_variables]


def state_arrays(t) -> list[np.ndarray]:
    return [a for group in loop.host_state(t.state) for a in group]


def test_calibrated_ntv_is_hgq2_trace_minmax_and_keeps_the_state(cal):
    t, train = cal["t"], cal["train"]
    i_pos, k_pos = t._wrap_index  # noqa: SLF001
    assert len(set(i_pos)) == len(i_pos) == len(set(k_pos)) == len(k_pos) > 10
    before = state_arrays(t)
    ntv, ebops = t.calibrated_ntv(train, N_CAL)
    assert all(itertools.starmap(np.array_equal, zip(before, state_arrays(t), strict=True)))
    first, second = list(itertools.islice(train.epoch(0), N_CAL))
    both = hgq2_trace(t, [first, second])
    for p in i_pos + k_pos:
        np.testing.assert_array_equal(np.asarray(ntv[p]), both[p])
    # non-vacuous: the stored 20 / signed lanes are gone, each batch alone gives other integer bits, some lane is unsigned
    assert max(float(np.max(np.asarray(ntv[p]))) for p in i_pos) < 20
    for alone in (hgq2_trace(t, [first]), hgq2_trace(t, [second])):
        assert any(not np.array_equal(alone[p], both[p]) for p in i_pos)
    assert any((np.asarray(ntv[p]) == 0).any() for p in k_pos)
    # EBOPs at the traced bits: HGQ2's stored per-leaf values (uint32, truncated) after the eager trace; the read-out is
    # a float32 sum (measured: 75977880 vs the stored 75977883, 4e-8 relative)
    stored = sum(float(both[i]) for i in t.steps.ebops_leaves.ebops_index)
    assert stored > 0
    assert abs(ebops - stored) <= 1e-6 * stored + len(t.steps.ebops_leaves)


def test_a_fit_logs_val_cal_next_to_an_unchanged_val(cal, tmp_path):
    rc, train, val = cal["rc"], cal["train"], cal["val"]
    plain = trainer(rc, tmp_path / "plain", train, max_epochs=2)
    plain.fit(train, val)
    calibrated = trainer(rc, tmp_path / "cal", train, max_epochs=2, val_calibration=N_CAL)
    calibrated.fit(train, val)
    assert differing(dict(enumerate(state_arrays(plain))), dict(enumerate(state_arrays(calibrated)))) == []
    rows_p = loop.read_csv(tmp_path / "plain/metrics.csv")
    rows_c = loop.read_csv(tmp_path / "cal/metrics.csv")
    assert len(rows_p) == len(rows_c) == 2
    for p, c in zip(rows_p, rows_c, strict=True):
        for key in (k for k in p if not k.startswith("time/")):
            assert p[key] == c[key] or (np.isnan(p[key]) and np.isnan(c[key])), key
        assert np.isfinite([c["val_cal/loss"], c["val_cal/ebops"], c["val_cal/bits_i_mean"]]).all()
    ntv, ebops = calibrated.calibrated_ntv(train, N_CAL)
    again = calibrated.validate(val, state=calibrated.state._replace(ntv=ntv), stage="val_cal", ebops=ebops)
    assert again["val_cal/loss"] == rows_c[-1]["val_cal/loss"]
    assert again["val_cal/ebops"] == rows_c[-1]["val_cal/ebops"]
    assert any(c["val_cal/loss"] != c["val/loss"] for c in rows_c)
    assert calibrated.compile_counts()["calibrate"] == 1
