"""effective_ebops: measure the EBOPs coefficient the training loss actually minimizes.

Background: the Polaris beta sweep was calibrated from `total_ebops` and came out inert
-- four decades of beta moved the mean learned bitwidth by 0.01%. Cause: `quant_losses()`
is affine in beta with a slope that is ~5 orders of magnitude below `total_ebops`, because
most of the reported total comes from QSoftmax layers whose cost never enters the
regularization loss. `effective_ebops` measures the slope instead.

Discriminating properties, each with the wrong implementation it catches:
- affine: the fitted (slope, floor) predicts quant_loss at a *third*, unprobed beta --
  catches a function that returns plausible-looking but unrelated numbers.
- distinct: the slope is orders of magnitude below `total_ebops` -- catches "fixing" this
  by aliasing it to total_ebops, which is exactly the bug that caused the inert sweep.
- restores: layer betas are unchanged afterwards -- catches a probe that silently leaves
  the model perturbed, which would corrupt a training run that called it mid-flight.
- calibrates: beta derived as f*L/E_eff really does make the resource term f*L --
  catches sign and scale errors in the fit.
"""

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

from parity_utils import make_padded_batch  # noqa: F401  (ty: ignore) — pins test conftest paths
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore

from hepattn.keras import keras
from hepattn.keras.callbacks import BitwidthMonitor
from hepattn.keras.evaluate import effective_ebops, total_ebops

QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0e-12},
}


@pytest.fixture(scope="module")
def built():
    model = make_keras_model(seed=0, quant=QUANT)
    inputs, _ = clic_dummy_batch(2)
    with torch.no_grad():
        model.eval()(inputs)  # quantized layers build lazily at first forward
    model.train()
    return model, inputs


def _betas(model):
    return [float(keras.ops.convert_to_numpy(layer._beta)) for layer in model.keras_layers() if getattr(layer, "_beta", None) is not None]  # noqa: SLF001


def _quant_loss_at(model, inputs, beta):
    layers = [layer for layer in model.keras_layers() if getattr(layer, "_beta", None) is not None]
    for layer in layers:
        layer._beta.assign(beta)  # noqa: SLF001
    model.train()
    with torch.no_grad():
        model(inputs)
        return float(model.quant_losses())


def test_affine_prediction_holds_at_an_unprobed_beta(built):
    """quant_loss(beta) = floor + beta*E_eff must hold at a beta the fit never saw."""
    model, inputs = built
    e_eff, floor = effective_ebops(model, inputs, probes=(1e-9, 1e-6))
    assert e_eff > 0.0, "slope must be positive: more beta means more resource penalty"

    probe = 1e-7  # strictly between the two fitted points, and not one of them
    predicted = floor + probe * e_eff
    measured = _quant_loss_at(model, inputs, probe)
    for layer in model.keras_layers():  # restore
        if getattr(layer, "_beta", None) is not None:
            layer._beta.assign(QUANT["ebops"]["beta0"])  # noqa: SLF001

    assert predicted == pytest.approx(measured, rel=1e-3), f"not affine in beta: predicted {predicted:.6g} vs measured {measured:.6g}"


def test_slope_is_far_below_total_ebops(built):
    """The two must not be conflated — treating them as equal is the original bug."""
    model, inputs = built
    e_eff, _ = effective_ebops(model, inputs)
    reported = total_ebops(model, inputs)
    assert reported > 0.0
    assert e_eff < reported / 1e3, (
        f"effective slope {e_eff:.4g} is within 1000x of total_ebops {reported:.4g} — "
        "if these ever agree, the beta-calibration rationale needs revisiting"
    )


def test_restores_layer_betas(built):
    """A probe that leaves beta perturbed would silently corrupt an in-flight run."""
    model, inputs = built
    before = _betas(model)
    effective_ebops(model, inputs, probes=(1e-9, 1e-6))
    assert _betas(model) == pytest.approx(before, rel=0, abs=0), "betas not restored after probing"


def test_calibration_round_trip(built):
    """Beta = f*L/E_eff must actually make the resource term f*L above the floor."""
    model, inputs = built
    e_eff, floor = effective_ebops(model, inputs)

    task_loss, fraction = 30.0, 0.25
    beta = fraction * task_loss / e_eff
    measured = _quant_loss_at(model, inputs, beta) - floor
    for layer in model.keras_layers():  # restore
        if getattr(layer, "_beta", None) is not None:
            layer._beta.assign(QUANT["ebops"]["beta0"])  # noqa: SLF001

    assert measured == pytest.approx(fraction * task_loss, rel=1e-2), f"calibration off: wanted {fraction * task_loss:.4g}, got {measured:.4g}"


def test_rejects_equal_probes(built):
    model, inputs = built
    with pytest.raises(ValueError, match="probes must differ"):
        effective_ebops(model, inputs, probes=(1e-9, 1e-9))


class _StubModule:
    """Minimal LightningModule stand-in: BitwidthMonitor only needs .model and .log."""

    def __init__(self, model):
        self.model = model
        self.logged: dict[str, float] = {}

    def log(self, name, value, **_):
        self.logged[name] = float(value)


def test_bitwidth_monitor_reports_real_layer_count(built):
    """A monitor that finds zero bitwidth params would log nothing and look 'fine'."""
    model, _ = built
    pl = _StubModule(model)
    BitwidthMonitor().on_validation_epoch_end(None, pl)
    assert pl.logged, "BitwidthMonitor logged nothing — it found no '/b' parameters"
    assert pl.logged["val/bits_n_layers"] > 0
    assert pl.logged["val/bits_min"] <= pl.logged["val/bits_mean"] <= pl.logged["val/bits_max"]


def test_bitwidth_monitor_tracks_a_change(built):
    """Catches a monitor that reports a constant (e.g. reads b0 config, not the variable)."""
    model, _ = built
    pl = _StubModule(model)
    BitwidthMonitor().on_validation_epoch_end(None, pl)
    before = pl.logged["val/bits_mean"]

    params = [p for name, p in model.named_parameters() if name.endswith("/b")]
    with torch.no_grad():
        params[0].sub_(1.0)
    try:
        BitwidthMonitor().on_validation_epoch_end(None, pl)
        assert pl.logged["val/bits_mean"] < before, "mean bitwidth did not follow a real change"
    finally:
        with torch.no_grad():
            params[0].add_(1.0)
