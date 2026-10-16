"""effective_ebops: measure the EBOPs coefficient the training loss actually minimizes.

Background: the Polaris beta sweep was calibrated from `total_ebops` and came out inert
-- four decades of beta moved the mean learned bitwidth by 0.01%. `quant_losses()` is
affine in beta, and its slope was ~5 orders of magnitude below `total_ebops`. The cause
(found in R4b): beta was written only to the top-level keras layers, while most of the
total sits in the exp/inv tables nested inside every QSoftmax, which are HGQ2 leaves with
their own beta. Beta now reaches every leaf (callbacks.write_beta), and the slope is
`total_ebops` plus the task heads' EBOPs, which `quant_losses()` counts twice.

Discriminating properties, each with the wrong implementation it catches:
- affine: the fitted (slope, floor) predicts quant_loss at a *third*, unprobed beta --
  catches a function that returns plausible-looking but unrelated numbers.
- leaves: the slope equals the leaf total plus the task heads -- catches a beta probe that
  misses nested leaves (the old slope was ~1e-5 of this) or a total that misses them.
- restores: every leaf's beta is unchanged afterwards -- catches a probe that silently
  leaves the model perturbed, which would corrupt a training run that called it mid-flight.
- calibrates: beta derived as f*L/E_eff really does make the resource term f*L --
  catches sign and scale errors in the fit.
"""

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

from parity_utils import make_padded_batch  # noqa: F401  (ty: ignore) — pins test conftest paths
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore

from hepattn.keras import keras
from hepattn.keras.callbacks import BitwidthMonitor, leaf_ebops, write_beta
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
    return [float(keras.ops.convert_to_numpy(layer._beta)) for layer in model.quantized_leaves()]  # noqa: SLF001


def _restore(model, betas):
    for layer, beta in zip(model.quantized_leaves(), betas, strict=True):
        layer._beta.assign(beta)  # noqa: SLF001


def _quant_loss_at(model, inputs, beta):
    write_beta(model, beta)
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
    betas = _betas(model)
    measured = _quant_loss_at(model, inputs, probe)
    _restore(model, betas)

    assert predicted == pytest.approx(measured, rel=1e-3), f"not affine in beta: predicted {predicted:.6g} vs measured {measured:.6g}"


def test_slope_is_the_leaf_total_plus_the_task_heads(built):
    """E_eff = total_ebops + the task heads' EBOPs (quant_losses() counts those twice). Measured on this model
    ($S/r4b_fix/logs/evaluate_ebops2.out): relative 4.2e-8 (generated Polaris models: 9.6e-8, 2.2e-9), tolerance 2e-7.
    With beta on the top-level layers only, the slope was ~5 orders of magnitude below the total.
    """
    model, inputs = built
    e_eff, _ = effective_ebops(model, inputs)
    reported = total_ebops(model, inputs)
    leaves = {id(layer) for layer in model.quantized_leaves()}
    heads = sum(leaf_ebops(m) for m in {id(m): m for m in model.tasks.modules() if id(m) in leaves}.values())
    assert reported > 0.0
    assert heads > 0.0
    assert e_eff == pytest.approx(reported + heads, rel=2e-7)


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
    betas = _betas(model)
    measured = _quant_loss_at(model, inputs, beta) - floor
    _restore(model, betas)

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


def test_bitwidth_monitor_reports_all_three_families(built):
    """Logging only /b hides the parameter EBOPs actually depends on.

    Measured: -1 bit on /f moves total_ebops -54%, on /i -8%, on /b -0.00%. The first
    Polaris sweep watched /b move 8.00 -> 7.89 while val/ebops never budged, because /b
    is not what the reported cost is a function of.
    """
    model, _ = built
    pl = _StubModule(model)
    BitwidthMonitor().on_validation_epoch_end(None, pl)

    for suffix in ("b", "f", "i"):
        assert f"val/bits_{suffix}_mean" in pl.logged, f"/{suffix} family not logged — EBOPs depends on /f and /i"
        assert pl.logged[f"val/bits_{suffix}_n"] > 0
        assert pl.logged[f"val/bits_{suffix}_min"] <= pl.logged[f"val/bits_{suffix}_mean"] <= pl.logged[f"val/bits_{suffix}_max"]


def test_bitwidth_monitor_tracks_each_family_independently(built):
    """Perturbing one family must move only that family's readout."""
    model, _ = built
    pl = _StubModule(model)
    BitwidthMonitor().on_validation_epoch_end(None, pl)
    before = {s: pl.logged[f"val/bits_{s}_mean"] for s in ("b", "f", "i")}

    params = [p for n, p in model.named_parameters() if "quantizer" in n and p.requires_grad and n.endswith("/f")]
    with torch.no_grad():
        for p in params:
            p.sub_(1.0)
    try:
        BitwidthMonitor().on_validation_epoch_end(None, pl)
        assert pl.logged["val/bits_f_mean"] < before["f"], "/f readout did not follow a real change"
        assert pl.logged["val/bits_b_mean"] == pytest.approx(before["b"]), "/b moved when only /f was perturbed"
        assert pl.logged["val/bits_i_mean"] == pytest.approx(before["i"]), "/i moved when only /f was perturbed"
    finally:
        with torch.no_grad():
            for p in params:
                p.add_(1.0)
