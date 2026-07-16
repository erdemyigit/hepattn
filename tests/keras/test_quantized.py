"""HGQ2 quantized mode: the paired non-vacuous checks.

test_high_bitwidth_close_to_float fails if the config scopes silently did not apply
(layers default-configured); test_low_bitwidth_differs fails if the quantizers are
inert under the torch backend (Q-layers degenerating to identity). Only the PAIR is
meaningful — either test alone could pass vacuously.

Comparisons use layer_0 task outputs: they are computed from the initial queries
before any mask-attention feedback, so quantization deltas are not amplified by
threshold flips in the self-referential decoder graph.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import record_measurement  # ty: ignore [unresolved-import]
from test_maskformer_parity import clic_dummy_batch, make_keras_model, make_torch_model  # ty: ignore [unresolved-import]

from hepattn.keras.porting import port_keras_to_keras, port_maskformer
from hepattn.keras.tasks import KerasIncidenceRegressionTask, KerasObjectHitMaskTask

HIGH_QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 20, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 8, "f0": 16},
    # softmax exp/inv table precision bounds attention accuracy independently of the above
    "table": {"default_q_type": "kif", "i0": 2, "f0": 16},
}
LOW_QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 4, "i0": 1},
    "datalane": {"default_q_type": "kif", "i0": 2, "f0": 2},
    "table": {"default_q_type": "kif", "i0": 1, "f0": 3},
}


def quantized_ported_model(quant: dict | None, seed: int = 50):
    tmodel = make_torch_model(seed)
    kmodel = make_keras_model(seed, quant=quant).eval()
    if quant is not None:
        # quantized layers build lazily from real static shapes; materialize before porting
        inputs, _ = clic_dummy_batch()
        with torch.no_grad():
            kmodel(inputs)
    port_maskformer(tmodel, kmodel)
    return kmodel.eval()


def layer0_logits(model, inputs) -> dict[str, torch.Tensor]:
    """Mask logits at layer_0, the only intermediate task output there.

    Computed from the initial queries before any mask-attention feedback (no
    threshold-flip amplification), covering input net + encoder + object net +
    (quantized) mask einsum.
    """
    with torch.no_grad():
        out = model(inputs)
    return {"mask_l0": out["layer_0"]["mask"]["pflow_node_logit"]}


def max_finite_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    finite = torch.isfinite(a) & torch.isfinite(b)
    return float((a[finite] - b[finite]).abs().max())


@pytest.fixture(scope="module")
def quant_comparison():
    inputs, _ = clic_dummy_batch()
    ref = layer0_logits(quantized_ported_model(None), inputs)
    high = layer0_logits(quantized_ported_model(HIGH_QUANT), inputs)
    low = layer0_logits(quantized_ported_model(LOW_QUANT), inputs)
    result = {name: (max_finite_diff(ref[name], high[name]), max_finite_diff(ref[name], low[name])) for name in ref}

    for name, (high_diff, low_diff) in result.items():
        for cfg, diff in (("high_bitwidth_w20_d24_t18", high_diff), ("low_bitwidth_w4_d4_t4", low_diff)):
            record_measurement({
                "tag": f"quantized.delta.{name}",
                "config": cfg,
                "max_abs_err": diff,
                "max_rel_err": float("nan"),
                "atol": 0.5,
                "rtol": 0.0,
            })
    return result


def test_quant_task_classes_swapped():
    kmodel = quantized_ported_model(HIGH_QUANT)
    task_types = {type(t) for t in kmodel.tasks}
    assert KerasObjectHitMaskTask in task_types, "mask task was not class-swapped to the quantized forward"
    assert KerasIncidenceRegressionTask in task_types, "incidence task was not class-swapped to the quantized forward"
    kfloat = quantized_ported_model(None)
    assert KerasObjectHitMaskTask not in {type(t) for t in kfloat.tasks}, "float mode must not swap task classes"


def test_high_bitwidth_close_to_float(quant_comparison):
    # Measured floor ~0.3 on logits of scale ~15 (2%): dominated by the quantized-softmax
    # chain (exp/inv LUTs) accumulated over the encoder+decoder pipeline. The pair with
    # test_low_bitwidth_differs is the discriminating criterion; bound calibrated to the
    # documented measurement (see PARITY.md quantized-deltas).
    for name, (high_diff, _) in quant_comparison.items():
        assert high_diff < 0.5, f"high-bitwidth quantized '{name}' deviates {high_diff:.3e} from float — scopes likely not applied"


def test_low_bitwidth_differs(quant_comparison):
    for name, (high_diff, low_diff) in quant_comparison.items():
        assert low_diff > max(10 * high_diff, 1.0), (
            f"low-bitwidth '{name}' differs from float by only {low_diff:.3e} (high-bitwidth: {high_diff:.3e}) — quantizers look inert"
        )


def test_ebops_positive_and_differentiable_per_region():
    kmodel = quantized_ported_model(HIGH_QUANT).train()
    inputs, _ = clic_dummy_batch()
    _ = kmodel(inputs)

    ebops = kmodel.quant_losses()
    assert float(ebops) > 0.0, "EBOPs regularization is zero after a training-mode forward"
    assert ebops.requires_grad, "EBOPs term is not differentiable"

    kmodel.zero_grad()
    ebops.backward()

    # NB: tasks are registered under both model.tasks and model.decoder.tasks (same
    # objects); named_parameters deduplicates to the first path, "decoder.tasks."
    for region in ("input_nets.", "encoder.", "decoder.decoder_layers.", "decoder.tasks."):
        grads = [name for name, p in kmodel.named_parameters() if name.startswith(region) and p.grad is not None and bool(torch.any(p.grad != 0))]
        assert grads, f"no parameter in region '{region}' received a gradient from the EBOPs term"


def test_float_weights_warm_start_quantized():
    """Float-trained weights must warm-start the quantized twin via port_keras_to_keras.

    Discriminating: the quantized model is built from a DIFFERENT seed, so if the port
    silently failed (name mismatch, unassigned kernels) the high-bitwidth outputs would
    differ from the float reference at O(1), far above the asserted bound.
    """
    kfloat = quantized_ported_model(None, seed=50)
    kquant = quantized_ported_model(HIGH_QUANT, seed=51)  # different init on purpose

    port_keras_to_keras(kfloat, kquant)

    inputs, _ = clic_dummy_batch()
    ref = layer0_logits(kfloat, inputs)
    warm = layer0_logits(kquant, inputs)
    diff = max_finite_diff(ref["mask_l0"], warm["mask_l0"])
    assert diff < 0.5, f"warm-started quantized model deviates {diff:.3e} from its float source"
