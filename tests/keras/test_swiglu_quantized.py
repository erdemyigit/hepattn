"""Quantized SwiGLU in the hybrid KerasDense (docs/hgq/RETARGET_ORG_MAIN.md section 5.4, lane R4a).

Leaf rule: x1 = QDense (no activation), silu(x2) = QDense(activation="silu"), x1 * silu(x2) = QEinsum ``bnc,bnc->bnc``.
The checks come as a set, because each one alone can pass vacuously:
- the paired delta: the high spec stays close to the torch float net, the low spec is at least 10x further away and
  still correlated with it (a low spec that zeroes every weight would also be "far away");
- liveness: switching off the input quantizer of any one of the three leaves changes the low-spec output;
- EBOPs: every leaf adds a positive, differentiable EBOPs term in training mode.

The high-spec floor (1.1e-2 to 1.3e-2) is the HGQ2 default bias quantizer, which the factory scopes do not set:
a plain SiLU Dense(256) has the same floor (1.31e-2, R4a log). That floor hides any SwiGLU error below about 1e-2, so
a second row zeroes the biases: the high-spec error then falls about 300x, to the floor of the three leaves.
"""

import pytest
import torch
from torch import nn

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import record_measurement  # ty: ignore [unresolved-import]
from test_quantized import HIGH_QUANT  # ty: ignore [unresolved-import]

from hepattn.keras.dense import KerasDense
from hepattn.keras.factory import LayerFactory
from hepattn.keras.porting import port_dense
from hepattn.models.dense import Dense

# Weights keep 5 fractional bits: a U(+-1/16) kernel (fan-in 256) keeps 75% of its entries nonzero. The shared
# LOW_QUANT of test_quantized.py (b0=4, i0=1) rounds every such weight to 0, so its output does not depend on x.
LOW_SWIGLU = {
    "weight": {"default_q_type": "kbi", "b0": 5, "i0": 0},
    "datalane": {"default_q_type": "kif", "i0": 2, "f0": 3},
    "table": {"default_q_type": "kif", "i0": 1, "f0": 3},
}
# org/main base.yaml SwiGLU nets: input net, encoder/decoder FFN (= mask nets), incidence node_net
CONFIGS = {
    "input_net_27_to_256": {"input_size": 27, "output_size": 256, "activation": "SwiGLU"},
    "ffn_256": {"input_size": 256, "activation": "SwiGLU"},
    "incidence_node_net": {"input_size": 256, "hidden_layers": [256], "activation": "SwiGLU"},
}
# Measured (R4a, macOS arm64, torch 2.10.0, CPU) over the 3 configs x seeds 0-2: high 1.12e-2 .. 1.30e-2; 2x.
HIGH_TOL = 2.6e-2
# The same with every bias set to 0 (R4a fix round, $S/r4a_fix/logs/swiglu_zero_bias.out): 2.96e-5 .. 4.27e-5; 2x.
ZERO_BIAS_TOL = 8.6e-5


def quantized_twin(tdense: Dense, quant: dict, x: torch.Tensor) -> KerasDense:
    factory = LayerFactory(quant)
    with factory.scopes():
        kdense = KerasDense.from_torch(tdense, factory=factory, name="ffn")
    kdense.eval()
    with torch.no_grad():
        kdense(x)  # quantized layers build lazily from the first input
    port_dense(tdense, kdense)
    return kdense


def fixture(name: str, seed: int = 0) -> tuple[Dense, torch.Tensor]:
    torch.manual_seed(seed)
    cfg = CONFIGS[name]
    x = torch.randn(2, 40, cfg["input_size"], generator=torch.Generator().manual_seed(seed + 10))
    return Dense(**cfg).eval(), x


def test_quantized_swiglu_leaf_classes():
    tdense, x = fixture("ffn_256")
    kdense = quantized_twin(tdense, HIGH_QUANT, x)
    names = [type(layer).__name__ for layer in (kdense.hidden[0], kdense.gates[0], kdense.products[0], kdense.final)]
    assert names == ["QDense", "QDense", "QEinsum", "QDense"]
    assert kdense.hidden[0].activation.__name__ == "linear"
    assert kdense.gates[0].activation.__name__ == "silu"


@pytest.mark.parametrize("name", CONFIGS)
def test_quantized_swiglu_paired_delta(name):
    tdense, x = fixture(name)
    with torch.no_grad():
        ref = tdense(x)
        high = quantized_twin(tdense, HIGH_QUANT, x)(x)
        low = quantized_twin(tdense, LOW_SWIGLU, x)(x)
    high_diff = (high - ref).abs().max().item()
    low_diff = (low - ref).abs().max().item()
    corr = torch.corrcoef(torch.stack([low.flatten(), ref.flatten()]))[0, 1].item()
    for cfg, diff in (("high_bitwidth", high_diff), ("low_swiglu", low_diff)):
        record_measurement({
            "tag": f"swiglu.quantized.{name}",
            "config": cfg,
            "max_abs_err": diff,
            "max_rel_err": float("nan"),
            "atol": HIGH_TOL,
            "rtol": 0.0,
        })
    assert high_diff <= HIGH_TOL, f"high spec deviates {high_diff:.3e} from float: scopes not applied?"
    assert low_diff >= 10 * high_diff, f"low spec only {low_diff:.3e} from float (high {high_diff:.3e}): quantizers inert?"
    assert corr > 0.5, f"low-spec output is not a quantized version of the float output (corr {corr:.3f})"


@pytest.mark.parametrize("name", CONFIGS)
def test_quantized_swiglu_high_spec_without_bias(name):
    """Without the bias-quantizer floor, the high spec resolves errors of the SwiGLU leaves down to about 1e-4."""
    tdense, x = fixture(name)
    with torch.no_grad():
        for module in tdense.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.zero_()
        ref = tdense(x)
        diff = (quantized_twin(tdense, HIGH_QUANT, x)(x) - ref).abs().max().item()
    record_measurement({
        "tag": f"swiglu.quantized.{name}",
        "config": "high_bitwidth_zero_bias",
        "max_abs_err": diff,
        "max_rel_err": float("nan"),
        "atol": ZERO_BIAS_TOL,
        "rtol": 0.0,
    })
    assert ref.abs().max().item() > 0.4  # not a vanishing output: the tolerance is about 2e-4 of it
    assert diff <= ZERO_BIAS_TOL, f"zero-bias high spec deviates {diff:.3e} from float"


@pytest.mark.parametrize("name", CONFIGS)
def test_quantized_swiglu_leaves_are_live(name):
    """Each leaf's input quantizer acts: bypassing it alone changes the low-spec output."""
    tdense, x = fixture(name)
    kdense = quantized_twin(tdense, LOW_SWIGLU, x)
    with torch.no_grad():
        base = kdense(x)
        for leaf in (kdense.hidden[0], kdense.gates[0], kdense.products[0]):
            leaf._enable_iq = False  # noqa: SLF001
            try:
                change = (kdense(x) - base).abs().max().item()
            finally:
                leaf._enable_iq = True  # noqa: SLF001
            # measured >= 5.86e-2 (R4a); float rounding alone cannot move the output by 1e-3
            assert change > 1e-3, f"input quantizer of {leaf.name} is inert (output change {change:.3e})"


def test_quantized_swiglu_ebops_per_leaf():
    tdense, x = fixture("ffn_256")
    kdense = quantized_twin(tdense, HIGH_QUANT, x).train()
    kdense(x)
    for leaf in (kdense.hidden[0], kdense.gates[0], kdense.products[0]):
        assert leaf.losses, f"{leaf.name} added no EBOPs loss"
        term = sum(leaf.losses)
        assert term.requires_grad, f"{leaf.name}: EBOPs term is not differentiable"
        assert float(term) / float(leaf.beta) > 0, f"{leaf.name}: zero EBOPs"
