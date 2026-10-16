"""Float parity: hepattn Dense vs KerasDense on CLIC-exact shapes.

Discrimination: KerasDense is built by INTROSPECTING the torch module, so a wrong
hidden-size, activation, gate, or final-activation mapping produces O(1) output
errors, far above the fp32 tolerance asserted here.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import assert_parity, make_padded_batch  # ty: ignore [unresolved-import]

from hepattn.keras.dense import KerasDense
from hepattn.keras.porting import port_dense
from hepattn.models.dense import Dense

# CLIC-exact Dense configurations (see src/hepattn/experiments/clic/configs/base.yaml)
CONFIGS = {
    "clic_input_net_27_to_256": {"input_size": 27, "output_size": 256},
    "clic_encoder_ffn_256": {"input_size": 256},
    "clic_classification_head": {"input_size": 256, "output_size": 6, "hidden_layers": [256, 128, 32]},
    "clic_regression_head_518": {"input_size": 518, "output_size": 5, "hidden_layers": [512, 256, 128, 64, 32]},
    "swiglu_ffn": {"input_size": 64, "activation": "SwiGLU"},
    # org/main base.yaml SwiGLU nets: input net, encoder/decoder FFN, mask object/constituent nets, incidence node_net
    "clic_swiglu_input_net_27_to_256": {"input_size": 27, "output_size": 256, "activation": "SwiGLU"},
    "clic_swiglu_ffn_256": {"input_size": 256, "activation": "SwiGLU"},
    "clic_swiglu_incidence_node_net": {"input_size": 256, "hidden_layers": [256], "activation": "SwiGLU"},
    "final_activation_sigmoid": {"input_size": 32, "output_size": 8, "final_activation": torch.nn.Sigmoid()},
    "no_bias": {"input_size": 32, "bias": False},
    "with_dropout_eval": {"input_size": 32, "dropout": 0.1},
}


@pytest.mark.parametrize("name", CONFIGS)
def test_dense_parity(name):
    torch.manual_seed(42)
    cfg = CONFIGS[name]
    tdense = Dense(**cfg).eval()
    kdense = KerasDense.from_torch(tdense).eval()
    port_dense(tdense, kdense)

    x, _ = make_padded_batch(4, 21, cfg["input_size"], seed=7)
    with torch.no_grad():
        out_t = tdense(x)
    out_k = kdense(x)

    assert_parity("dense", name, out_t, out_k, atol=1e-6, rtol=1e-5)


def test_dense_introspection_is_discriminating():
    """from_torch must reproduce structure: a SwiGLU net splits the packed (64 -> 2*128) Linear into two halves."""
    tdense = Dense(input_size=64, activation="SwiGLU")
    assert tdense.net[0].weight.shape == (256, 64)  # torch packs [x1 | x2]
    kdense = KerasDense.from_torch(tdense)
    assert kdense.gate
    assert kdense.hidden[0].kernel.shape == (64, 128)  # x1
    assert kdense.gates[0].kernel.shape == (64, 128)  # x2, silu inside the layer
    assert kdense.gates[0].activation.__name__ == "silu"
    assert kdense.hidden[0].activation.__name__ == "linear"
    assert len(kdense.products) == 1
    tdense_plain = Dense(input_size=64)
    kdense_plain = KerasDense.from_torch(tdense_plain)
    assert not kdense_plain.gate
    assert kdense_plain.hidden[0].kernel.shape == (64, 128)
    assert len(kdense_plain.gates) == len(kdense_plain.products) == 0


def test_swiglu_halves_are_not_interchangeable():
    """Control for the swiglu rows: porting x1 into the silu half gives an O(1) error, far above their 1e-6 atol."""
    torch.manual_seed(42)
    tdense = Dense(input_size=256, activation="SwiGLU").eval()
    kdense = KerasDense.from_torch(tdense).eval()
    port_dense(tdense, kdense)
    x, _ = make_padded_batch(4, 21, 256, seed=7)
    with torch.no_grad():
        ref = tdense(x)
        kernel_x1, kernel_x2 = kdense.hidden[0].kernel.numpy().copy(), kdense.gates[0].kernel.numpy().copy()
        bias_x1, bias_x2 = kdense.hidden[0].bias.numpy().copy(), kdense.gates[0].bias.numpy().copy()
        kdense.hidden[0].kernel.assign(kernel_x2)
        kdense.hidden[0].bias.assign(bias_x2)
        kdense.gates[0].kernel.assign(kernel_x1)
        kdense.gates[0].bias.assign(bias_x1)
        swapped = (kdense(x) - ref).abs().max().item()
    assert swapped > 1e-2, swapped  # measured 3.05e-1 (R4a); the swiglu rows assert 1e-6 + 1e-5 * |ref|
