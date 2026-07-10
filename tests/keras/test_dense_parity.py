"""Float parity: hepattn Dense vs KerasDense on CLIC-exact shapes.

Discrimination: KerasDense is built by INTROSPECTING the torch module, so a wrong
hidden-size, activation, gate, or final-activation mapping produces O(1) output
errors, far above the fp32 tolerance asserted here.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import assert_parity, make_padded_batch

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
    """from_torch must reproduce structure: a SwiGLU net has doubled inner projections."""
    tdense = Dense(input_size=64, activation="SwiGLU")
    kdense = KerasDense.from_torch(tdense)
    assert kdense.gate
    assert kdense.hidden[0].kernel.shape == (64, 256)  # 128 * 2 for the gate
    tdense_plain = Dense(input_size=64)
    kdense_plain = KerasDense.from_torch(tdense_plain)
    assert not kdense_plain.gate
    assert kdense_plain.hidden[0].kernel.shape == (64, 128)
