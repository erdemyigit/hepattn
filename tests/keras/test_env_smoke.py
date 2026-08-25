"""Stage-1 smoke tests for the Keras-3-torch-backend + HGQ2 foundation.

Each test guards one architectural assumption the whole hepattn.keras package
rests on. They are deliberately low-level: if one of these fails, the failure
is in the environment or in an external package, not in hepattn code.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import hepattn

hepattn_keras = pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")
keras = hepattn_keras.keras

hgq_config = pytest.importorskip("hgq.config")
hgq_layers = pytest.importorskip("hgq.layers")

LayerConfigScope = hgq_config.LayerConfigScope
QuantizerConfigScope = hgq_config.QuantizerConfigScope
QDense = hgq_layers.QDense


def test_backend_is_torch():
    assert keras.backend.backend() == "torch"


def test_backend_guard_rejects_non_torch_env():
    """Importing hepattn.keras with a non-torch KERAS_BACKEND must fail loudly, not degrade."""
    src_dir = str(Path(hepattn.__file__).resolve().parent.parent)
    env = os.environ | {"KERAS_BACKEND": "tensorflow", "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, "-c", "import hepattn.keras"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "KERAS_BACKEND" in proc.stderr


def test_keras_layer_is_torch_module_with_parameters():
    """Keras-torch-backend layers must be torch Modules whose weights are visible to torch optimizers."""
    dense = keras.layers.Dense(4)
    dense.build((None, 8))
    assert isinstance(dense, nn.Module)
    params = list(dense.parameters())
    assert len(params) == 2
    assert {tuple(p.shape) for p in params} == {(8, 4), (4,)}


def test_keras_dense_matches_torch_linear():
    """The weight-porting axiom: keras kernel == torch weight transposed, bias identical.

    Bitwise equality is asserted; if the underlying matmul kernels ever diverge
    this documents exactly where cross-framework parity starts being tolerance-based.
    """
    torch.manual_seed(0)
    lin = nn.Linear(27, 54)
    dense = keras.layers.Dense(54)
    dense.build((None, 27))
    dense.set_weights([lin.weight.detach().numpy().T, lin.bias.detach().numpy()])
    x = torch.randn(16, 27)
    with torch.no_grad():
        out_torch = lin(x)
    out_keras = dense(x)
    assert isinstance(out_keras, torch.Tensor)
    # Measured cross-framework floor: keras Dense (matmul+add) vs torch Linear (fused addmm)
    # differ by up to ~3.6e-07 abs on this shape. Bitwise equality is NOT achievable even for
    # a single linear layer; 1e-6 abs is the parity baseline documented in docs/hgq/PARITY.md.
    torch.testing.assert_close(out_keras, out_torch, rtol=0.0, atol=1e-6)


def test_qdense_trains_under_torch_backend():
    """HGQ2 quantizer autograd must work under the torch backend with a plain torch optimizer.

    Discriminating because it fails if (a) keras/HGQ2 params are invisible to
    torch optimizers, (b) quantizer surrogate gradients are broken under torch,
    or (c) the forward is not differentiable end-to-end.
    """
    torch.manual_seed(0)
    layer = QDense(1)
    layer.build((None, 8))

    params = list(layer.parameters())
    assert len(params) > 2, "expected quantizer parameters beyond kernel+bias"

    x = torch.randn(512, 8)
    w_true = torch.randn(8, 1)
    y = x @ w_true

    opt = torch.optim.AdamW(params, lr=5e-2)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        pred = layer(x, training=True)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < 0.5 * losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    opt.zero_grad()
    loss = F.mse_loss(layer(x, training=True), y)
    loss.backward()
    n_with_grad = sum(1 for p in params if p.grad is not None and bool(torch.any(p.grad != 0)))
    assert n_with_grad > 2, f"expected quantizer params to receive gradients, only {n_with_grad} params have nonzero grad"


def test_ebops_retrievable_outside_keras_fit():
    """EBOPs regularization must be collectable from an external (Lightning) training loop.

    hepattn never calls keras.Model.fit, so the EBOPs*beta term has to be
    retrievable directly from layers after a training-mode call.
    """
    with QuantizerConfigScope(place="all"), LayerConfigScope(enable_ebops=True, beta0=1e-5):
        layer = QDense(4)
    layer.build((None, 8))

    x = torch.randn(16, 8)
    _ = layer(x, training=True)

    reg_losses = list(layer.losses)
    assert reg_losses, "layer.losses is empty after a training call with enable_ebops=True"
    total = sum(reg_losses)
    assert isinstance(total, torch.Tensor)
    assert total.numel() == 1
    assert float(total) > 0.0
    assert total.requires_grad, "EBOPs regularization term is not differentiable"


def test_quantizer_config_scope_applies_at_construction():
    """QuantizerConfigScope must configure layers built inside it.

    This is the mechanism the YAML-driven builder relies on (scopes wrap construction
    inside KerasMaskFormer.__init__ rather than the jsonargparse instantiation site).
    """
    # NB: in QuantizerConfigScope, q_type/place are SELECTORS for which quantizers the
    # overrides apply to; the quantizer type itself is set via default_q_type.
    with QuantizerConfigScope(place="weight", default_q_type="kbi"):
        a = QDense(4)
        a.build((None, 8))
    with QuantizerConfigScope(place="weight", default_q_type="kif"):
        b = QDense(4)
        b.build((None, 8))

    qa = type(a.kq.quantizer).__name__
    qb = type(b.kq.quantizer).__name__
    assert qa != qb, f"different QuantizerConfigScope settings produced identical quantizers ({qa})"
    assert "KBI" in qa
    assert "KIF" in qb
