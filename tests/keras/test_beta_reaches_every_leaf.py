"""BetaScheduler, BetaPID and EBOPsMonitor act on every HGQ2 leaf, also the nested ones.

Every HGQ2 layer adds ``beta * ebops`` with its OWN beta variable. The callbacks used to walk only the top-level keras
layers, which misses two kinds of nested leaves (measured on the generated Polaris model, $S/r4b/logs/beta_walk.out):
- the exp/inv tables inside every QSoftmax: in the dense-attention model 38 of 257 leaves, but 99.99% of its EBOPs
  (the masked-softmax fill term), so beta never reached the term that dominates the cost;
- every leaf of the masked-Linformer cores (R4b): in the org Linformer model 212 of 377 leaves, 26% of the EBOPs.
The discriminating check: with beta written by the callback, the EBOPs loss of a training forward grows by exactly
``delta beta * total EBOPs`` when beta changes, which holds only if every leaf took the new beta and the total counts
every leaf once.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import org_model_utils as U  # ty: ignore [unresolved-import]

from hepattn.keras.callbacks import BetaPID, BetaScheduler, EBOPsMonitor, total_ebops


class Trainer:
    global_step = 10**9  # past the BetaScheduler warmup: it writes beta_end
    current_epoch = 0
    callbacks = ()


class Module:
    def __init__(self, model):
        self.model = model
        self.logged: dict[str, float] = {}

    def log(self, key, value, **_):
        self.logged[key] = float(value)


@pytest.fixture(scope="module", params=["linformer", "dense"])
def model(request, tmp_path_factory):
    cfg = U.generate_polaris_config(tmp_path_factory.mktemp("gen") / "polaris_hgq.yaml")
    if request.param == "dense":
        cfg = U.to_dense_attention(cfg)
    torch.manual_seed(0)
    model = U.instantiate(cfg, quantized=True)
    inputs, _ = U.clic_dummy_batch()
    with torch.no_grad():
        model(inputs)
    model.train()
    return request.param, model, inputs


def top_level(model) -> list:
    """keras_layers() once per layer: it yields the task heads twice (tasks and decoder.tasks, port_log/S3.md
    deviation 9), so quant_losses() counts their terms twice; the checks below count every layer once.
    """
    return list({id(layer): layer for layer in model.keras_layers()}.values())


def ebops_loss(model, inputs) -> float:
    with torch.no_grad():
        model(inputs)
    return sum(float(term) for layer in top_level(model) for term in layer.losses)


def test_quantized_leaves_include_the_nested_ones(model):
    attn, kmodel, _ = model
    leaves = list(kmodel.quantized_leaves())
    top = [layer for layer in top_level(kmodel) if getattr(layer, "_beta", None) is not None]
    assert len({id(x) for x in leaves}) == len(leaves)
    assert {id(x) for x in top} < {id(x) for x in leaves}
    names = {type(x).__name__ for x in leaves if id(x) not in {id(y) for y in top}}
    # measured: linformer 377 leaves, 165 top-level; dense 257, 219 (the 38 others are QSoftmax tables)
    assert names == ({"QEinsum", "QEinsumDense", "QSoftmax", "QUnaryFunctionLUT"} if attn == "linformer" else {"QUnaryFunctionLUT"})


@pytest.mark.parametrize("callback", ["scheduler", "pid"])
def test_beta_from_the_callback_scales_every_ebops_term(model, callback):
    _, kmodel, inputs = model
    module = Module(kmodel)

    def set_beta(beta: float) -> None:
        if callback == "scheduler":
            BetaScheduler(beta_start=beta, beta_end=beta, warmup_steps=1).on_train_batch_start(Trainer(), module, None, 0)
        else:
            BetaPID(target_ebops=1.0, init_beta=beta, warmup_epochs=1).on_train_epoch_start(Trainer(), module)

    set_beta(1e-9)
    low = ebops_loss(kmodel, inputs)
    set_beta(2e-9)
    high = ebops_loss(kmodel, inputs)
    total = total_ebops(kmodel)
    assert total > 0
    assert all(abs(float(x._beta.value) - 2e-9) <= 1e-15 for x in kmodel.quantized_leaves())  # noqa: SLF001
    # the quantizer regularizers do not depend on beta, so the difference is the EBOPs term alone
    assert (high - low) == pytest.approx(1e-9 * total, rel=1e-3)
    if callback == "pid":
        assert module.logged["train/pid_ebops"] == pytest.approx(total, rel=1e-6)


def test_ebops_monitor_counts_every_leaf_once(model):
    attn, kmodel, inputs = model
    ebops_loss(kmodel, inputs)
    module = Module(kmodel)
    EBOPsMonitor().on_validation_epoch_end(Trainer(), module)
    own = sum(float(x._ebops.value) for x in kmodel.quantized_leaves())  # noqa: SLF001
    top = sum(float(torch.as_tensor(layer.ebops)) for layer in top_level(kmodel) if getattr(layer, "ebops", None) is not None)
    assert module.logged["val/ebops"] == pytest.approx(own, rel=1e-6)
    if attn == "linformer":  # the top-level read misses the core leaves (measured 2.117e11 of 2.509e11)
        assert top < 0.9 * own
    else:  # QSoftmax.ebops adds its tables, so for dense attention the top-level read was already complete
        assert top == pytest.approx(own, rel=1e-6)
