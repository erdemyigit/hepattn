"""polaris/03_make_config.py writes the org/main model (base.yaml + linformer.yaml) with the HGQ2 settings.

Lane R4a gates (docs/hgq/RETARGET_ORG_MAIN.md section 6): the generated config equals the org model section up to the
listed HGQ edits; it builds in float and quantized through jsonargparse, as main_hgq.py builds it; and it runs a
forward pass and the matched loss. With ``HGQ_ATTN=linformer`` (the default) the forward needs the hybrid
masked-Linformer branch of lane R4b: Helen's layer refuses k=64 < 160 keys under a mask, so those rows are strict
xfails until R4b lands. ``HGQ_ATTN=torch`` (base.yaml alone, dense attention) runs now.
"""

import ast
import copy

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import org_model_utils as U  # ty: ignore [unresolved-import]
import yaml
from test_swiglu_quantized import ebops_term  # ty: ignore [unresolved-import]

from hepattn.keras.dense import KerasDense
from hepattn.keras.encoder import KerasEncoder
from hepattn.keras.linformer import KerasLinformerAttention

R4B = "R4b: Helen's KerasLinformerAttention asserts k >= kv_len under a mask (k=64, decoder kv_len=160)"
ATTNS = ["linformer", "torch"]


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("gen")
    return {attn: U.generate_polaris_config(out / f"{attn}.yaml", HGQ_ATTN=attn) for attn in ATTNS}


def expected_model_section(attn: str) -> dict:
    """The org model section with only the edits the generator documents."""
    cfg = U.merge_org_configs() if attn == "linformer" else yaml.safe_load((U.CONFIGS / "base.yaml").read_text())
    net = copy.deepcopy(cfg["model"]["model"]["init_args"])
    encoder = net["encoder"]["init_args"]
    assert encoder.pop("dim") == net["dim"]
    if attn == "torch":
        encoder["attn_type"] = "torch"
    net["encoder"] = encoder
    net["matcher"]["init_args"]["default_solver"] = "scipy"
    return net


@pytest.mark.parametrize("attn", ATTNS)
def test_generated_model_is_the_org_model(generated, attn):
    net = copy.deepcopy(generated[attn]["model"]["model"])
    assert net["class_path"] == "hepattn.keras.maskformer.KerasMaskFormer"
    init = net["init_args"]
    assert set(init.pop("quant")) == {"weight", "datalane", "table", "ebops"}
    for task in init["tasks"]["init_args"]["modules"]:  # absolute in the generated file, relative in the repo
        if "scale_dict_path" in task["init_args"]:
            task["init_args"]["scale_dict_path"] = "configs/clic_var_transform.yaml"
    assert init == expected_model_section(attn)
    if attn == "linformer":  # the org/main settings this lane retargets to (k=256 and no sort before R4a)
        assert init["encoder"]["attn_kwargs"] == {"num_heads": 16, "linformer_seq_len": 168, "linformer_k": 64}
        assert init["decoder"]["decoder_layer_config"]["attn_kwargs"]["linformer_k"] == 64
        assert init["input_sort_field"] == "phi"
    else:
        assert "input_sort_field" not in init


def test_default_is_the_linformer_model(generated, tmp_path, monkeypatch):
    """What Polaris runs: 03_make_config.py with no HGQ_ATTN."""
    monkeypatch.delenv("HGQ_ATTN", raising=False)
    default = U.generate_polaris_config(tmp_path / "default.yaml")
    assert default["model"] == generated["linformer"]["model"]
    assert default["model"] != generated["torch"]["model"]


def generator_merge():
    """The generator's own ``merge``, taken from its source: importing 03_make_config.py would write a config."""
    tree = ast.parse(U.MAKE_CONFIG.read_text())
    (fn,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "merge"]
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(U.MAKE_CONFIG), "exec"), namespace)  # noqa: S102
    return namespace["merge"]


def test_generator_merge_is_a_deep_overlay():
    """Assertion-level check of the overlay rule (a broken rule otherwise surfaces only as a generator crash)."""
    merge = generator_merge()
    base = {"a": {"b": 1, "c": {"d": 2, "e": 3}}, "f": [1, 2], "g": 4}
    overlay = {"a": {"c": {"d": 5}}, "f": [9], "h": {"i": 6}}
    assert merge(base, overlay) == {"a": {"b": 1, "c": {"d": 5, "e": 3}}, "f": [9], "g": 4, "h": {"i": 6}}
    assert base == {"a": {"b": 1, "c": {"d": 2, "e": 3}}, "f": [1, 2], "g": 4}, "merge changed its input"
    org_base = yaml.safe_load((U.CONFIGS / "base.yaml").read_text())
    org_overlay = yaml.safe_load((U.CONFIGS / "linformer.yaml").read_text())
    assert merge(org_base, org_overlay) == U.merge(org_base, org_overlay)


def test_swiglu_products_are_trained_with_the_model(generated):
    """Every quantized SwiGLU product leaf is seen by the model-level walks that QAT relies on.

    keras_layers() feeds quant_losses() and trainable_parameter_groups() (the optimizer). A product leaf that is not
    a registered submodule still runs in the forward, but its quantizers would get no EBOPs pressure and no updates.
    """
    model = U.instantiate(generated["torch"], quantized=True)
    inputs, _ = U.clic_dummy_batch()
    with torch.no_grad():
        model(inputs)
    model.register_keras_parameters()
    products = {id(leaf): leaf for m in model.modules() if isinstance(m, KerasDense) for leaf in getattr(m, "products", [])}
    # input net 1, encoder FFNs 6, decoder q and kv FFNs 8, mask object and constituent nets 2, incidence node net 1
    assert len(products) == 18, len(products)
    assert all(type(leaf).__name__ == "QEinsum" for leaf in products.values())
    layers = {id(layer) for layer in model.keras_layers()}
    assert set(products) <= layers, [leaf.name for key, leaf in products.items() if key not in layers]
    decay, quant = model.trainable_parameter_groups()
    optimized = {id(p) for p in decay + quant}
    leaf_vars = [w for leaf in products.values() for w in leaf.trainable_weights]
    assert leaf_vars and all(id(w.value) in optimized for w in leaf_vars)
    for layer in model.keras_layers():  # as BetaScheduler/BetaPID set it; the config's initial beta is 0
        if getattr(layer, "_beta", None) is not None:
            layer._beta.assign(1e-6)  # noqa: SLF001
    model.train()
    with torch.no_grad():
        model(inputs)
    # each product counted EBOPs in this training forward (leaf.losses alone also holds regularizer terms)
    product_ebops = sum(float(ebops_term(leaf)) for leaf in products.values())
    # the walk yields the task heads twice (model.tasks and model.decoder.tasks, port_log/S3.md deviation 9)
    walked = [(id(layer) in products, float(t)) for layer in model.keras_layers() for t in layer.losses]
    walked_products = sum(t for is_product, t in walked if is_product)
    assert walked_products >= product_ebops * (1 - 1e-6)
    assert float(model.quant_losses()) == pytest.approx(sum(t for _, t in walked), rel=1e-5)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("attn", ATTNS)
def test_generated_config_builds(generated, attn, quantized):
    model = U.instantiate(generated[attn], quantized=quantized)
    assert isinstance(model.encoder, KerasEncoder)
    assert model.factory.quantize == quantized
    assert model.input_sort_field == ("phi" if attn == "linformer" else None)
    assert type(model.matcher).__name__ == "Matcher" and model.matcher.solver == "scipy"
    enc0 = model.encoder.layers[0]
    assert type(enc0.attn.norm).__name__ == "FastLayerNorm"
    dec0 = model.decoder.decoder_layers[0]
    assert (dec0.query_update_order, dec0.self_attn_norm_kv) == ("ca_sa_dense", True)
    ffn = enc0.dense.fn
    assert ffn.gate and len(ffn.products) == 1
    leaves = [type(layer).__name__ for layer in (ffn.hidden[0], ffn.gates[0], ffn.products[0])]
    assert leaves == (["QDense", "QDense", "QEinsum"] if quantized else ["Dense", "Dense", "EinsumOp"])
    if attn == "linformer":  # R0 carried gate: linformer_k must reach the layer (256 is the hybrid default)
        shapes = {tuple(layer.attn.fn.proj_k.shape) for layer in model.encoder.layers}
        shapes |= {tuple(getattr(layer, part).fn.proj_k.shape) for layer in model.decoder.decoder_layers for part in ("q_ca", "q_sa", "kv_ca")}
        assert shapes == {(168, 64), (160, 64)}
        assert isinstance(dec0.q_ca.fn, KerasLinformerAttention)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("attn", [pytest.param("linformer", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=R4B)), "torch"])
def test_generated_config_forward_and_matched_loss(generated, attn, quantized):
    torch.manual_seed(0)
    model = U.instantiate(generated[attn], quantized=quantized)
    inputs, targets = U.clic_dummy_batch()
    with torch.no_grad():
        outputs = model(inputs)
        model.register_keras_parameters()
        _, _, losses = model.loss(outputs, dict(targets))
    assert {name for name, layer in losses.items() if layer} == {"layer_0", "layer_1", "layer_2", "layer_3", "final"}
    terms = [v for layer in losses.values() for task in layer.values() for v in task.values()]
    assert len(terms) == 4 * 3 + 5, len(terms)  # object_ce, mask_bce, mask_dice per layer; + kl_div, regression
    assert all(bool(torch.isfinite(t)) for t in terms), losses
    assert outputs["final"]["mask"]["pflow_node_logit"].shape == (2, U.NUM_QUERIES, U.MAX_NODES)
