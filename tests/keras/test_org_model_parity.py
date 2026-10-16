"""The hybrid against the org/main torch model (base.yaml + linformer.yaml, ac0da8f) at full size, float.

Lane R4a gates (docs/hgq/RETARGET_ORG_MAIN.md sections 4 and 6):
- full-model float parity, measured then 2x, with the PR #8 controls at model level;
- the torch/hybrid state_dict key and shape check.

The torch side is built from the org configs, the hybrid side from the config polaris/03_make_config.py writes, both
by jsonargparse (org_model_utils.py). The masked-Linformer attention core of the hybrid is lane R4b, so the passing
rows run both models with dense attention (``to_dense_attention``) and the Linformer rows are strict xfails that R4b
turns on. Every event of the fixture has a padded node (RETARGET_ORG_MAIN.md section 1.2, batch of one).
"""

import copy

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import org_model_utils as U  # ty: ignore [unresolved-import]
from parity_utils import record_measurement  # ty: ignore [unresolved-import]
from test_maskformer_parity import flatten_outputs  # ty: ignore [unresolved-import]

from hepattn.keras.porting import port_maskformer

# Error metric: max over elements of |hybrid - torch| / (1 + |torch|), finite slots. Measured (macOS arm64, torch
# 2.10.0, CPU, dense attention, max over seeds 0-7, R4a and its fix round): outputs 1.65e-5 (seed 6, final mask
# logits), losses 1.70e-7 (the loss maximum moves between runs of the same seed, 5e-8 .. 1.7e-7). Tolerances are 2x.
# The rows run seeds 0 and 1. Re-measure on the first Linux/GPU run (carried gate).
OUT_TOL = 3.3e-5
LOSS_TOL = 3.4e-7
SEEDS = [0, 1]
R4B = "R4b: the hybrid masked-Linformer branch (RETARGET_ORG_MAIN.md section 4); Helen's KerasLinformerAttention has no q_proj"


@pytest.fixture(scope="module")
def configs(tmp_path_factory) -> dict:
    gen = U.generate_polaris_config(tmp_path_factory.mktemp("gen") / "polaris_hgq.yaml")
    org = U.torch_reference_config(U.merge_org_configs())
    return {"linformer": (org, gen), "dense": (U.to_dense_attention(org), U.to_dense_attention(gen))}


def scaled_err(ref: torch.Tensor, cand: torch.Tensor) -> float:
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(cand)), "non-finite patterns differ"
    return float(((cand[finite] - ref[finite]).abs() / (1 + ref[finite].abs())).max()) if finite.any() else 0.0


def run_pair(org: dict, hyb: dict, seed: int) -> dict:
    """Port torch -> hybrid, run both on the same events; per-key errors of outputs and matched losses."""
    torch.manual_seed(seed)
    tmodel = U.instantiate(org)
    kmodel = U.instantiate(hyb, quantized=False)
    port_maskformer(tmodel, kmodel)
    inputs, targets = U.clic_dummy_batch(2, seed=seed)
    with torch.no_grad():
        out_t, out_k = tmodel(inputs), kmodel(inputs)
        _, _, loss_t = tmodel.loss(out_t, dict(targets))
        _, _, loss_k = kmodel.loss(out_k, dict(targets))
    flat_t, flat_k = flatten_outputs(out_t), flatten_outputs(out_k)
    assert set(flat_t) == set(flat_k), set(flat_t) ^ set(flat_k)
    bools = {key: torch.equal(flat_t[key], flat_k[key]) for key in flat_t if flat_t[key].dtype == torch.bool}
    outs = {key: scaled_err(flat_t[key], flat_k[key]) for key in flat_t if flat_t[key].dtype != torch.bool}
    lt, lk = flatten_outputs(loss_t), flatten_outputs(loss_k)
    assert set(lt) == set(lk) and lt, set(lt) ^ set(lk)
    losses = {key: scaled_err(lt[key], lk[key]) for key in lt}
    return {"bools": bools, "outs": outs, "losses": losses}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("attn", ["dense", pytest.param("linformer", marks=pytest.mark.xfail(strict=True, raises=AttributeError, reason=R4B))])
def test_full_model_float_parity(configs, attn, seed):
    res = run_pair(*configs[attn], seed)
    assert res["bools"] and all(res["bools"].values()), [k for k, ok in res["bools"].items() if not ok]
    assert any(key.endswith("attn_mask") for key in res["bools"]), "the per-layer attention masks were not compared"
    out_err, loss_err = max(res["outs"].values()), max(res["losses"].values())
    for tag, err, tol in (("outputs", out_err, OUT_TOL), ("losses", loss_err, LOSS_TOL)):
        record_measurement({
            "tag": f"org_model.{attn}.{tag}",
            "config": f"seed{seed}",
            "max_abs_err": err,
            "max_rel_err": float("nan"),
            "atol": tol,
            "rtol": tol,
        })
    assert out_err <= OUT_TOL, max(res["outs"], key=res["outs"].get)
    assert loss_err <= LOSS_TOL, max(res["losses"], key=res["losses"].get)


@pytest.mark.parametrize(("key", "value"), [("query_update_order", "ca_dense_sa"), ("self_attn_norm_kv", False)])
def test_control_pr8_decoder_options(configs, key, value):
    """The other PR #8 decoder setting in the hybrid only: outputs and losses move far beyond the tolerances."""
    org, hyb = configs["dense"]
    hyb = copy.deepcopy(hyb)
    hyb["model"]["model"]["init_args"]["decoder"]["decoder_layer_config"][key] = value
    res = run_pair(org, hyb, seed=0)
    # measured (R4a): query_update_order outputs 1.56e1, losses 3.49e-2; self_attn_norm_kv 5.35, 3.27e-3
    assert max(res["outs"].values()) >= 10 * OUT_TOL
    assert max(res["losses"].values()) >= 10 * LOSS_TOL


def test_control_pr8_dense_norm_placement_is_refused(configs):
    org, hyb = configs["dense"]
    hyb = copy.deepcopy(hyb)
    hyb["model"]["model"]["init_args"]["encoder"]["dense_norm_placement"] = "hybridnorm"
    with pytest.raises(AssertionError, match="post_norm mismatch"):
        run_pair(org, hyb, seed=0)


def test_control_sorting_is_invisible_to_dense_attention(configs):
    """Documents why sorting is tested by its mechanics (test_sorting.py): without it, dense parity still holds."""
    org, hyb = configs["dense"]
    hyb = copy.deepcopy(hyb)
    hyb["model"]["model"]["init_args"].pop("input_sort_field")
    res = run_pair(org, hyb, seed=0)
    assert max(res["outs"].values()) <= OUT_TOL  # measured 9.29e-6, the matched floor


# ------------------------------------------------------------------------------- state_dict keys and shapes
def owner(key: str) -> str | None:
    """Module path of the torch leaf a hybrid twin replaces (Attention projections, Dense linears); None if reused."""
    for marker in (".in_proj_weight", ".in_proj_bias", ".out_proj.", ".value_residual_mix.0."):
        if marker in key:
            return key.split(marker, maxsplit=1)[0]
    if ".net." in key:
        return key.rsplit(".net.", 1)[0]
    return None


def relations(hyb: torch.Tensor, ref: torch.Tensor) -> str | None:
    """How a hybrid tensor holds a torch tensor: the same, transposed (Linear -> kernel), or one chunk of a packed one."""
    if hyb.shape == ref.shape and torch.equal(hyb, ref):
        return "same"
    if ref.dim() == 2 and hyb.shape == ref.T.shape and torch.equal(hyb, ref.T):
        return "T"
    for n in (2, 3):  # SwiGLU [x1 | x2], attention [q | k | v]
        if ref.shape[0] % n:
            continue
        for j, part in enumerate(ref.chunk(n, dim=0)):
            if part.dim() == 2 and hyb.shape == part.T.shape and torch.equal(hyb, part.T):
                return f"chunk{j}/{n}.T"
            if part.dim() == 1 and hyb.shape == part.shape and torch.equal(hyb, part):
                return f"chunk{j}/{n}"
    return None


def region(rel: str):
    """Index of the torch tensor part that a relation covers: all of it, or the rows of chunk j of n."""
    if rel in {"same", "T"}:
        return ...
    j, n = map(int, rel.removeprefix("chunk").removesuffix(".T").split("/"))
    return (j, n)


def unique_state(model) -> dict[str, torch.Tensor]:
    """state_dict without the decoder's alias of the tasks (MaskFormer registers them under tasks. and decoder.tasks.)."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("decoder.tasks.")}


def state_map(org: dict, hyb: dict, quantized: bool) -> tuple[dict, dict, list]:
    """Trace every hybrid state_dict entry to the torch entry it was ported from, by value.

    Every torch floating tensor is first refilled with distinct random values, so equal values identify the source.
    Returns (hybrid key -> (torch key, relation), torch key -> per-element cover count, unmatched hybrid keys).
    The cover count is kept per element, so a chunk ported twice and its sibling never is shows as 2 and 0, not as a
    full element total.
    """
    torch.manual_seed(0)
    tmodel = U.instantiate(org)
    kmodel = U.instantiate(hyb, quantized=quantized)
    if quantized:
        inputs, _ = U.clic_dummy_batch()
        with torch.no_grad():
            kmodel(inputs)  # quantized layers build lazily
    kmodel.register_keras_parameters()
    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for tensor in tmodel.state_dict(keep_vars=True).values():
            if tensor.is_floating_point():
                tensor.copy_(torch.randn(tensor.shape, generator=gen))
    port_maskformer(tmodel, kmodel)
    tsd, ksd = unique_state(tmodel), unique_state(kmodel)
    mapping, unmatched = {}, []
    covered = {key: torch.zeros(tensor.shape, dtype=torch.int32) for key, tensor in tsd.items()}
    for hkey, htensor in ksd.items():
        hits = [
            (tkey, rel)
            for tkey, ttensor in tsd.items()
            if ttensor.is_floating_point() == htensor.is_floating_point() and (rel := relations(htensor, ttensor))
        ]
        if not htensor.is_floating_point():
            hits = [(hkey, "same")] if hkey in tsd and torch.equal(tsd[hkey], htensor) else []
        if len(hits) != 1:
            unmatched.append((hkey, hits))
            continue
        mapping[hkey] = hits[0]
        tkey, rel = hits[0]
        part = region(rel)
        if part is ...:
            covered[tkey][...] += 1
        else:
            covered[tkey].chunk(part[1], dim=0)[part[0]].add_(1)  # chunk is a view: counts land in covered[tkey]
    return mapping, covered, unmatched


def check_state_map(mapping: dict, covered: dict, unmatched: list, quantized: bool) -> None:
    for hkey, (tkey, rel) in mapping.items():
        own = owner(tkey)
        if own is None:
            assert hkey == tkey and rel == "same", f"reused torch module state moved: {tkey} -> {hkey} ({rel})"
        else:
            assert hkey.startswith(own + "."), f"{tkey} ported outside its module: {hkey}"
    # (elements never covered, elements covered more than once) per torch entry that is not covered exactly once
    bad = {k: (int((c == 0).sum()), int((c > 1).sum())) for k, c in covered.items() if not bool((c == 1).all())}
    assert not bad, f"torch entries not covered exactly once (uncovered, multiply covered elements): {bad}"
    if quantized:  # the HGQ2 state has no torch source: quantizer variables (k, i, f, b, i_decay_speed), beta, ebops
        quantizer_state = ("/k", "/i", "/f", "/b", "/i_decay_speed")
        unmatched = [(k, h) for k, h in unmatched if not (("quantizer" in k and k.endswith(quantizer_state)) or k.endswith(("/beta", "/ebops")))]
    assert not unmatched, f"hybrid entries with no unique torch source: {unmatched[:5]}"


@pytest.mark.parametrize(
    ("attn", "quantized"),
    [
        ("dense", False),
        ("dense", True),
        pytest.param("linformer", False, marks=pytest.mark.xfail(strict=True, raises=AttributeError, reason=R4B)),
        # the quantized build materializes with a forward pass, which Helen's layer refuses (k=64 < 160 keys)
        pytest.param("linformer", True, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=R4B)),
    ],
)
def test_state_dict_keys_and_shapes_match_torch(configs, attn, quantized):
    mapping, covered, unmatched = state_map(*configs[attn], quantized=quantized)
    assert len(covered) > 150, len(covered)  # the full model (184 torch entries, 292 hybrid ones): not a vacuous map
    check_state_map(mapping, covered, unmatched, quantized)
    # the packed tensors were checked chunk by chunk: SwiGLU halves (2) and attention q/k/v (3)
    assert {rel for _, rel in mapping.values()} >= {"chunk0/2.T", "chunk1/2.T", "chunk0/3.T", "chunk1/3.T", "chunk2/3.T"}
