"""The hybrid against the org/main torch model (base.yaml + linformer.yaml, ac0da8f) at full size, float.

Lane R4a and R4b gates (docs/hgq/RETARGET_ORG_MAIN.md sections 4 and 6):
- full-model float parity, measured then 2x, with the PR #8 controls at model level;
- the torch/hybrid state_dict key and shape check.

The torch side is built from the org configs, the hybrid side from the config polaris/03_make_config.py writes, both
by jsonargparse (org_model_utils.py). ``linformer`` is the org model as it is (masked-Linformer attention everywhere,
the keras_native core inside the hybrid since R4b); ``dense`` swaps every attention for dense attention on both sides
(``to_dense_attention``), which keeps the R4a rows. Every event of the fixture has a padded node
(RETARGET_ORG_MAIN.md section 1.2, batch of one).
"""

import copy

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import org_model_utils as U  # ty: ignore [unresolved-import]
from parity_utils import record_measurement  # ty: ignore [unresolved-import]
from test_maskformer_parity import flatten_outputs  # ty: ignore [unresolved-import]

from hepattn.keras.porting import port_maskformer

# Error metric: max over elements of |hybrid - torch| / (1 + |torch|), finite slots. Tolerances are 2x the measured
# maximum (macOS arm64, torch 2.10.0, CPU). The rows run seeds 0 and 1. Re-measure on the first Linux/GPU run (carried).
# dense (R4a fix round 2, $S/r4a_fix2/logs/loss_seeds*.out): seeds 0-15, each run 3 times in one process and once in
# another: outputs 1.855e-5 (seed 12, final regression), losses 2.75e-7 (seed 12, layer_3 mask_bce).
# linformer (R4b, $S/r4b/logs/model_parity_seeds.out): seeds 0-15: outputs 5.088e-6 (seed 3, final mask logits),
# losses 1.875e-7 (seed 8, layer_1 mask_bce); the 4 attention-mask booleans equal for every seed.
TOLS = {"dense": (3.7e-5, 5.5e-7), "linformer": (1.02e-5, 3.75e-7)}
OUT_TOL, LOSS_TOL = TOLS["dense"]
SEEDS = [0, 1]


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
@pytest.mark.parametrize("attn", ["dense", "linformer"])
def test_full_model_float_parity(configs, attn, seed):
    res = run_pair(*configs[attn], seed)
    assert res["bools"] and all(res["bools"].values()), [k for k, ok in res["bools"].items() if not ok]
    assert any(key.endswith("attn_mask") for key in res["bools"]), "the per-layer attention masks were not compared"
    out_err, loss_err = max(res["outs"].values()), max(res["losses"].values())
    out_tol, loss_tol = TOLS[attn]
    for tag, err, tol in (("outputs", out_err, out_tol), ("losses", loss_err, loss_tol)):
        record_measurement({
            "tag": f"org_model.{attn}.{tag}",
            "config": f"seed{seed}",
            "max_abs_err": err,
            "max_rel_err": float("nan"),
            "atol": tol,
            "rtol": tol,
        })
    assert out_err <= out_tol, max(res["outs"], key=res["outs"].get)
    assert loss_err <= loss_tol, max(res["losses"], key=res["losses"].get)


@pytest.mark.parametrize("attn", ["dense", "linformer"])
@pytest.mark.parametrize(("key", "value"), [("query_update_order", "ca_dense_sa"), ("self_attn_norm_kv", False)])
def test_control_pr8_decoder_options(configs, attn, key, value):
    """The other PR #8 decoder setting in the hybrid only: outputs and losses move far beyond the tolerances."""
    org, hyb = configs[attn]
    hyb = copy.deepcopy(hyb)
    hyb["model"]["model"]["init_args"]["decoder"]["decoder_layer_config"][key] = value
    res = run_pair(org, hyb, seed=0)
    # measured: dense (R4a) query_update_order outputs 1.56e1, losses 3.49e-2; self_attn_norm_kv 5.35, 3.27e-3.
    # linformer (R4b, $S/r4b/logs/model_controls.out): query_update_order 3.25, 4.97e-4; self_attn_norm_kv 2.41, 5.57e-4
    out_tol, loss_tol = TOLS[attn]
    assert max(res["outs"].values()) >= 10 * out_tol
    assert max(res["losses"].values()) >= 10 * loss_tol


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


def test_control_sorting_sets_the_linformer_projection_rows(configs):
    """Plan 5.8 control: with the masked Linformer the sort order selects the E/F row of each token, so without it
    the hybrid leaves the org model (measured: outputs 3.47, losses 2.39e-2, 0 of 4 attention masks equal).
    """
    org, hyb = configs["linformer"]
    hyb = copy.deepcopy(hyb)
    hyb["model"]["model"]["init_args"].pop("input_sort_field")
    res = run_pair(org, hyb, seed=0)
    out_tol, loss_tol = TOLS["linformer"]
    assert max(res["outs"].values()) >= 10 * out_tol
    assert max(res["losses"].values()) >= 10 * loss_tol


# ------------------------------------------------------------------------------- state_dict keys and shapes
def owner(key: str) -> str | None:
    """Module path of the torch leaf a hybrid twin replaces (Attention projections, Dense linears); None if reused."""
    for marker in (".in_proj_weight", ".in_proj_bias", ".out_proj.", ".value_residual_mix.0."):
        if marker in key:
            return key.split(marker, maxsplit=1)[0]
    if key.endswith((".proj_k", ".proj_v")):  # linformer sequence projections (seq_len, k)
        return key.rsplit(".", 1)[0]
    if ".net." in key:
        return key.rsplit(".net.", 1)[0]
    return None


def relations(hyb: torch.Tensor, ref: torch.Tensor) -> str | None:
    """How a hybrid tensor holds a torch tensor: the same, transposed (Linear -> kernel), one chunk of a packed one, or
    a block of rows (a linformer projection split into the core's head kernel and its tail rows).
    """
    if hyb.shape == ref.shape and torch.equal(hyb, ref):
        return "same"
    if ref.dim() == 2 and hyb.shape == ref.T.shape and torch.equal(hyb, ref.T):
        return "T"
    if ref.dim() == 2 and hyb.dim() == 2 and hyb.shape[1] == ref.shape[1] and hyb.shape[0] < ref.shape[0]:
        h, r = hyb.shape[0], ref.shape[0]
        if torch.equal(hyb, ref[:h]):
            return f"rows0:{h}"
        if torch.equal(hyb, ref[r - h :]):
            return f"rows{r - h}:{r}"
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
    """Index of the torch tensor part that a relation covers: all of it, a block of rows, or the rows of chunk j of n."""
    if rel in {"same", "T"}:
        return ...
    if rel.startswith("rows"):
        a, b = map(int, rel.removeprefix("rows").split(":"))
        return slice(a, b)
    j, n = map(int, rel.removeprefix("chunk").removesuffix(".T").split("/"))
    return (j, n)


def unique_state(model) -> dict[str, torch.Tensor]:
    """state_dict without aliases: the decoder's copy of the tasks (MaskFormer registers them under tasks. and
    decoder.tasks.), and the keras variables that every enclosing keras layer lists again (``Layer.variables`` is
    recursive: the kernel of a linformer core's E projection is in the ParameterDict of the core, of E and of E's
    einsum). An alias is the same storage under the same keras variable path; the deepest key is kept.
    """
    sd = model.state_dict()
    groups: dict = {}
    for key, value in sd.items():
        if key.startswith("decoder.tasks."):
            continue
        ident = (value.data_ptr(), tuple(value.shape), value.stride()) if value.numel() else key
        groups.setdefault(ident, []).append(key)
    for keys in groups.values():
        paths = {key.rsplit("_torch_params.", 1)[-1] for key in keys}
        assert len(paths) == 1, f"one storage under different keras variables or torch keys: {keys}"
    return {(keep := max(keys, key=len)): sd[keep] for keys in groups.values()}


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
    inputs, _ = U.clic_dummy_batch()
    with torch.no_grad():
        kmodel(inputs)  # quantized layers, and the linformer cores in both modes, build at the first forward
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
        if part is ... or isinstance(part, slice):
            covered[tkey][part] += 1
        else:
            covered[tkey].chunk(part[1], dim=0)[part[0]].add_(1)  # chunk is a view: counts land in covered[tkey]
    return mapping, covered, unmatched


# Packed torch tensors, by the name of the hybrid leaf that holds one part: SwiGLU [x1 | x2], attention [q | k | v].
PACKED_PARTS = {"_x1": "chunk0/2", "_x2": "chunk1/2", "_q_proj": "chunk0/3", "_k_proj": "chunk1/3", "_v_proj": "chunk2/3"}
# Linformer projections by the name of the hybrid entry: (torch leaf, rows). E holds proj_k, F proj_v; the core's
# kernel is the head (rows :n, all rows when n = seq_len), its tail and the q_sa buffers the rows n:.
SEQ_PROJ = {"_E_proj/kernel": ("proj_k", "head"), "_F_proj/kernel": ("proj_v", "head"), "_E/tail": ("proj_k", "tail")}
SEQ_PROJ |= {"_F/tail": ("proj_v", "tail"), ".proj_k_dead": ("proj_k", "tail"), ".proj_v_dead": ("proj_v", "tail")}


def expected_relation(hkey: str) -> str:
    """The relation a hybrid key's name implies: a Keras kernel is the transposed torch weight, a bias the same."""
    leaf = hkey.rsplit(".", 1)[-1]
    layer, _, var = leaf.rpartition("/")
    part = next((rel for suffix, rel in PACKED_PARTS.items() if layer.endswith(suffix)), None)
    if var == "kernel":
        return f"{part}.T" if part else "T"
    if var == "bias":
        return part or "same"
    return "same"


def relation_ok(hkey: str, tkey: str, rel: str) -> bool:
    """Whether ``hkey`` may hold ``tkey`` as ``rel``, from the hybrid key's name alone."""
    seq = next((want for suffix, want in SEQ_PROJ.items() if hkey.endswith(suffix)), None)
    if seq is None:
        return rel == expected_relation(hkey)
    leaf, rows = seq
    if not tkey.endswith("." + leaf):
        return False
    head = rel == "same" or rel.startswith("rows0:")
    return head if rows == "head" else rel.startswith("rows") and not head


def check_state_map(mapping: dict, covered: dict, unmatched: list, quantized: bool) -> None:
    for hkey, (tkey, rel) in mapping.items():
        assert relation_ok(hkey, tkey, rel), f"{hkey} holds {tkey} as {rel}, which its name does not imply"
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


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("attn", ["dense", "linformer"])
def test_state_dict_keys_and_shapes_match_torch(configs, attn, quantized):
    mapping, covered, unmatched = state_map(*configs[attn], quantized=quantized)
    assert len(covered) > 150, len(covered)  # the full model (184 torch entries, 292 hybrid ones): not a vacuous map
    check_state_map(mapping, covered, unmatched, quantized)
    rels = {rel for _, rel in mapping.values()}
    # the packed tensors were checked chunk by chunk: SwiGLU halves (2) and attention q/k/v (3)
    assert rels >= {"chunk0/2.T", "chunk1/2.T", "chunk0/3.T", "chunk1/3.T", "chunk2/3.T"}
    if attn == "linformer":
        # every org proj_k / proj_v (24 each: 6 encoder, 3 x 4 decoder), split as the cores hold them: whole (encoder
        # 168, q_ca 160), head 150 + tail 10 (kv_ca), head 150 + q_sa buffer 10
        projections = [key for key in covered if key.endswith((".proj_k", ".proj_v"))]
        assert len(projections) == 2 * (6 + 3 * 4), projections
        assert rels >= {"rows0:150", "rows150:160"}
        assert sum(key.endswith("_dead") for key in mapping) == 2 * 4  # q_sa: the torch rows org never reads
