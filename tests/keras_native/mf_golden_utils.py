"""Native MaskFormer twins of the S3 golden cases (``golden/make_golden_maskformer.py``), and shared S3 helpers.

Weights: the seeded torch reference model is rebuilt (its state digest must equal the stored one) and ported with
``porting.port_maskformer``; every stored leaf variable (quantizer state) is then assigned by leaf name + position with
name/shape/dtype asserted, and every leaf kernel/bias must hash to the hybrid's digest. So the native model holds
exactly the hybrid's weights and quantizer state.
"""

import json
from pathlib import Path

import keras
import numpy as np
from hgq.layers import QDense, QEinsum, QSoftmax

from hepattn.keras_native.factory import EinsumOp, SoftmaxOp
from hepattn.keras_native.maskformer import MaskFormer

from .golden.mf_cases import array_digest, native_kwargs, torch_model, torch_state_digest
from .porting import port_maskformer

GOLDEN_DIR = Path(__file__).parent / "golden"
LEAF_TYPES = (QDense, QEinsum, QSoftmax, keras.layers.Dense, EinsumOp, SoftmaxOp)
MF_CASES = {"mf_torch": "torch", "mf_lin": "lin"}
MF_MODES = ("float", "high", "low", "prod", "prodsat")
# golden rows also checked with the jitted forward (the production attention; float, high and production bitwidths)
JIT_ROWS = {("mf_lin", "float"), ("mf_lin", "high"), ("mf_lin", "prodsat")}
_CACHE: dict[str, dict] = {}


def load_mf_golden(case: str) -> dict[str, np.ndarray]:
    if case not in _CACHE:
        path = GOLDEN_DIR / f"{case}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing: it is tracked in git; regenerate with golden/make_golden_maskformer.py in the hybrid torch env"
            )
        with np.load(path) as data:
            _CACHE[case] = {key: data[key] for key in data.files}
    return _CACHE[case]


def mf_meta(golden: dict) -> dict:
    return json.loads(str(golden["meta"]))


def mf_inputs(golden: dict) -> dict[str, np.ndarray]:
    return {key.removeprefix("in::"): value for key, value in golden.items() if key.startswith("in::")}


def mf_outputs(golden: dict, mode: str) -> dict[str, np.ndarray]:
    prefix = f"{mode}::out::"
    return {key.removeprefix(prefix): value for key, value in golden.items() if key.startswith(prefix)}


def native_leaves(model) -> dict:
    found = [layer for layer in model._flatten_layers(include_self=True, recursive=True) if isinstance(layer, LEAF_TYPES)]  # noqa: SLF001
    names = [layer.name for layer in found]
    assert len(set(names)) == len(names), f"native leaf names are not unique: {sorted(n for n in names if names.count(n) > 1)}"
    return dict(zip(names, found, strict=True))


def build_mf(variant: str, quant, inputs: dict, **kwargs) -> MaskFormer:
    """A native MaskFormer, built by one eager eval call (inside the HGQ2 scopes, via ``MaskFormer.build``)."""
    model = MaskFormer(**native_kwargs(variant, **kwargs), quant=quant)
    model(inputs, training=False)
    return model


def assign_mf_leaves(model, golden: dict, mode: str) -> int:
    """Assign the stored leaf variables, check kernel/bias digests; returns the number of variables checked."""
    meta = mf_meta(golden)[mode]
    leaves = native_leaves(model)
    stored = set(meta["leaves"])
    # the hybrid's float model runs the mask / incidence einsum and softmax as torch ops (no keras leaf); the native
    # float model has variable-free EinsumOp / SoftmaxOp leaves there
    extra = {name for name in set(leaves) - stored if not leaves[name].variables}
    assert set(leaves) - extra == stored, f"leaf sets differ: {(set(leaves) - extra) ^ stored}"
    count = 0
    for name in meta["leaves"]:
        layer = leaves[name]
        names = meta["variables"][name]
        assert [v.name for v in layer.variables] == names, f"{name}: variable order differs: {[v.name for v in layer.variables]} vs {names}"
        for i, var in enumerate(layer.variables):
            key = f"{name}::{i:03d}::{var.name}"
            if key in meta["digests"]:
                value = keras.ops.convert_to_numpy(var.value).astype(var.dtype)
                assert array_digest(value) == meta["digests"][key], f"{key}: ported weight differs from the hybrid's"
            else:
                value = golden[f"{mode}::kv::{key}"]
                assert tuple(var.shape) == value.shape, f"{key}: shape {tuple(var.shape)} vs {value.shape}"
                assert np.dtype(var.dtype) == value.dtype, f"{key}: dtype {var.dtype} vs {value.dtype}"
                var.assign(value)
            count += 1
    return count


def build_mf_from_golden(case: str, mode: str):
    """A native twin of ``case`` in ``mode`` holding the hybrid golden weights and quantizer state; returns (model, golden)."""
    golden = load_mf_golden(case)
    meta = mf_meta(golden)
    variant = MF_CASES[case]
    tmodel = torch_model(variant, seed=meta["seed"])
    assert torch_state_digest(tmodel) == meta["torch_state_sha256"], "rebuilt torch reference differs from the one the golden file was written from"
    model = build_mf(variant, meta[mode]["quant"], mf_inputs(golden))
    port_maskformer(tmodel, model)
    assign_mf_leaves(model, golden, mode)
    return model, golden


def flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        else:
            flat[path] = np.asarray(keras.ops.convert_to_numpy(value)) if not isinstance(value, np.ndarray) else value
    return flat


def regression_tie_rows(incidence: np.ndarray, is_track: np.ndarray) -> np.ndarray:
    """(B, P) True where a particle's regression proxy depends on how top-k breaks ties.

    ``get_proxy_feats`` gives a particle without any max-track node the top-2 particles per node instead. When the
    2nd and 3rd largest charged incidence of a node are equal and > 0, the top-2 set is implementation-defined
    (torch's CPU topk and JAX's ``top_k`` pick different members: measured 2 of 640 columns on 4-level random data).
    Quantized incidences sit on a coarse grid, so such ties are common there; they are absent in float.
    """
    ci = incidence * is_track[:, None, :]
    positive = ci > 0
    is_max = (ci == ci.max(axis=1, keepdims=True)) & positive
    zero_track = ~is_max.any(axis=2)  # (B, P)
    ranked = -np.sort(-ci, axis=1)  # descending over particles
    second, third = ranked[:, 1, :], ranked[:, 2, :]  # (B, N)
    ambiguous = (second > 0) & (second == third)
    candidate = ambiguous[:, None, :] & (ci == second[:, None, :])
    return zero_track & candidate.any(axis=2)


SENTINEL = 1e38  # |logit| above this is the finfo.min padding sentinel, compared exactly instead
LAYER0_LOGIT = "layer_0.mask.pflow_node_logit"


def _max_abs(ref: np.ndarray, cand: np.ndarray, keep: np.ndarray) -> float:
    d = np.abs(ref[keep].astype(np.float64) - cand[keep].astype(np.float64))
    return float(d.max()) if d.size else 0.0


def eager_forward(model):
    def run(inputs, attn_masks=None):
        return model(inputs, training=False, attn_masks=attn_masks)

    return run


def jitted_forward(model):
    """The forward as training runs it: ``jax.jit`` over ``stateless_call`` (XLA may fuse ops, so rounding can differ)."""
    import jax  # noqa: PLC0415

    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    fwd = jax.jit(lambda tv, ntv, inputs, masks: model.stateless_call(tv, ntv, inputs, training=False, attn_masks=masks)[0])

    def run(inputs, attn_masks=None):
        return fwd(tv, ntv, inputs, attn_masks)

    return run


def compare_mf(model, golden: dict, mode: str, jit: bool = False) -> dict:
    """Native vs hybrid golden outputs of one mode (eager, or jitted with ``jit=True``). Returns the measured metrics.

    - ``free``: the native model runs on its own (mask feedback included). ``free.flips.<layer>`` counts attention-mask
      elements that differ; ``free.<output>`` is the max abs difference.
    - ``forced``: the hybrid's attention masks are injected per layer (``attn_masks``), so a threshold flip in one layer
      cannot amplify into the next; ``forced.<output>`` is the max abs difference. The forced run returns the injected
      masks as its ``attn_mask`` outputs, so they are asserted equal (structural: teacher forcing took effect) and are
      not a parity metric; the mask-equality evidence is ``free.flips``.
    Non-finite / sentinel (finfo.min) elements must match exactly (asserted). Regression rows whose proxy depends on
    top-k tie-breaking (``regression_tie_rows``) are excluded and counted in ``tie_rows``.
    """
    inputs = mf_inputs(golden)
    ref = mf_outputs(golden, mode)
    run = jitted_forward(model) if jit else eager_forward(model)
    free = flatten(run(inputs))
    forced_masks = {key.split(".")[0]: value for key, value in ref.items() if key.endswith(".attn_mask")}
    forced = flatten(run(inputs, forced_masks))
    assert set(free) == set(ref) == set(forced), f"output keys differ: {set(free) ^ set(ref)}"
    ties = regression_tie_rows(ref["final.incidence.pflow_incidence"], inputs["node_is_track"])
    metrics = {"tie_rows": int(ties.sum())}
    for run, out in (("free", free), ("forced", forced)):
        for key, value in ref.items():
            cand = out[key]
            assert cand.shape == value.shape, f"{run}.{key}: shape {cand.shape} vs {value.shape}"
            if value.dtype == bool:
                if run == "forced":
                    assert np.array_equal(cand, value), f"forced.{key}: the injected hybrid mask was not used"
                else:
                    metrics[f"{run}.flips.{key}"] = int((cand != value).sum())
                continue
            special = ~np.isfinite(value) | (np.abs(value) > SENTINEL)
            if run == "forced" or key == LAYER0_LOGIT:
                assert np.array_equal(value[special], cand[special]), f"{run}.{key}: sentinel / non-finite elements differ"
            keep = ~special
            if key.startswith("final.regression"):
                keep &= ~ties[..., None]
            metrics[f"{run}.{key}"] = _max_abs(value, cand, keep)
    return metrics


def qloss_terms(model, golden: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(native, hybrid) training-mode HGQ2 loss terms, each sorted. Native: ``stateless_call(return_losses=True)``."""
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    _, _, losses = model.stateless_call(tv, ntv, mf_inputs(golden), training=True, return_losses=True)
    native = np.sort(np.array([float(keras.ops.convert_to_numpy(x)) for x in losses]))
    return native, np.sort(golden[f"{mode}::qlosses"])


def mf_init_state_mismatches(model, golden: dict, mode: str) -> list[str]:
    """Stored leaf variables (quantizer state, not kernel/bias) whose INITIAL native value differs from the hybrid's.

    The hybrid ran only eval-mode forwards before its variables were stored, so the stored quantizer state is its
    initial state. Loading the golden values (``assign_mf_leaves``) would hide a layer built outside the HGQ2 scopes
    (S2 mutation M13); this compares a freshly built model instead.
    """
    meta = mf_meta(golden)[mode]
    leaves = native_leaves(model)
    bad = []
    for name in meta["leaves"]:
        for i, var in enumerate(leaves[name].variables):
            key = f"{name}::{i:03d}::{var.name}"
            if key in meta["digests"]:
                continue
            ref = golden[f"{mode}::kv::{key}"]
            value = keras.ops.convert_to_numpy(var.value)
            if ref.shape != value.shape or not np.array_equal(ref, value):
                bad.append(key)
    return bad
