"""Gate H0 (docs/hgq/RETARGET_ORG_MAIN.md section 4) and the hybrid masked-Linformer attention against org/main.

The hybrid ``KerasAttention(attn_type="linformer")`` runs the keras_native masked-Linformer core
(``hepattn.keras_native.masked_linformer``) under the torch backend. H0 asks for:
1. the core runs under ``KERAS_BACKEND=torch`` without jax (``hepattn/.venv`` has none);
2. float parity with ``masked_linformer.attend`` / ``project`` at <= 2x the JAX-side measurement (the tolerances of
   tests/keras_native/test_masked_linformer.py, which are 2x the R2 measurement; the same fixtures and seeds here);
3. quantized: the torch-backend output equals the JAX-backend output for the same weights and inputs
   (golden/linformer_backend.npz, written by make_linformer_backend_golden.py under JAX);
4. every core variable is a registered torch parameter and gets optimizer updates.
Then the hybrid attention module against org ``Attention`` at the org call sites, float and quantized.

Shapes are the production ones of ``configs/linformer.yaml``: decoder ``q_ca`` (150 queries over 160 constituents,
seq_len 160), ``kv_ca`` (160 over 150: the E/F rows 150-159 exist only for the renorm), encoder self-attention (168 =
160 + 8 registers), decoder ``q_sa`` (no mask: ``E[:150]``), and a short encoder (40 tokens, seq_len 168, tail rows in
the renorm). Measured values are next to each tolerance (macOS arm64 CPU, torch 2.10.0); docs/hgq/port_log/R4b.md
lists the controls and the mutants that exceed them.
"""

import copy
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from itertools import starmap

import linformer_backend_cases as C  # ty: ignore [unresolved-import]
import masked_linformer as ml
import numpy as np
import org_model_utils as U  # ty: ignore [unresolved-import]
from parity_utils import assert_parity  # ty: ignore [unresolved-import]

from hepattn.keras.attention import KerasAttention
from hepattn.keras.callbacks import write_beta
from hepattn.keras.factory import LayerFactory
from hepattn.keras.porting import port_attention
from hepattn.keras_native import masked_linformer as nml
from hepattn.models.attention import Attention

DIM, HEADS, HEAD_DIM, K = 256, 16, 16, 64
B = 2
SHAPES = {  # (q_len, kv_len, linformer_seq_len, core)
    "q_ca": (150, 160, 160, "attend"),
    "kv_ca": (160, 150, 160, "attend"),
    "enc": (168, 168, 168, "project"),
    "enc_short": (40, 40, 168, "project"),
    "q_sa": (150, 150, 160, "project_unmasked"),
}
SELF = {"enc", "enc_short", "q_sa"}
HIGH_SPEC = C.SPECS["high"]
# the low end of the paired delta (tests/keras_native/test_masked_linformer.py): the golden "low" spec makes this
# attention return a constant, this one does not
COARSE_SPEC = {
    "weight": {"default_q_type": "kbi", "b0": 6, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 8, "f0": 4},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 4},
}
GOLDEN = pathlib.Path(__file__).parent / "golden" / "linformer_backend.npz"


def t(x):
    return None if x is None else torch.from_numpy(np.asarray(x))


# ------------------------------------------------------------------------------- H0 (1): no jax under torch
NO_JAX_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, sys

    class BlockJax(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "jax" or name.startswith("jax."):
                raise ImportError("blocked: " + name)

    sys.meta_path.insert(0, BlockJax())
    import hepattn.keras
    hepattn.keras.set_keras_default_device("cpu")
    import keras, torch
    from hepattn.keras.attention import KerasAttention
    from hepattn.keras.factory import LayerFactory

    if sys.argv[1] == "control":
        try:
            import hepattn.keras_native.matching  # imports jax
        except ImportError as err:
            print("BLOCKED", err)
        sys.exit(0)
    spec = {"weight": {"default_q_type": "kbi", "b0": 8, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8}}
    ran = []
    for quant in (None, spec):
        factory = LayerFactory(quant)
        with factory.scopes():
            cross = KerasAttention(32, 2, attn_type="linformer", linformer_seq_len=12, linformer_k=4, factory=factory, name="x")
            selfa = KerasAttention(32, 2, attn_type="linformer", linformer_seq_len=12, linformer_k=4, factory=factory, name="s")
        q, kv = torch.randn(2, 5, 32), torch.randn(2, 8, 32)
        kv_mask = torch.ones(2, 8, dtype=torch.bool)
        kv_mask[0, -2:] = False
        out = cross(q, kv, kv, attn_mask=torch.rand(2, 5, 8) < 0.5, kv_mask=kv_mask)
        out2 = selfa(kv, kv_mask=kv_mask)  # project core with renorm and tail rows (8 < 12)
        assert torch.isfinite(out).all() and torch.isfinite(out2[kv_mask]).all()
        ran.append((type(cross.core).__name__, type(selfa.core).__name__, selfa.core.E.tail is not None))
    print("RAN", ran, "JAX", sorted(m for m in sys.modules if m.split(".")[0] == "jax"), keras.backend.backend())
    """
)


def run_no_jax(mode: str) -> str:
    env = {**os.environ, "KERAS_BACKEND": "torch", "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    proc = subprocess.run([sys.executable, "-c", NO_JAX_SCRIPT, mode], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


def test_core_runs_under_torch_with_jax_blocked():
    """The hybrid linformer attention, float and quantized, both cores, in a process where importing jax raises."""
    out = run_no_jax("run")
    assert "RAN [('AttendCore', 'ProjectCore', True), ('AttendCore', 'ProjectCore', True)] JAX [] torch" in out, out


def test_the_jax_blocker_works():
    """Control for the test above: keras_native.matching imports jax and fails under the same hook."""
    assert "BLOCKED blocked: jax" in run_no_jax("control")


# ----------------------------------------------------------------------- H0 (2): float core vs masked_linformer
# Tolerances: the JAX-side ones (2x the R2 measurement, tests/keras_native/test_masked_linformer.py). Measured here
# (torch backend, max over q_ca, kv_ca): forward 8.345e-7; gradients q 7.153e-7, k 3.874e-7, v 3.576e-7, E 1.907e-5,
# F 1.717e-5. JAX side (R2): 8.345e-7; q 8.345e-7, k 3.278e-7, v 3.576e-7, E 4.244e-5, F 5.913e-5.
ATTEND_FWD_TOL = 1.67e-6
ATTEND_GRAD_TOL = {"q": 1.67e-6, "k": 6.56e-7, "v": 7.15e-7, "E": 8.49e-5, "F": 1.18e-4}


def core_projections(seq_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-(K**-0.5), K**-0.5, (seq_len, K)).astype(np.float32) for _ in range(2))


def attend_case(shape: str):
    """The fixture of the native attend test, value for value: NaN in padded k/v rows, an all-False query row."""
    q_len, kv_len, seq_len, _ = SHAPES[shape]
    rng = np.random.default_rng(1)
    q = rng.standard_normal((B, HEADS, q_len, HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32)
    kv_mask = np.ones((B, kv_len), bool)
    kv_mask[0, -7:] = False
    kv_mask[1, -1:] = False
    attn_mask = (rng.random((B, q_len, kv_len)) < 0.5) & kv_mask[:, None]
    attn_mask[0, 1] = False
    k = np.where(kv_mask[:, None, :, None], k, np.nan).astype(np.float32)
    v = np.where(kv_mask[:, None, :, None], v, np.nan).astype(np.float32)
    pk, pv = core_projections(seq_len, 2)
    core = nml.AttendCore(LayerFactory(), HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, name="att")
    core.build()
    core.E.assign_full(pk)
    core.F.assign_full(pv)
    assert core.E.has_tail == (seq_len > kv_len)
    cot = rng.standard_normal((B, HEADS, q_len, HEAD_DIM)).astype(np.float32)
    return core, (q, k, v, pk, pv), kv_mask, attn_mask, t(cot)


@pytest.mark.parametrize("shape", ["q_ca", "kv_ca"])
def test_attend_core_forward_and_gradients_equal_masked_linformer(shape):
    """Forward and torch-autograd gradients for q, k, v, E, F; padded k/v rows hold NaN and get exactly 0 gradient;
    an all-False query row gives exactly 0.
    """
    core, arrays, kv_mask, attn_mask, cot = attend_case(shape)
    ref_in = [torch.tensor(a, requires_grad=True) for a in arrays]
    ref = ml.attend(*ref_in, t(attn_mask))
    (ref * cot).sum().backward()
    hyb_in = [torch.tensor(a, requires_grad=True) for a in arrays[:3]]
    out = core(*hyb_in, t(attn_mask))
    (out * cot).sum().backward()

    assert torch.isfinite(out).all()
    assert_parity("h0.masked_linformer.attend", shape, ref, out, atol=ATTEND_FWD_TOL, rtol=0.0)
    all_false = ~t(attn_mask).any(-1)  # (B, Q)
    assert all_false.any() and out.detach().transpose(1, 2)[all_false].abs().max() == 0.0

    def full(proj) -> torch.Tensor:
        rows = [proj.kernel.value.grad]
        if proj.tail is not None:
            rows.append(proj.tail.value.grad)
        return torch.cat(rows, 0)

    got = {"q": hyb_in[0].grad, "k": hyb_in[1].grad, "v": hyb_in[2].grad, "E": full(core.E), "F": full(core.F)}
    want = dict(zip("qkvEF", (x.grad for x in ref_in), strict=True))
    padded = (~t(kv_mask))[:, None, :, None].expand_as(got["k"])
    for name in ("k", "v"):
        assert got[name][padded].abs().max() == 0.0 and want[name][padded].abs().max() == 0.0, f"gradient into padded {name}"
        got[name], want[name] = got[name][~padded], want[name][~padded]
    for name, tol in ATTEND_GRAD_TOL.items():
        assert_parity("h0.masked_linformer.attend_grad", f"{shape}.{name}", want[name], got[name], atol=tol, rtol=0.0)


# measured (torch backend): enc 1.013e-6, enc_short 5.960e-7, q_sa 0.0. JAX side (R2): 1.013e-6, 5.960e-7, 2.980e-7
PROJECT_TOL = {"enc": 2.03e-6, "enc_short": 1.19e-6, "q_sa": 5.96e-7}


def project_case(shape: str) -> tuple:
    """The fixture of the native project test, value for value."""
    q_len, kv_len, seq_len, kind = SHAPES[shape]
    masked = kind == "project"
    rng = np.random.default_rng(3)
    q, k, v = (rng.standard_normal((B, HEADS, kv_len, HEAD_DIM)).astype(np.float32) for _ in range(3))
    kv_mask = None
    if masked:
        kv_mask = np.ones((B, kv_len), bool)
        kv_mask[0, -9:] = False
        kv_mask[1, -1:] = False
        k = np.where(kv_mask[:, None, :, None], k, np.nan).astype(np.float32)
        v = np.where(kv_mask[:, None, :, None], v, np.nan).astype(np.float32)
    pk, pv = core_projections(seq_len, 4)
    core = nml.ProjectCore(LayerFactory(), HEADS, HEAD_DIM, q_len, kv_len, seq_len, K, masked=masked, name="prj")
    core.build()
    core.E.assign_full(pk)
    core.F.assign_full(pv)
    assert core.E.has_tail == (masked and seq_len > kv_len)
    with torch.no_grad():
        out = core(t(q), t(k), t(v), t(kv_mask))
    return out, (t(q), t(k), t(v), t(pk), t(pv), t(kv_mask))


@pytest.mark.parametrize("shape", ["enc", "enc_short", "q_sa"])
def test_project_core_equals_project_then_unmasked_attention(shape):
    out, (q, k, v, pk, pv, mask) = project_case(shape)
    ref = F.scaled_dot_product_attention(q, ml.project(k, pk, mask), ml.project(v, pv, mask))
    assert torch.isfinite(out).all()
    assert_parity("h0.masked_linformer.project", shape, ref, out, atol=PROJECT_TOL[shape], rtol=0.0)


def test_core_parity_tolerances_resolve_the_renorm():
    """Control: masked_linformer without its renorm is >= 1e4x outside the attend and project tolerances, so those
    rows see the renorm (measured: attend q_ca 0.647, project enc_short 0.921).
    """
    core, (q, k, v, pk, pv), _, attn_mask, _ = attend_case("q_ca")
    with torch.no_grad():
        out = core(t(q), t(k), t(v), t(attn_mask))
        no_renorm = ml.attend(t(q), t(k), t(v), t(pk), t(pv), t(attn_mask), renorm=False)
    assert (out - no_renorm).nan_to_num().abs().max() > 1e4 * ATTEND_FWD_TOL
    out, (q, k, v, pk, pv, mask) = project_case("enc_short")
    no_renorm = F.scaled_dot_product_attention(q, ml.project(k, pk, mask, renorm=False), ml.project(v, pv, mask, renorm=False))
    assert (out - no_renorm).abs().max() > 1e4 * PROJECT_TOL["enc_short"]


# ----------------------------------------------------------------- H0 (3): torch backend == JAX backend, quantized
# Measured (golden/linformer_backend.npz, 2 heads, 5 sites x {high, prod} x {eval, train}): 0.0 on every element, so
# the tolerance is 0 (plan section 5 policy; re-measure on the first Linux/GPU run). The 16-head rows are below.
BACKEND_TOL = 0.0
# 16 heads (production), attend sites ($S/r4b_fix/logs/h16_measure.out; the same with OMP_NUM_THREADS=1): prod 0.0
# everywhere; high: q_ca eval 1.656e-4 (64 elements), q_ca train 7.875e-5 (16), kv_ca eval 1.437e-4 (48), kv_ca
# train 1.069e-4 (112) of 76800 / 81920. Every differing element belongs to a whole (event, head, query) row of
# head_dim elements: a float32 rounding tie upstream of the values einsum that the two backends break differently.
# Tolerance (2x): (max |diff|, number of differing elements). The count is what a real bug exceeds: one datalane bit
# less differs in >= 57686 elements ($S/r4b_fix/logs/h16_control.out), >= 257x the bound; its max |diff| at the high
# spec, 5.6e-4, is only 1.7x the 3.31e-4 bound.
PROD_HEADS_TOL = {"high": (3.31e-4, 224), "prod": (0.0, 0)}


@pytest.fixture(scope="module")
def backend_golden():
    golden = np.load(GOLDEN)
    assert str(golden["meta.inputs_sha256"]) == C.inputs_digest(), "the cases' inputs changed: regenerate the golden"
    return golden


@pytest.mark.parametrize("mode", list(C.MODES))
@pytest.mark.parametrize("site", list(C.SITES))
@pytest.mark.parametrize("spec_name", list(C.SPECS))
def test_quantized_core_torch_backend_equals_jax_backend(backend_golden, spec_name, site, mode):
    ref = torch.from_numpy(backend_golden[C.key(spec_name, site, mode)])
    got = torch.from_numpy(C.run_case(site, C.SPECS[spec_name], C.MODES[mode]))
    assert torch.isfinite(got).all()
    assert_parity("h0.backend", C.key(spec_name, site, mode), ref, got, atol=BACKEND_TOL, rtol=0.0)


@pytest.mark.parametrize("mode", list(C.MODES))
@pytest.mark.parametrize("site", C.PROD_HEAD_SITES)
@pytest.mark.parametrize("spec_name", list(C.SPECS))
def test_quantized_core_at_production_heads_equals_jax_up_to_rounding_ties(backend_golden, spec_name, site, mode):
    ref = backend_golden[C.key(spec_name, site, mode, C.PROD_HEADS)]
    got = C.run_case(site, C.SPECS[spec_name], C.MODES[mode], C.PROD_HEADS)
    assert np.isfinite(got).all()
    differs = np.abs(got - ref) > 0
    max_diff, max_count = PROD_HEADS_TOL[spec_name]
    assert float(np.abs(got - ref).max()) <= max_diff
    assert int(differs.sum()) <= max_count, int(differs.sum())
    assert int(differs.sum()) == C.HEAD_DIM * int(differs.any(-1).sum()), "differences outside whole query rows"


def test_production_heads_control_one_datalane_bit():
    """Control for the bound above: one fractional datalane bit less at the high spec (measured 76540 elements)."""
    spec = copy.deepcopy(C.SPECS["high"])
    spec["datalane"]["f0"] -= 1
    ref = np.load(GOLDEN)[C.key("high", "q_ca", "train", C.PROD_HEADS)]
    assert int((np.abs(C.run_case("q_ca", spec, True, C.PROD_HEADS) - ref) > 0).sum()) >= 10 * PROD_HEADS_TOL["high"][1]


@pytest.mark.parametrize("site", list(C.SITES))
def test_backend_golden_resolves_one_datalane_bit(backend_golden, site):
    """Control: one fractional datalane bit less (high spec) moves the output (measured >= 2.84e-4 on every site), and
    the float core is far from the quantized golden (>= 3.56e-2): the cases run a live quantized graph.
    """
    spec = copy.deepcopy(C.SPECS["high"])
    spec["datalane"]["f0"] -= 1
    assert np.abs(C.run_case(site, spec, True) - backend_golden[C.key("high", site, "train")]).max() > 1e-4
    assert np.abs(C.run_case(site, None, False) - backend_golden[C.key("high", site, "eval")]).max() > 1e-2


# --------------------------------------------------------- H0 (4): core variables are torch parameters and train
@pytest.fixture(scope="module")
def quantized_model(tmp_path_factory):
    """The generated Polaris model (org linformer, production spec), materialized and registered as MPflowHGQ.setup."""
    cfg = U.generate_polaris_config(tmp_path_factory.mktemp("gen") / "polaris_hgq.yaml")
    torch.manual_seed(0)
    model = U.instantiate(cfg, quantized=True)
    inputs, targets = U.clic_dummy_batch()
    with torch.no_grad():
        model(inputs)
    model.register_keras_parameters()
    return model, inputs, targets


def cores(model) -> list:
    found = [m for m in model.modules() if isinstance(m, nml.AttendCore | nml.ProjectCore)]
    assert len(found) == 6 + 3 * 4, len(found)  # one per attention: 6 encoder layers, q_ca/q_sa/kv_ca x 4 decoder layers
    return found


def test_every_core_variable_is_a_registered_torch_parameter(quantized_model):
    model, _, _ = quantized_model
    variables = {id(v): v for core in cores(model) for v in core.variables}
    assert len(variables) > 1000, len(variables)  # measured 1840: kernels, tails, quantizer state, beta, ebops
    params = {id(p) for p in model.parameters()}
    assert [v.path for v in variables.values() if id(v.value) not in params] == []
    state = {tensor.data_ptr() for tensor in model.state_dict().values()}
    assert [v.path for v in variables.values() if v.value.data_ptr() not in state] == []
    decay, quant = model.trainable_parameter_groups()
    quant_ids = {id(p) for p in quant}
    grouped = quant_ids | {id(p) for p in decay}
    trainable = [v for v in variables.values() if v.trainable]
    assert len(trainable) > 300, len(trainable)  # measured 404
    assert [v.path for v in trainable if id(v.value) not in grouped] == []
    assert [v.path for v in variables.values() if not v.trainable and id(v.value) in grouped] == []
    # kernels and tails take weight decay, bitwidths do not
    assert [v.path for v in trainable if (id(v.value) in quant_ids) != ("quantizer" in v.path)] == []


def test_every_trainable_core_variable_gets_a_gradient_and_moves(quantized_model):
    """One training step (task losses + EBOPs at beta 1e-6, as BetaScheduler sets beta) with AdamW at weight decay 0:
    every trainable core variable has a finite nonzero gradient and changes (measured 404 of 404). SGD at lr 1e-2
    would move only 190 (measured at R4b with beta on the top-level leaves only): some bitwidths get EBOPs gradients
    of 1e-8, below float32 resolution after the lr.
    """
    model, inputs, targets = quantized_model
    trainable = {id(v): v for core in cores(model) for v in core.variables if v.trainable}
    before = {key: v.value.detach().clone() for key, v in trainable.items()}
    write_beta(model, 1e-6)  # as BetaScheduler: every leaf, the core leaves included
    core_leaves = [m for core in cores(model) for m in core.modules() if getattr(m, "_beta", None) is not None]
    assert len(core_leaves) > 200, len(core_leaves)  # measured 212
    assert all(float(leaf._beta.value) == pytest.approx(1e-6, rel=1e-6) for leaf in core_leaves)  # noqa: SLF001
    decay, quant = model.trainable_parameter_groups()
    opt = torch.optim.AdamW([{"params": decay}, {"params": quant}], lr=1e-3, weight_decay=0.0)
    model.train()
    opt.zero_grad()
    outputs = model(inputs)
    _, _, losses = model.loss(outputs, dict(targets))
    total = sum(v for layer in losses.values() for task in layer.values() for v in task.values()) + model.quant_losses()
    assert torch.isfinite(total)
    total.backward()
    opt.step()
    assert [v.path for v in trainable.values() if v.value.grad is None or not bool((v.value.grad != 0).any())] == []
    assert all(bool(torch.isfinite(v.value.grad).all()) for v in trainable.values())
    frozen = [v.path for key, v in trainable.items() if torch.equal(before[key], v.value.detach())]
    assert frozen == [], f"{len(frozen)} of {len(trainable)} trainable core variables did not move: {frozen[:3]}"


def test_float_cores_warm_start_the_quantized_model(tmp_path):
    """port_keras_to_keras (float -> QAT warm start) copies every core's projections, head, tail and q_sa rows:
    after the port the quantized model holds the float model's (seq_len, k) proj_k / proj_v at all 18 attentions,
    which it did not before (the two models start from different seeds).
    """
    from hepattn.keras.porting import port_keras_to_keras  # noqa: PLC0415

    cfg = U.generate_polaris_config(tmp_path / "polaris_hgq.yaml")
    inputs, _ = U.clic_dummy_batch()
    models = []
    for seed, quantized in ((1, False), (2, True)):
        torch.manual_seed(seed)
        model = U.instantiate(cfg, quantized=quantized)
        with torch.no_grad():
            model(inputs)
        models.append(model)
    kfloat, kquant = models

    def projections(model) -> list[torch.Tensor]:
        return [p for m in model.modules() if isinstance(m, KerasAttention) and m.attn_type == "linformer" for p in m.full_projections()]

    assert len(projections(kfloat)) == 2 * 18
    assert not any(starmap(torch.equal, zip(projections(kfloat), projections(kquant), strict=True)))
    port_keras_to_keras(kfloat, kquant)
    assert all(starmap(torch.equal, zip(projections(kfloat), projections(kquant), strict=True)))


# ---------------------------------------------------------------------- hybrid attention vs org Attention
@torch.no_grad()
def perturb_affine_params(module: torch.nn.Module, seed: int) -> None:
    """Non-default biases and norm affines, so a missing or swapped port of them changes outputs at O(0.1)."""
    gen = torch.Generator().manual_seed(seed)
    for name, param in module.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"bias", "in_proj_bias"} or (leaf == "weight" and param.dim() == 1):
            param.add_(0.1 * torch.randn(param.shape, generator=gen))


def make_inputs(shape: str, seed: int, sort: bool = False) -> dict:
    """Inputs of one call site (as tests/keras_native/test_masked_linformer.py). Key padding is last (event 0: 7
    padded, event 1: 1 padded); q_ca has an all-False query row.
    """
    q_len, kv_len, _, kind = SHAPES[shape]
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, q_len, DIM)).astype(np.float32)
    kv = q if shape in SELF else rng.standard_normal((B, kv_len, DIM)).astype(np.float32)
    kv_mask = attn_mask = idx = None
    if shape in {"q_ca", "enc", "enc_short"}:
        kv_mask = np.ones((B, kv_len), bool)
        kv_mask[0, -7:] = False
        kv_mask[1, -1:] = False
    if kind == "attend":
        attn_mask = rng.random((B, q_len, kv_len)) < 0.4
        attn_mask[0, 3] = False
    if sort:
        idx = np.stack([rng.permutation(kv_len) for _ in range(B)])
        assert (idx != np.arange(kv_len)).any()
    return {"q": q, "kv": kv, "kv_mask": kv_mask, "attn_mask": attn_mask, "kv_sort_idx": idx}


def torch_attention(shape: str, seed: int, **kwargs) -> Attention:
    torch.manual_seed(seed)
    tattn = Attention(DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=SHAPES[shape][2], linformer_k=K, **kwargs).eval()
    perturb_affine_params(tattn, seed)
    return tattn


def call(attn, x: dict, **kwargs) -> torch.Tensor:
    kv = None if x["kv"] is x["q"] else t(x["kv"])
    masks = {name: t(x[name]) for name in ("kv_mask", "attn_mask", "kv_sort_idx")}
    with torch.no_grad():
        return attn(t(x["q"]), kv, kv, **masks, **kwargs)


def hybrid_attention(shape: str, tattn: Attention | None, x: dict, spec=None, name: str = "attn", **kwargs) -> KerasAttention:
    """The hybrid twin; q_sa names its core as the decoder does. A first call builds the lazy core (and the quantized
    leaves), then the torch weights are ported.
    """
    _, _, seq_len, kind = SHAPES[shape]
    if kind == "project_unmasked":
        kwargs = {"linformer_core": kind, **kwargs}
    factory = LayerFactory(spec)
    with factory.scopes():
        kattn = KerasAttention(DIM, HEADS, attn_type="linformer", linformer_seq_len=seq_len, linformer_k=K, factory=factory, name=name, **kwargs)
    kattn.eval()
    call(kattn, x)
    assert kattn.core_kind == kind
    if tattn is not None:
        port_attention(tattn, kattn)
    return kattn


# measured max |hybrid - org|: q_ca 6.855e-7, kv_ca 6.557e-7, enc 6.258e-7, enc_short 4.470e-7, q_sa 6.258e-7; q_ca
# sorted 5.364e-7; distinct k and v: q_ca 4.768e-7, kv_ca 4.470e-7; value residual 4.470e-7 (both layers); batch of
# one <= 4.321e-7. Controls ($S/r4b/logs/controls.out): v = k 0.326 / 0.334; no sort 0.333; masks not gathered 0.0855;
# no value-residual mix 0.0780; mask of ones vs none at n < seq_len 0.120 (at n = seq_len 8.9e-8)
ATTN_TOL = 1.37e-6


@pytest.mark.parametrize("shape", list(SHAPES))
def test_linformer_attention_equals_org(shape):
    """qkv_norm, perturbed biases and norm affines, kv_mask / attn_mask / kv_sort_idx as at the org call site."""
    tattn = torch_attention(shape, 5, qkv_norm=True, norm="LayerNorm")
    x = make_inputs(shape, 6, sort=shape == "q_ca")
    kattn = hybrid_attention(shape, tattn, x, qkv_norm=True, norm="LayerNorm")
    assert_parity("hybrid.linformer", shape, call(tattn, x), call(kattn, x), atol=ATTN_TOL, rtol=0.0)


@pytest.mark.parametrize("shape", ["q_ca", "kv_ca"])
def test_linformer_cross_attention_with_distinct_values_equals_org(shape):
    """As the decoder calls them: q_ca(q_pe, k=kv_pe, v=kv), kv_ca(kv_pe, k=q_pe, v=q), so k and v differ."""
    tattn = torch_attention(shape, 25)
    x = make_inputs(shape, 26, sort=shape == "q_ca")
    kattn = hybrid_attention(shape, tattn, x)
    v = t(np.random.default_rng(27).standard_normal(x["kv"].shape).astype(np.float32))
    masks = {name: t(x[name]) for name in ("kv_mask", "attn_mask", "kv_sort_idx")}
    with torch.no_grad():
        ref = tattn(t(x["q"]), t(x["kv"]), v, **masks)
        got = kattn(t(x["q"]), t(x["kv"]), v, **masks)
        same = kattn(t(x["q"]), t(x["kv"]), t(x["kv"]), **masks)
    assert_parity("hybrid.linformer_distinct_v", shape, ref, got, atol=ATTN_TOL, rtol=0.0)
    assert (same - ref).abs().max() > 1e3 * ATTN_TOL  # control, v = k


def test_kv_sort_idx_gathers_keys_masks_and_attn_mask_columns():
    """Without the gather the q_ca output is O(1) off org; gathering k, v but not the masks is wrong too."""
    tattn = torch_attention("q_ca", 7)
    x = make_inputs("q_ca", 8, sort=True)
    kattn = hybrid_attention("q_ca", tattn, x)
    ref = call(tattn, x)
    assert_parity("hybrid.linformer", "q_ca.sorted", ref, call(kattn, x), atol=ATTN_TOL, rtol=0.0)
    assert (call(kattn, {**x, "kv_sort_idx": None}) - ref).abs().max() > 1e3 * ATTN_TOL
    only_kv = {**x, "kv_mask": x["kv_mask"][np.arange(B)[:, None], np.argsort(x["kv_sort_idx"])]}
    assert (call(kattn, only_kv) - ref).abs().max() > 1e3 * ATTN_TOL


def test_linformer_value_residual_two_layers_equal_org():
    """Encoder layers 0 and 1: layer 1 mixes layer 0's values before the sequence projection, as org."""
    x = make_inputs("enc", 11)
    t0 = torch_attention("enc", 9, value_residual=True, is_first_layer=True)
    t1 = torch_attention("enc", 10, value_residual=True, is_first_layer=False)
    k0 = hybrid_attention("enc", t0, x, name="a0", value_residual=True, is_first_layer=True)
    k1 = hybrid_attention("enc", t1, x, name="a1", value_residual=True, is_first_layer=False)
    iv_t: dict = {}
    iv_k: dict = {}
    mask = t(x["kv_mask"])
    with torch.no_grad():
        h_t = t0(t(x["q"]), kv_mask=mask, initial_values=iv_t)
        out_t = t1(h_t, kv_mask=mask, initial_values=iv_t)
        h_k = k0(t(x["q"]), kv_mask=mask, initial_values=iv_k)
        out_k = k1(h_k, kv_mask=mask, initial_values=iv_k)
        no_mix = k1(h_k, kv_mask=mask, initial_values=None)
    assert_parity("hybrid.linformer_value_residual", "layer0", h_t, h_k, atol=ATTN_TOL, rtol=0.0)
    assert_parity("hybrid.linformer_value_residual", "layer1", out_t, out_k, atol=ATTN_TOL, rtol=0.0)
    assert (no_mix - out_k).abs().max() > 1e3 * ATTN_TOL  # control: the mix is not a no-op


@pytest.mark.parametrize("shape", ["enc_short", "enc"])
def test_batch_of_one_all_valid_mask_differs_from_no_mask_below_seq_len(shape):
    """Plan 1.2: org passes kv_mask=None for B=1 with every key valid; a mask of ones turns the renorm on when
    n < seq_len. Both hybrid calls equal org; at n = seq_len the two agree with each other.
    """
    tattn = torch_attention(shape, 14)
    x = make_inputs(shape, 15)
    one = {**x, "q": x["q"][:1], "kv_mask": np.ones((1, SHAPES[shape][1]), bool)}
    one["kv"] = one["q"]
    kattn = hybrid_attention(shape, tattn, one)
    none = {**one, "kv_mask": None}
    got_ones, got_none = call(kattn, one), call(kattn, none)
    assert_parity("hybrid.linformer_b1", f"{shape}.ones", call(tattn, one), got_ones, atol=ATTN_TOL, rtol=0.0)
    assert_parity("hybrid.linformer_b1", f"{shape}.none", call(tattn, none), got_none, atol=ATTN_TOL, rtol=0.0)
    gap = (got_ones - got_none).abs().max()
    assert gap > 1e-2 if SHAPES[shape][1] < SHAPES[shape][2] else gap <= ATTN_TOL


def test_q_sa_keeps_the_torch_layout_of_rows_org_never_reads():
    """q_sa: the core holds E[:150] (no tail); rows 150-159 stay as buffers, so full_projections returns org's
    (160, 64) tensors. Changing those rows in org changes nothing (exactly); changing row 149 does.
    """
    tattn = torch_attention("q_sa", 12)
    x = make_inputs("q_sa", 13)
    kattn = hybrid_attention("q_sa", tattn, x)
    assert tuple(kattn.core.E.kernel.shape) == (150, K) and kattn.core.E.tail is None
    assert tuple(kattn.proj_k_dead.shape) == (10, K) and "proj_k_dead" in dict(kattn.named_buffers())
    for got, want in zip(kattn.full_projections(), (tattn.proj_k, tattn.proj_v), strict=True):
        assert torch.equal(got, want.detach())
    ref = call(tattn, x)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        tattn.proj_k[150:] += torch.randn(tattn.proj_k[150:].shape, generator=gen)
    assert torch.equal(call(tattn, x), ref)
    with torch.no_grad():
        tattn.proj_k[149] += torch.randn(K, generator=gen)
    assert (call(tattn, x) - ref).abs().max() > 1e-2  # positive control


@pytest.mark.parametrize("mode", ["float", "high"])
@pytest.mark.parametrize("shape", ["q_ca", "enc"])
def test_padded_tokens_do_not_reach_valid_outputs(shape, mode):
    """Valid outputs change by exactly 0.0 when padded tokens change (NaN in float); a valid token changes them."""
    tattn = torch_attention(shape, 16)
    x = make_inputs(shape, 17, sort=shape == "q_ca")
    kattn = hybrid_attention(shape, tattn, x, spec=None if mode == "float" else HIGH_SPEC)
    base = call(kattn, x)
    pad = ~x["kv_mask"]
    kv = np.where(pad[..., None], np.nan if mode == "float" else 100.0, x["kv"]).astype(np.float32)
    out = call(kattn, {**x, "kv": kv, "q": kv if shape in SELF else x["q"]})
    rows = torch.ones(base.shape[:2], dtype=torch.bool) if shape not in SELF else t(x["kv_mask"])  # real output tokens
    assert torch.equal(out[rows], base[rows])
    valid_kv = x["kv"].copy()
    valid_kv[0, 0] += 5.0
    changed = call(kattn, {**x, "kv": valid_kv, "q": valid_kv if shape in SELF else x["q"]})
    assert (changed[rows] - base[rows]).abs().max() > 1e-3  # positive control


@pytest.mark.parametrize("mode", ["float", "high"])
def test_key_outside_a_query_mask_does_not_reach_that_query(mode):
    """attend: perturbing a key query 0 does not see changes query 0 by exactly 0.0 and moves every query that sees it."""
    tattn = torch_attention("q_ca", 18)
    x = make_inputs("q_ca", 19)
    kattn = hybrid_attention("q_ca", tattn, x, spec=None if mode == "float" else HIGH_SPEC)
    j = int(np.flatnonzero(~x["attn_mask"][0, 0] & x["kv_mask"][0] & x["attn_mask"][0, 1])[0])
    kv = x["kv"].copy()
    kv[0, j] += 5.0
    base, out = call(kattn, x), call(kattn, {**x, "kv": kv})
    assert torch.equal(out[0, 0], base[0, 0])
    seen = t(x["attn_mask"][0, :, j])
    assert (out[0, seen] - base[0, seen]).abs().max(-1).values.min() > 1e-4


# measured max |hybrid - org float| on q_ca (qkv_norm, perturbed norm affines, zero q/k/v/out biases, sorted): high
# 2.146e-2, coarse 0.7650 (35.7x); native measured 2.147e-2 / 0.7650 on the same case (R2)
HIGH_TOL = 4.29e-2


def test_paired_delta_high_vs_coarse():
    """The quantized hybrid attention: close to org float at the high spec, >= 10x further at the coarse one."""
    tattn = torch_attention("q_ca", 0, qkv_norm=True, norm="LayerNorm")
    with torch.no_grad():
        tattn.in_proj_bias.zero_()  # no QuantSpec place reaches the QDense bias quantizers (R2 fix round 2)
        tattn.out_proj.bias.zero_()
    x = make_inputs("q_ca", 22, sort=True)
    ref = call(tattn, x)
    high = call(hybrid_attention("q_ca", tattn, x, spec=HIGH_SPEC, qkv_norm=True, norm="LayerNorm"), x)
    coarse = call(hybrid_attention("q_ca", tattn, x, spec=COARSE_SPEC, qkv_norm=True, norm="LayerNorm"), x)
    err_high, err_coarse = float((high - ref).abs().max()), float((coarse - ref).abs().max())
    assert err_high <= HIGH_TOL
    assert coarse.std(dim=1).mean() > 0.05, "the coarse spec must still depend on the input"
    assert err_coarse >= 10 * err_high, (err_high, err_coarse)


@pytest.mark.parametrize("spec_name", list(C.SPECS))
@pytest.mark.parametrize("shape", ["q_ca", "enc"])
def test_train_mode_attention_runs_the_core_in_train_mode(shape, spec_name, monkeypatch):
    """QAT calls the hybrid attention in train mode. There HGQ2's WRAP-mode quantizers fit their integer bits to the
    data (a state update) instead of wrapping, so from the same core state a train-mode and an eval-mode core call
    differ. Inside a train-mode forward, the core output and the core state afterwards equal those of the core run
    directly with training=True on the same inputs and from the same state, bit for bit.
    """
    x = make_inputs(shape, 41, sort=shape == "q_ca")
    kattn = hybrid_attention(shape, torch_attention(shape, 40), x, spec=C.SPECS[spec_name])
    core = kattn.core
    start = [v.value.detach().clone() for v in core.variables]

    def state() -> list[torch.Tensor]:
        return [v.value.detach().clone() for v in core.variables]

    def restore() -> None:
        for v, value in zip(core.variables, start, strict=True):
            v.assign(value)

    seen = []
    original = type(core).call

    def record(self, *args, training=None):
        out = original(self, *args, training=training)
        seen.append((args, out))
        return out

    monkeypatch.setattr(type(core), "call", record)
    kattn.train()
    call(kattn, x)
    monkeypatch.undo()
    ((args, out),) = seen
    after = state()
    with torch.no_grad():
        restore()
        train = core(*args, training=True)
        assert all(starmap(torch.equal, zip(state(), after, strict=True)))
        restore()
        evaluated = core(*args, training=False)
    assert not all(starmap(torch.equal, zip(after, start, strict=True))), "the train-mode call fitted no integer bits"
    assert torch.equal(out, train)
    # control, the core in eval mode from the same state (measured max |train - eval|, $S/r4b_fix/logs/train_mode.out):
    # high q_ca 4.33e-4, enc 3.94e-4; prod q_ca 37.3, enc 45.4 (the prod datalanes wrap in eval mode)
    assert float((train - evaluated).abs().max()) > {"high": 1e-4, "prod": 1.0}[spec_name]


def test_core_leaves_built_at_the_first_forward_take_the_factory_spec():
    """The hybrid builds the core inside forward, outside the model's construction scopes; build_core enters them
    again, so the leaves get the spec: at the high spec E/F weights at the fractional-bit cap 7 and the qk input lanes
    at (i, f) = (8, 16). HGQ2's defaults would give f = 6 and (2, 6) (R2, test_core_leaves_take_the_spec_...).
    """
    x = make_inputs("q_ca", 33)
    core = hybrid_attention("q_ca", None, x, spec=HIGH_SPEC).core

    def values(quantizer, name: str) -> set[float]:
        return {float(v) for v in np.ravel(np.asarray(getattr(quantizer.quantizer, name).detach().cpu()))}

    for quantizer in (core.E.proj.kq, core.F.proj.kq):
        assert values(quantizer, "f") == {7.0}
    for lane in core.qk.iq:
        assert (values(lane, "i"), values(lane, "f")) == ({8.0}, {16.0})


def test_calls_that_do_not_match_the_built_core_raise():
    x = make_inputs("q_ca", 31)
    kattn = hybrid_attention("q_ca", None, x)
    with pytest.raises(ValueError, match="needs an attn_mask"):
        call(kattn, {**x, "attn_mask": None})
    short = {**x, "kv": x["kv"][:, :100], "kv_mask": x["kv_mask"][:, :100], "attn_mask": x["attn_mask"][..., :100]}
    with pytest.raises(ValueError, match="the built"):
        call(kattn, short)
    y = make_inputs("q_sa", 32)
    unmasked = hybrid_attention("q_sa", None, y)
    with pytest.raises(ValueError, match="without mask support"):
        call(unmasked, {**y, "kv_mask": np.ones((B, 150), bool)})
    long = {**x, "kv": np.zeros((B, 170, DIM), np.float32), "kv_mask": None, "attn_mask": np.ones((B, 150, 170), bool)}
    with pytest.raises(ValueError, match="exceeds linformer_seq_len"):
        hybrid_attention("q_ca", None, long)
