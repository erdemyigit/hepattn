"""S6a gate: the config loader reads the four config forms and they build the same native model.

Forms: the raw org YAMLs (``base.yaml`` + ``linformer.yaml``), ``pflow_hgq.yaml``, the generated Polaris config
(``polaris/03_make_config.py``, run here into a temp dir), and jsonargparse-normalised run configs (fixtures in
``s6_fixtures/``, written by ``make_normalised_configs.py`` with LightningCLI ``--print_config``, the form
``SaveConfig`` writes). Same model = same variable keys, shapes and dtypes, and, with one model's values copied
into the other, bit-identical forward outputs on two CLIC dummy events, at the production dims of each config.

Controls: the same check fails for a model without input sorting (forward only) and for k = 32 (shapes); native
``from_config`` without the loader's normalisation rejects the saved config (R-critique open issue).

The run sections (optimizer and LR schedule, checkpoint policy, data, seed, BetaPID) are pinned per family and equal
between the raw and saved forms; ``model_hash`` is equal for every form of one model and for a relocated copy, and
changes with a model option, a non-default value spelled out, or the scale dict's content.
"""

import ast
import copy
import dataclasses
import os
import subprocess
import sys
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

import hepattn
from hepattn.keras_native.checkpoint import state_keys
from hepattn.keras_native.config import (
    BETA_PID_ARGS,
    BETA_SCHEDULER_ARGS,
    CheckpointCallback,
    CheckpointConfig,
    ConfigError,
    OptimizerConfig,
    config_hash,
    deep_merge,
    load_config,
    model_description,
    normalise_model_section,
)
from hepattn.keras_native.maskformer import from_config
from hepattn.keras_native.objective import clic_specs

from .golden.mf_cases import clic_dummy_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = Path(hepattn.__file__).parent / "experiments/clic/configs"
FIXTURES = Path(__file__).parent / "s6_fixtures"
ORG = [CONFIGS / "base.yaml", CONFIGS / "linformer.yaml"]


@pytest.fixture(scope="module")
def polaris_generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("polaris") / "polaris_hgq.yaml"
    env = {**os.environ, "HGQ_CONFIG_OUT": str(out), "DATA_DIR": "/eagle/hgcal-maskformer-fpga/clic_data", "CKPT_DIR": str(out.parent / "ckpts")}
    subprocess.run([sys.executable, str(REPO_ROOT / "polaris/03_make_config.py")], env=env, check=True, capture_output=True, text=True)
    return out


def form_paths(form: str, generated: Path):
    return {
        "org_raw": ORG,
        "org_normalised": FIXTURES / "org_normalised.yaml",
        "pflow_hgq_raw": CONFIGS / "pflow_hgq.yaml",
        "pflow_hgq_normalised": FIXTURES / "pflow_hgq_normalised.yaml",
        "polaris_generated": generated,
        "polaris_normalised": FIXTURES / "polaris_normalised.yaml",
    }[form]


FORMS = ["org_raw", "org_normalised", "pflow_hgq_raw", "pflow_hgq_normalised", "polaris_generated", "polaris_normalised"]
# expected (optimizer, quantized, accumulate_grad_batches, PeriodicEBOPs period)
EXPECTED = {
    "org": ("Lion", False, 1, 1),
    "pflow_hgq": ("AdamW", True, 1, 1),
    "polaris": ("AdamW", True, 4, 8),
}
# As written in base.yaml (Lion, lrs_config), pflow_hgq.yaml (AdamW, max 5e-5) and polaris/03_make_config.py (LR_MAX
# 5e-5, HGQ_ACCUM 4, HGQ_BATCH 32, num_workers 8, step ModelCheckpoint every 1000 with save_last).
OPTIMIZER = {
    "org": OptimizerConfig("Lion", 1e-6, 8e-5, 1e-6, 0.05, 1e-4, skip_scheduler=False, clip_norm=0.1, accumulate_grad_batches=1),
    "pflow_hgq": OptimizerConfig("AdamW", 1e-6, 5e-5, 1e-6, 0.05, 1e-4, skip_scheduler=False, clip_norm=0.1, accumulate_grad_batches=1),
    "polaris": OptimizerConfig("AdamW", 1e-6, 5e-5, 1e-6, 0.05, 1e-4, skip_scheduler=False, clip_norm=0.1, accumulate_grad_batches=4),
}
# (batch_size, num_workers, num_train, num_val, num_test, every_n_train_steps)
DATA_AND_CADENCE = {"org": (512, 16, -1, 5000, -1, None), "pflow_hgq": (128, 16, -1, 5000, -1, None), "polaris": (32, 8, -1, 5000, -1, 1000)}
# BetaPID of the generated config: INIT_BETA and MAX_BETA times HGQ_EBOPS_EVERY (8)
BETA_PID = {
    "pflow_hgq": {"target_ebops": 1e15, "init_beta": 2.6e-10, "max_beta": 1e-7},
    "polaris": {"target_ebops": 1e15, "init_beta": 2.08e-9, "max_beta": 8e-7},
}


def checkpoint_dirpath(form: str, generated: Path) -> str | None:
    """The generated config writes CKPT_DIR (set by the fixture); its saved fixture has the Polaris path."""
    return {"polaris_generated": str(generated.parent / "ckpts"), "polaris_normalised": "/polaris/home/hepattn/hgq_ckpts"}.get(form)


# What the callbacks do, not their YAML text (lightning 2.5.2 model_checkpoint.py, hepattn/callbacks/checkpoint.py):
# hepattn Checkpoint passes save_top_k=-1 (keep every epoch-end checkpoint) and its setup() sets save_last = False and
# dirpath = <log_dir>/ckpts, although base.yaml and pflow_hgq.yaml say save_last: true. No trigger set -> every_n_epochs=1.
EPOCH_CALLBACK = CheckpointCallback(
    monitor="val/loss", mode="min", save_top_k=-1, save_last=False, every_n_train_steps=0, every_n_epochs=1, dirpath=None
)


def checkpoint_config(form: str, generated: Path) -> CheckpointConfig:
    if not form.startswith("polaris"):
        return CheckpointConfig(epoch=EPOCH_CALLBACK, step=None)
    # polaris/03_make_config.py: ModelCheckpoint(every_n_train_steps=1000, save_last=True, save_top_k=1, monitor=None)
    step = CheckpointCallback(
        monitor=None,
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=1000,
        every_n_epochs=0,
        dirpath=checkpoint_dirpath(form, generated),
    )
    return CheckpointConfig(epoch=EPOCH_CALLBACK, step=step)


def load(form: str, generated: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the packaged scale dict fallback of configs saved elsewhere
        return load_config(form_paths(form, generated))


@pytest.fixture(scope="module")
def batch():
    inputs, _ = clic_dummy_batch(2)
    return inputs


_MODELS: dict[str, object] = {}


def built(name: str, section: dict, inputs: dict):
    if name not in _MODELS:
        model = from_config(section)
        model(inputs, training=False)
        _MODELS[name] = model
    return _MODELS[name]


def leaves(tree) -> dict[str, np.ndarray]:
    return {jax.tree_util.keystr(path): np.asarray(x) for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


def assert_same_model(a, b, inputs) -> int:
    """Same keys / shapes / dtypes; with a's values in b, identical outputs. Returns the number of variables."""
    assert state_keys(a) == state_keys(b), "variable keys differ"
    assert [(tuple(v.shape), str(v.dtype)) for v in a.variables] == [(tuple(v.shape), str(v.dtype)) for v in b.variables], (
        "variable shapes or dtypes differ"
    )
    for va, vb in zip(a.variables, b.variables, strict=True):
        vb.assign(va.value)
    out_a, out_b = leaves(a(inputs, training=False)), leaves(b(inputs, training=False))
    assert out_a.keys() == out_b.keys()
    different = [k for k in out_a if not np.array_equal(out_a[k], out_b[k], equal_nan=out_a[k].dtype.kind == "f")]
    assert different == [], f"forward outputs differ: {different}"
    return len(a.variables)


@pytest.mark.parametrize("form", FORMS)
def test_loader_reads_every_form(form, polaris_generated):
    rc = load(form, polaris_generated)
    family = form.rsplit("_", 1)[0]
    optimizer, quantized, accumulate, ebops_every = EXPECTED[family]
    assert rc.task_specs == clic_specs()
    assert rc.matcher_n_jobs == 16
    assert rc.optimizer.name == optimizer
    assert rc.optimizer.clip_norm == 0.1
    assert rc.optimizer.accumulate_grad_batches == accumulate
    assert (rc.quant.spec is not None) == quantized
    assert rc.quant.ebops_every_n_steps == ebops_every
    assert (rc.quant.beta_pid is not None) == quantized
    assert rc.checkpoint.epoch.monitor == "val/loss"
    assert rc.optimizer == OPTIMIZER[family]
    batch_size, workers, num_train, num_val, num_test, every = DATA_AND_CADENCE[family]
    assert (rc.data.batch_size, rc.data.num_workers, rc.data.num_train, rc.data.num_val, rc.data.num_test) == (
        batch_size,
        workers,
        num_train,
        num_val,
        num_test,
    )
    assert rc.checkpoint == checkpoint_config(form, polaris_generated)
    assert (rc.checkpoint.step.every_n_train_steps if rc.checkpoint.step else None) == every
    assert rc.seed == 42
    if quantized:
        assert rc.quant.beta_pid == {
            "p": 1.0,
            "i": 0.5,
            "d": 0.0,
            "warmup_epochs": 2,
            "min_beta": 0.0,
            "damp_beta_on_target": 0.0,
            **BETA_PID[family],
        }
    assert Path(rc.data.dataset_kwargs["scale_dict_path"]).is_file()
    for task in rc.model_section["model"]["init_args"]["tasks"]["init_args"]["modules"]:
        if "scale_dict_path" in task["init_args"]:
            assert Path(task["init_args"]["scale_dict_path"]).is_file()


@pytest.mark.parametrize(
    ("first", "second"), [("org_raw", "org_normalised"), ("pflow_hgq_raw", "pflow_hgq_normalised"), ("polaris_generated", "polaris_normalised")]
)
def test_raw_and_saved_forms_give_the_same_run_sections(first, second, polaris_generated):
    """Besides the model: optimizer, checkpoint policy (but the host-specific dirpath), data, quant callbacks, seed."""
    a, b = load(first, polaris_generated), load(second, polaris_generated)
    assert a.optimizer == b.optimizer
    assert a.checkpoint.epoch == b.checkpoint.epoch
    assert (a.checkpoint.step is None) == (b.checkpoint.step is None)
    if a.checkpoint.step is not None:
        assert dataclasses.replace(a.checkpoint.step, dirpath=None) == dataclasses.replace(b.checkpoint.step, dirpath=None)
    assert (a.quant.beta_pid, a.quant.beta_scheduler, a.quant.ebops_every_n_steps) == (
        b.quant.beta_pid,
        b.quant.beta_scheduler,
        b.quant.ebops_every_n_steps,
    )
    assert a.seed == b.seed
    assert dataclasses.replace(a.data, dataset_kwargs={}) == dataclasses.replace(b.data, dataset_kwargs={})
    # the saved form adds the CLICDataset defaults (max_nodes, remove_wrong_idxs, ...); the raw keys agree
    assert {k: b.data.dataset_kwargs[k] for k in a.data.dataset_kwargs} == a.data.dataset_kwargs


def hybrid_init_defaults(class_name: str) -> dict:
    """Constructor defaults of a hybrid callback, read from the source (importing it needs lightning + the torch backend)."""
    tree = ast.parse((Path(hepattn.__file__).parent / "keras" / "callbacks.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    args = init.args.args[1:]
    defaults = [None] * (len(args) - len(init.args.defaults)) + [ast.literal_eval(d) for d in init.args.defaults]
    return {a.arg: d for a, d in zip(args, defaults, strict=True)}


@pytest.mark.parametrize(("class_name", "table"), [("BetaPID", BETA_PID_ARGS), ("BetaScheduler", BETA_SCHEDULER_ARGS)])
def test_beta_callback_defaults_are_the_hybrid_ones(class_name, table):
    reference = hybrid_init_defaults(class_name)
    assert {k: default for k, (_, default) in table.items()} == reference
    assert reference  # the source was found and parsed


def test_beta_pid_arguments_are_completed_and_checked(polaris_generated, tmp_path):
    cfg = small_raw(polaris_generated)
    pid = next(c for c in cfg["trainer"]["callbacks"] if c["class_path"].endswith("BetaPID"))["init_args"]
    pid["target_ebops"] = "1.0e15"  # YAML 1.1 reads this spelling as a string; the hybrid casts it
    assert load_config(write(tmp_path / "a.yaml", cfg)).quant.beta_pid["target_ebops"] == 1e15
    pid["typo_gain"] = 1.0
    with pytest.raises(ConfigError, match="typo_gain"):
        load_config(write(tmp_path / "b.yaml", cfg))
    del pid["typo_gain"], pid["init_beta"]
    with pytest.raises(ConfigError, match="init_beta"):
        load_config(write(tmp_path / "c.yaml", cfg))


# measured variable counts (S6a log): float org model; quantized models (HGQ2 quantizer state included)
VARIABLE_COUNTS = {"org_raw": 335, "pflow_hgq_raw": 3967, "polaris_generated": 4223}
PAIRS = [
    ("org_raw", "org_normalised"),
    ("pflow_hgq_raw", "pflow_hgq_normalised"),
    ("polaris_generated", "polaris_normalised"),
]


@pytest.mark.parametrize(("first", "second"), PAIRS)
def test_raw_and_saved_forms_build_the_same_model(first, second, polaris_generated, batch):
    a = built(first, load(first, polaris_generated).model_section, batch)
    b = built(second, load(second, polaris_generated).model_section, batch)
    count = assert_same_model(a, b, batch)
    print(f"{first} vs {second}: {count} variables")
    assert count == VARIABLE_COUNTS[first]


def test_generated_polaris_config_is_the_org_model_with_the_quant_spec(polaris_generated, batch):
    """The generated config is base.yaml + linformer.yaml (org) plus the HGQ2 spec: the same native model."""
    org = copy.deepcopy(load("org_raw", polaris_generated).model_section)
    polaris = load("polaris_generated", polaris_generated)
    org["model"]["init_args"]["quant"] = polaris.quant.spec
    a = built("org_raw+quant", org, batch)
    b = built("polaris_generated", polaris.model_section, batch)
    assert assert_same_model(a, b, batch) == 4223  # measured: the quantized org model (S6a log)


def test_control_forward_check_catches_a_model_without_sorting(polaris_generated, batch):
    section = copy.deepcopy(load("org_normalised", polaris_generated).model_section)
    section["model"]["init_args"]["input_sort_field"] = None
    a = built("org_raw", load("org_raw", polaris_generated).model_section, batch)
    unsorted = built("org_unsorted", section, batch)
    assert state_keys(a) == state_keys(unsorted)
    with pytest.raises(AssertionError, match="forward outputs differ"):
        assert_same_model(a, unsorted, batch)


def test_control_shape_check_catches_another_linformer_k(polaris_generated, batch):
    section = copy.deepcopy(load("org_normalised", polaris_generated).model_section)
    section["model"]["init_args"]["encoder"]["init_args"]["attn_kwargs"]["linformer_k"] = 32
    a = built("org_raw", load("org_raw", polaris_generated).model_section, batch)
    with pytest.raises(AssertionError, match="shapes or dtypes differ"):
        assert_same_model(a, built("org_k32", section, batch), batch)


def test_control_saved_config_needs_the_normalisation():
    """Without the loader, native ``from_config`` refuses a saved config (``activation: {class_path: ...}``)."""
    saved = yaml.safe_load((FIXTURES / "org_normalised.yaml").read_text())["model"]
    with pytest.raises(TypeError, match="activation must be a name"):
        from_config(saved)
    normalised = normalise_model_section(saved)
    net = normalised["model"]["init_args"]["input_nets"]["init_args"]["modules"][0]["init_args"]["net"]["init_args"]
    assert net["activation"] == "SwiGLU"
    task_net = normalised["model"]["init_args"]["tasks"]["init_args"]["modules"][0]["init_args"]["net"]["init_args"]
    assert task_net["activation"] == "SiLU"
    assert "class_path" not in normalised["model"]["init_args"]["decoder"]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"class_path": "torch.nn.Softplus"}, "not supported"),
        ({"class_path": "torch.nn.SiLU", "init_args": {"inplace": True}}, "inplace"),
        ({"class_path": "torch.nn.SiLU", "init_args": {"foo": 1}}, "inplace"),
    ],
)
def test_unsupported_activation_forms_raise(spec, message):
    with pytest.raises(ConfigError, match=message):
        normalise_model_section({"model": {"init_args": {"net": {"activation": spec}}}})


def test_unsupported_decoder_class_raises():
    with pytest.raises(ConfigError, match="MaskFormerDecoder"):
        normalise_model_section({"model": {"init_args": {"decoder": {"class_path": "x.OtherDecoder", "init_args": {}}}}})


def write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg))
    return path


def small_raw(polaris_generated) -> dict:
    return yaml.safe_load(Path(polaris_generated).read_text())


def periodic_ebops(cfg: dict) -> dict:
    return next(c for c in cfg["trainer"]["callbacks"] if c["class_path"].endswith("PeriodicEBOPs"))["init_args"]


def checkpoint_args(cfg: dict, name: str) -> dict:
    entry = next(c for c in cfg["trainer"]["callbacks"] if c["class_path"].rsplit(".", 1)[-1] == name)
    return entry.setdefault("init_args", {})


def test_hepattn_checkpoint_settings_overridden_by_its_setup_are_ignored(polaris_generated, tmp_path):
    """``Checkpoint.setup`` sets ``save_last = False`` and ``dirpath = <log_dir>/ckpts``; a step trigger passes through."""
    cfg = small_raw(polaris_generated)
    checkpoint_args(cfg, "Checkpoint").update({"dirpath": "/elsewhere", "save_last": True})
    assert load_config(write(tmp_path / "a.yaml", cfg)).checkpoint.epoch == EPOCH_CALLBACK
    checkpoint_args(cfg, "Checkpoint")["every_n_train_steps"] = 1000  # the commented-out line of base.yaml
    epoch = load_config(write(tmp_path / "b.yaml", cfg)).checkpoint.epoch
    assert epoch == dataclasses.replace(EPOCH_CALLBACK, every_n_train_steps=1000, every_n_epochs=0)


def test_optimizer_name_is_matched_case_insensitively(polaris_generated, tmp_path):
    """As ``self.optimizer.lower()`` in ``models/wrapper.py``; the canonical name is stored."""
    cfg = small_raw(polaris_generated)
    cfg["model"]["optimizer"] = "lion"
    rc = load_config(write(tmp_path / "c.yaml", cfg))
    assert rc.optimizer.name == "Lion"
    assert rc.optimizer.epsilon == 1e-8  # torch AdamW's eps, not keras' 1e-7


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c["model"].__setitem__("optimizer", "Adam"), "optimizer 'Adam'"),
        (lambda c: c["model"]["lrs_config"].pop("weight_decay"), "weight_decay"),
        (lambda c: checkpoint_args(c, "ModelCheckpoint").__setitem__("every_n_epochs", 1), "mutually exclusive"),
        (lambda c: checkpoint_args(c, "ModelCheckpoint").__setitem__("save_top_k", 3), "needs a monitor"),
        (lambda c: checkpoint_args(c, "Checkpoint").__setitem__("save_on_train_epoch_end", True), "save_on_train_epoch_end"),
        (lambda c: c["trainer"].__setitem__("gradient_clip_algorithm", "value"), "norm clipping"),
        (
            lambda c: c["trainer"]["callbacks"].append({"class_path": "hepattn.keras.callbacks.BetaScheduler", "init_args": {}}),
            "BetaPID and BetaScheduler",
        ),
        (lambda c: periodic_ebops(c).__setitem__("every_n_steps", 0), "every_n_steps"),
        (lambda c: c["trainer"]["callbacks"].append(c["trainer"]["callbacks"][0]), "registered twice"),
    ],
)
def test_unsupported_run_options_raise(edit, message, polaris_generated, tmp_path):
    cfg = small_raw(polaris_generated)
    load_config(write(tmp_path / "ok.yaml", cfg))  # control: unedited, it loads
    edit(cfg)
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path / "c.yaml", cfg))


def test_scale_dict_resolution(polaris_generated, tmp_path):
    cfg = small_raw(polaris_generated)
    # relative to a parent of the config file
    (tmp_path / "sub" / "configs").mkdir(parents=True)
    local = tmp_path / "configs" / "clic_var_transform.yaml"
    local.parent.mkdir()
    local.write_text((CONFIGS / "clic_var_transform.yaml").read_text())
    cfg["data"]["scale_dict_path"] = "configs/clic_var_transform.yaml"
    rc = load_config(write(tmp_path / "sub" / "configs" / "c.yaml", cfg))
    assert Path(rc.data.dataset_kwargs["scale_dict_path"]) == local
    # an absolute path from another host falls back to the packaged file, with a warning
    cfg["data"]["scale_dict_path"] = "/nowhere/clic_var_transform.yaml"
    with pytest.warns(UserWarning, match="packaged"):
        rc = load_config(write(tmp_path / "c2.yaml", cfg))
    assert Path(rc.data.dataset_kwargs["scale_dict_path"]) == CONFIGS / "clic_var_transform.yaml"
    cfg["data"]["scale_dict_path"] = "/nowhere/no_such_scale_dict.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(write(tmp_path / "c3.yaml", cfg))


def test_model_hash_follows_the_model_section_only(polaris_generated, tmp_path):
    cfg = small_raw(polaris_generated)
    base = load_config(write(tmp_path / "a.yaml", cfg)).model_hash
    cfg["data"]["batch_size"] = 7
    cfg["model"]["lrs_config"]["max"] = 1e-4  # a resume with another LR schedule or optimizer is the same model
    cfg["model"]["optimizer"] = "Lion"
    cfg["model"]["model"]["init_args"]["matcher"]["init_args"]["verbose"] = True
    assert load_config(write(tmp_path / "b.yaml", cfg)).model_hash == base
    cfg["model"]["model"]["init_args"]["encoder"]["attn_kwargs"]["linformer_k"] = 32
    assert load_config(write(tmp_path / "c.yaml", cfg)).model_hash != base
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


@pytest.mark.parametrize(("first", "second"), PAIRS)
def test_model_hash_is_the_same_for_every_form_of_one_model(first, second, polaris_generated):
    """The saved form spells out defaults, other class-path spellings and floats; the polaris fixture's scale dict
    path is another host's (packaged fallback). Controls: a model change in the saved form changes the hash.
    """
    a, b = load(first, polaris_generated), load(second, polaris_generated)
    assert a.model_section != b.model_section  # the forms differ as text
    assert a.model_hash == b.model_hash
    init_args = b.model_section["model"]["init_args"]
    for edit in (
        lambda c: c.__setitem__("input_sort_field", None if c.get("input_sort_field") else "phi"),
        lambda c: c["tasks"]["init_args"]["modules"][1]["init_args"].__setitem__("mask_attn", False),  # a default spelled out
        lambda c: c["tasks"]["init_args"]["modules"][0]["init_args"]["net"]["init_args"].__setitem__("activation", "GELU"),
    ):
        changed = copy.deepcopy(b.model_section)
        edit(changed["model"]["init_args"])
        assert config_hash(model_description(changed)) != b.model_hash
    assert init_args["tasks"]["init_args"]["modules"][1]["init_args"]["mask_attn"] is True


def test_model_hash_follows_the_scale_dict_content_not_its_path(polaris_generated, tmp_path):
    cfg = small_raw(polaris_generated)
    base = load_config(write(tmp_path / "a.yaml", cfg)).model_hash
    (tmp_path / "copy").mkdir()
    text = (CONFIGS / "clic_var_transform.yaml").read_text()
    (tmp_path / "copy" / "clic_var_transform.yaml").write_text(text)
    tasks = cfg["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]
    regression = next(t for t in tasks if "scale_dict_path" in t["init_args"])["init_args"]
    regression["scale_dict_path"] = str(tmp_path / "copy" / "clic_var_transform.yaml")
    assert load_config(write(tmp_path / "b.yaml", cfg)).model_hash == base  # a byte-identical copy elsewhere
    assert "max: 15.136" in text
    (tmp_path / "copy" / "clic_var_transform.yaml").write_text(text.replace("max: 15.136", "max: 15.137"))
    assert load_config(write(tmp_path / "c.yaml", cfg)).model_hash != base  # control: other content


def test_deep_merge_replaces_lists_and_merges_dicts():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    over = {"a": {"y": [3]}, "c": 2}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}
