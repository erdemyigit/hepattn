"""Run configuration for the keras_native driver, read from the hepattn CLIC YAML files.

Four forms are read (plan section 3.7; ``docs/hgq/port_log/S6a.md``):

1. the raw CLIC YAMLs, several merged in order as LightningCLI merges several ``--config`` files (org: ``base.yaml``,
   then ``linformer.yaml``; ``merge_configs``: sections and class init args merge key by key (an overlay may give a
   class's init args as a plain dict once an earlier file gave its ``class_path``), lists, scalars and dict-typed
   arguments such as ``attn_kwargs`` are replaced whole, and forms LightningCLI rejects or merges in a way the loader
   does not reproduce raise ``ConfigError``; ``polaris/03_make_config.py`` merges every dict key by key, which gives the
   same result for org's overlay);
2. ``pflow_hgq.yaml`` (Helen's hybrid config);
3. the generated ``polaris_hgq.yaml`` (``polaris/03_make_config.py``; encoder as a plain dict, ``quant`` block);
4. a jsonargparse-normalised run config (the ``config.yaml`` that ``SaveConfig`` writes, or ``--print_config``):
   defaults expanded, and activations / the decoder in ``class_path`` + ``init_args`` form.

``normalise_model_section`` maps form 4 onto what ``maskformer.from_config`` takes: an activation
``{"class_path": "torch.nn.SiLU", "init_args": {"inplace": false}}`` becomes ``"SiLU"`` and the decoder's
``class_path`` wrapper is removed. The model code is not changed; the expanded default keys are passed through,
and the native layers either accept them or raise.

Relative ``scale_dict_path`` values are resolved against the working directory, then the config file's directory
and its parents, then the packaged CLIC ``configs`` directory (by file name). Data file paths are not resolved.

``RunConfig.model_hash`` (for ``checkpoint.load_checkpoint(expect_config_hash=...)``) hashes ``model_description``:
the same for every form of one model config and for a copy on another host, different for another model option or
scale-dict content. The LR schedule, optimizer, matcher and data settings are not part of it.
"""

import copy
import hashlib
import inspect
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hepattn.keras_native.decoder import MaskFormerDecoder
from hepattn.keras_native.dense import Dense
from hepattn.keras_native.encoder import Encoder, EncoderLayer
from hepattn.keras_native.input import FourierPositionEncoder, InputNet
from hepattn.keras_native.maskformer import MaskFormer, from_config
from hepattn.keras_native.objective import TaskSpec, matcher_n_jobs_from_config, specs_from_config
from hepattn.keras_native.tasks import TASK_CLASSES

# not an import: the packaged CLIC configs directory, last fallback for scale dicts
CLIC_CONFIGS = Path(__file__).resolve().parents[1] / "experiments" / "clic" / "configs"
ACTIVATION_KEYS = {"activation", "final_activation"}
# jsonargparse class paths of the activations the native Dense takes as names (dense.resolve_activation)
ACTIVATION_CLASSES = {"SiLU", "ReLU", "GELU", "Sigmoid", "Tanh", "SwiGLU"}
DECODER_CLASSES = {"MaskFormerDecoder"}
OPTIMIZERS = {"AdamW", "Lion"}
BETA_CALLBACKS = {"BetaPID", "BetaScheduler"}


class ConfigError(ValueError):
    """A config that the keras_native driver cannot run."""


def read_yaml(paths: str | Path | list[str | Path]) -> dict:
    """One YAML file, or several merged in order with ``merge_configs`` (as several LightningCLI ``--config``).

    Raises:
        ConfigError: If no file is given or a file's top level is not a mapping.
    """
    paths = [paths] if isinstance(paths, str | Path) else list(paths)
    if not paths:
        raise ConfigError("no config file given")
    cfg: dict = {}
    for path in paths:
        loaded = yaml.safe_load(Path(path).read_text())
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: the top level is not a mapping")
        cfg = merge_configs(cfg, loaded)
    return cfg


def _short(class_path: str) -> str:
    return class_path.rsplit(".", 1)[-1]


def _activation_name(spec: dict) -> str:
    """``{"class_path": ..., "init_args": {...}}`` of an activation -> its class name.

    Raises:
        ConfigError: For a class the native Dense does not take, or init args other than ``inplace: false``.
    """
    extra = set(spec) - {"class_path", "init_args"}
    name = _short(spec["class_path"])
    init_args = dict(spec.get("init_args") or {})
    if extra or name not in ACTIVATION_CLASSES:
        raise ConfigError(f"activation {spec} is not supported (supported: {sorted(ACTIVATION_CLASSES)})")
    if init_args.pop("inplace", False) or init_args:
        raise ConfigError(f"activation {spec}: only 'inplace: false' is supported as init args")
    return name


def _normalise(node: Any, key: str | None = None) -> Any:
    if isinstance(node, list):
        return [_normalise(item) for item in node]
    if not isinstance(node, dict):
        return node
    if key in ACTIVATION_KEYS and "class_path" in node:
        return _activation_name(node)
    if key == "decoder" and "class_path" in node:
        if _short(node["class_path"]) not in DECODER_CLASSES or set(node) - {"class_path", "init_args"}:
            raise ConfigError(f"decoder {node.get('class_path')!r} is not supported (only MaskFormerDecoder)")
        return _normalise(dict(node.get("init_args") or {}))
    return {k: _normalise(v, k) for k, v in node.items()}


def normalise_model_section(model_section: dict) -> dict:
    """The ``model:`` section with jsonargparse ``class_path`` forms mapped to what ``from_config`` takes (a copy)."""
    return _normalise(copy.deepcopy(model_section))


def resolve_file(path: str, search_dirs: list[Path]) -> str:
    """An existing file for ``path``: as given, then under each of ``search_dirs``, then ``CLIC_CONFIGS/<name>``.

    Raises:
        FileNotFoundError: If none exists.
    """
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    if not candidate.is_absolute():
        for root in search_dirs:
            if (root / candidate).is_file():
                return str(root / candidate)
    packaged = CLIC_CONFIGS / candidate.name
    if packaged.is_file():
        warnings.warn(f"{path} not found; using the packaged {packaged}", stacklevel=2)
        return str(packaged)
    raise FileNotFoundError(f"{path} not found (searched {[str(d) for d in search_dirs]} and {CLIC_CONFIGS})")


def _resolve_scale_dicts(node: Any, search_dirs: list[Path]) -> Any:
    if isinstance(node, list):
        return [_resolve_scale_dicts(item, search_dirs) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        k: resolve_file(v, search_dirs) if k == "scale_dict_path" and isinstance(v, str) else _resolve_scale_dicts(v, search_dirs)
        for k, v in node.items()
    }


def _search_dirs(paths: list[Path]) -> list[Path]:
    dirs = [Path.cwd()]
    for path in paths:
        dirs += [path.resolve().parent, *path.resolve().parent.parents]
    return list(dict.fromkeys(dirs))


def config_hash(section: Any) -> str:
    """sha256 of the canonical JSON of ``section`` (sorted keys)."""
    return hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode()).hexdigest()


# Native classes whose constructor defaults define "not set" for a config entry; the Encoder also forwards its extra
# kwargs to EncoderLayer. The torch defaults that jsonargparse writes into a saved config must equal these, or the
# two forms build different models and the hashes differ (as they should).
_DEFAULTS_FROM = {
    "MaskFormer": (MaskFormer,),
    "InputNet": (InputNet,),
    "FourierPositionEncoder": (FourierPositionEncoder,),
    "Dense": (Dense,),
    "Encoder": (Encoder, EncoderLayer),
    "MaskFormerDecoder": (MaskFormerDecoder,),
    **{name: (cls,) for name, cls in TASK_CLASSES.items()},
}
# entries given without a class_path (plain dicts, or the decoder after normalisation), by key
_CLASS_OF_KEY = {
    "encoder": "Encoder",
    "decoder": "MaskFormerDecoder",
    "posenc": "FourierPositionEncoder",
    "net": "Dense",
    "object_net": "Dense",
    "constituent_net": "Dense",
    "node_net": "Dense",
}
# hybrid (Keras-on-torch) class names of the same models
_ALIASES = {"KerasMaskFormer": "MaskFormer", "KerasEncoder": "Encoder"}
_NOT_ARGS = {"self", "factory", "name", "layer_name"}
# not part of the model: host matching options (S4) and the multi-task flag (from_config raises for mtl: true)
_NOT_MODEL = {"matcher"}


# YAML merge as LightningCLI (jsonargparse 4.40.1, measured; docs/hgq/port_log/S6a.md decision 13). A class entry is
# a dict with class_path / init_args, or a plain dict at a class-typed parameter (the torch MaskFormer's raw decoder).
_CLASS_ENTRY_KEYS = {"class_path", "init_args", "dict_kwargs"}
# plain-dict parameters that are dict-typed (replaced whole) although their key is in _CLASS_OF_KEY, by owning class
# (the torch signature's type): hybrid KerasMaskFormer encoder / decoder: dict; torch MaskFormerDecoder posenc:
# dict[str, float] | None (alpha / base, src/hepattn/models/decoder.py:32), unlike InputNet's FourierPositionEncoder
_DICT_TYPED = {"KerasMaskFormer": {"encoder", "decoder"}, "MaskFormerDecoder": {"posenc"}}


def _class_of(key: str | None, parent: str | None) -> str | None:
    """The class of a plain dict given at ``parent``'s parameter ``key`` (None: not class-typed)."""
    return None if key in _DICT_TYPED.get(parent or "", ()) else _CLASS_OF_KEY.get(key or "")


def _check_entries(node: Any) -> None:
    """Raise for a ``class_path`` dict with keys other than ``init_args`` / ``dict_kwargs``, anywhere in ``node``.

    jsonargparse 4.40.1 rejects this form ("Key 'class_path' is not expected") in one file and in an overlay, at the
    top-level class, a nested one and a callbacks item (LightningCLI probes, s6_fixtures/lightningcli_probes.yaml).

    Raises:
        ConfigError: For such an entry.
    """
    if isinstance(node, list):
        for item in node:
            _check_entries(item)
    elif isinstance(node, dict):
        stray = sorted(k for k in node if k not in _CLASS_ENTRY_KEYS)
        if "class_path" in node and stray:
            raise ConfigError(
                f"class entry {node['class_path']!r} has {stray} next to class_path; LightningCLI rejects this: give them under init_args"
            )
        for value in node.values():
            _check_entries(value)


def _is_class_entry(node: Any, key: str | None, parent: str | None) -> bool:
    if not isinstance(node, dict):
        return False
    return bool(_CLASS_ENTRY_KEYS & set(node)) or _class_of(key, parent) is not None


def _split_entry(node: dict) -> tuple[str | None, dict, Any]:
    if not _CLASS_ENTRY_KEYS & set(node):
        return None, dict(node), None
    if set(node) - _CLASS_ENTRY_KEYS:  # left after _check_entries only without a class_path
        raise ConfigError(f"class entry {node} has init_args and also other keys next to them")
    return node.get("class_path"), dict(node.get("init_args") or {}), node.get("dict_kwargs")


def _check_list_overlay(base: Any, overlay: Any, key: str) -> None:
    """Raise when ``overlay`` replaces a list of class entries and gives an item without a ``class_path``.

    LightningCLI merges such an item into the base item at its index for some parameter types (measured for
    ``torch.nn.ModuleList.modules``: class_path and the init args the overlay leaves out are kept) and rejects it for
    others (``list[Base]``); the loader replaces lists whole, so it would build another model without an error.

    Raises:
        ConfigError: For such an overlay.
    """
    if not (isinstance(base, list) and isinstance(overlay, list)):
        return
    if any(isinstance(item, dict) and "class_path" in item for item in base):
        bare = [i for i, item in enumerate(overlay) if isinstance(item, dict) and "class_path" not in item]
        if bare:
            raise ConfigError(
                f"{key}: overlay list items {bare} have no class_path; LightningCLI merges them into the earlier items, "
                "which the loader does not reproduce: give each item's class_path and all its init_args"
            )


def _merge_value(base: Any, overlay: Any, key: str, parent: str | None) -> Any:
    _check_list_overlay(base, overlay, key)
    both_dicts = isinstance(base, dict) and isinstance(overlay, dict)
    if not both_dicts or not (_is_class_entry(base, key, parent) or _is_class_entry(overlay, key, parent)):
        return copy.deepcopy(overlay)  # scalars, lists, None and dict-typed arguments are replaced whole
    base_path, base_args, base_kwargs = _split_entry(base)
    path, args, kwargs = _split_entry(overlay)
    if base_path and path and _short(base_path) != _short(path):
        raise ConfigError(f"{key}: an overlay that changes class_path {base_path!r} -> {path!r} is not supported")
    path = path or base_path
    merged = _merge_args(base_args, args, _short(path) if path else _class_of(key, parent))
    if path is None:
        return merged
    out = {"class_path": path, "init_args": merged}
    if (kwargs if kwargs is not None else base_kwargs) is not None:
        out["dict_kwargs"] = copy.deepcopy(kwargs if kwargs is not None else base_kwargs)
    return out


def _merge_args(base: dict, overlay: dict, parent: str | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        out[key] = _merge_value(base[key], value, key, parent) if key in base else copy.deepcopy(value)
    return out


def merge_configs(base: dict, overlay: dict) -> dict:
    """``overlay`` over ``base`` as LightningCLI merges two ``--config`` files. Returns a copy.

    Top-level sections (``model``, ``data``, ``trainer``) are argument groups and merge key by key. Inside them, a
    class entry merges its init args key by key; an overlay may give them as a plain dict without ``class_path`` /
    ``init_args`` (eval.yaml's ``model: model: encoder: attn_type: torch``) once an earlier file gave the class. Every
    other value is replaced whole: scalars, lists, None, and dict-typed arguments such as ``lrs_config``,
    ``attn_kwargs``, ``decoder_layer_config``, the torch decoder's ``posenc`` and the hybrid ``KerasMaskFormer``'s
    ``encoder`` / ``decoder``.

    ``ConfigError`` (raised by the helpers) where LightningCLI rejects the form or does something the loader does not
    reproduce: keys next to a ``class_path`` (jsonargparse 4.40.1 rejects them; give them under ``init_args``),
    ``init_args`` next to other keys, an overlay that changes a ``class_path`` (jsonargparse keeps only the init args
    the new class accepts, which needs the class), and an overlay list of class entries with an item that has no
    ``class_path`` (LightningCLI merges it into the earlier item for ``ModuleList``).
    """
    _check_entries(base)
    _check_entries(overlay)
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        old = base.get(key)
        if isinstance(old, dict) and isinstance(value, dict) and not (_is_class_entry(old, key, None) or _is_class_entry(value, key, None)):
            out[key] = _merge_args(old, value, None)  # an argument group
        else:
            out[key] = _merge_value(old, value, key, None) if key in base else copy.deepcopy(value)
    return out


def _defaults(class_name: str) -> dict:
    out: dict = {}
    for cls in _DEFAULTS_FROM.get(class_name, ()):
        for param in inspect.signature(cls.__init__).parameters.values():
            if param.name not in _NOT_ARGS and param.default is not inspect.Parameter.empty:
                out.setdefault(param.name, param.default)
    return out


def _file_digest(path: str) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(node: Any, key: str | None = None, parent: str | None = None) -> Any:
    if isinstance(node, list):
        return [_canonical(item) for item in node]
    if key in ACTIVATION_KEYS and isinstance(node, str):
        return _short(node)
    if key == "scale_dict_path" and isinstance(node, str):
        return _file_digest(node)
    if isinstance(node, int | float) and not isinstance(node, bool):
        return float(node)  # jsonargparse writes 2.0 for a float-typed 2
    if not isinstance(node, dict):
        return node
    if "class_path" in node:
        class_name = _short(node["class_path"])
        class_name = _ALIASES.get(class_name, class_name)
        if class_name == "ModuleList":
            return _canonical(list((node.get("init_args") or {}).get("modules") or []))
        args = dict(node.get("init_args") or {})
    elif _class_of(key, parent) is not None:
        class_name, args = _class_of(key, parent), dict(node)
    else:
        return {k: _canonical(v, k) for k, v in node.items()}
    defaults = _defaults(class_name)
    args = {k: _canonical(v, k, class_name) for k, v in args.items() if k not in _NOT_MODEL}
    return {"class": class_name, "args": {k: v for k, v in args.items() if k not in defaults or v != _canonical(defaults[k], k)}}


def model_description(model_section: dict) -> dict:
    """What defines the model in a (normalised, scale-dict-resolved) ``model:`` section, independent of its form.

    The ``model.model`` entry with: class paths reduced to class names; entries equal to the native constructor
    default removed (a saved config spells the defaults out, a raw one omits them); ``scale_dict_path`` replaced by
    the sha256 of the file's content (so the path, host and checkout do not matter, the content does); the encoder's
    repeated ``dim`` and the matcher dropped. ``lrs_config``, ``optimizer`` and ``mtl`` are not part of it.
    """
    entry = model_section["model"]
    entry = {"class_path": "MaskFormer", "init_args": entry} if "class_path" not in entry else entry
    desc = _canonical(entry)
    args = desc["args"]
    encoder = args.get("encoder")
    if isinstance(encoder, dict) and encoder["args"].get("dim") == args.get("dim"):
        encoder["args"].pop("dim")
    return desc


@dataclass(frozen=True)
class OptimizerConfig:
    """``model.optimizer`` + ``model.lrs_config`` + the trainer's clipping and accumulation (``models/wrapper.py``).

    ``epsilon`` is the torch ``AdamW`` default (1e-8), which the hybrid run uses; keras ``AdamW`` defaults to 1e-7,
    so the native optimizer builder must pass it. Lion has no epsilon (lion_pytorch betas (0.9, 0.99) are the keras
    Lion defaults).
    """

    name: str
    lr_initial: float
    lr_max: float
    lr_end: float
    pct_start: float
    weight_decay: float
    skip_scheduler: bool
    clip_norm: float | None
    accumulate_grad_batches: int
    epsilon: float = 1e-8
    quantizer_lr_scale: float = 1.0  # native only (optim.py)


@dataclass(frozen=True)
class QuantConfig:
    """The HGQ2 spec (``model.model.init_args.quant``) and the beta / EBOPs callbacks of ``trainer.callbacks``.

    ``beta_pid`` / ``beta_scheduler`` hold every constructor argument of the hybrid callback (defaults filled in).
    """

    spec: dict | None
    beta_pid: dict | None
    beta_scheduler: dict | None
    ebops_every_n_steps: int


@dataclass(frozen=True)
class CheckpointCallback:
    """What one Lightning ``ModelCheckpoint`` does (lightning 2.5.2 ``callbacks/model_checkpoint.py``), not its YAML text.

    ``every_n_epochs`` / ``every_n_train_steps``: 0 is "no trigger"; with neither set, Lightning uses
    ``every_n_epochs=1`` (``__init_triggers``), i.e. a save at every train epoch end, after that epoch's validation
    (with ``val_check_interval`` 1.0, as in every CLIC config, Lightning saves in ``on_train_epoch_end``, not in
    ``on_validation_end``: ``model_checkpoint.py:324-339,429-447``; an epoch stopped by ``max_steps`` runs no
    validation and is saved with the last logged monitor value). ``save_top_k`` -1 keeps every checkpoint, 0 none;
    with ``monitor`` None, 1 keeps only the newest. ``save_last``: ``last`` is rewritten at every save and at every
    epoch end. ``dirpath`` None is Lightning's default directory. ``checkpoint.CheckpointManager.from_callback``
    builds the manager that does this.
    """

    monitor: str | None
    mode: str
    save_top_k: int
    save_last: bool
    every_n_train_steps: int
    every_n_epochs: int
    dirpath: str | None


@dataclass(frozen=True)
class CheckpointConfig:
    """The two checkpoint callbacks of the CLIC configs; None if the config does not register it.

    - ``epoch``: ``hepattn.callbacks.Checkpoint`` (``src/hepattn/callbacks/checkpoint.py``): ``save_top_k=-1``, so
      every epoch-end checkpoint is kept (``epoch=NNN-val_loss=X``); ``setup`` forces ``save_last = False`` and, for
      fit, ``dirpath = <log_dir>/ckpts`` whatever the YAML says (``save_last: true`` in base.yaml / pflow_hgq.yaml has
      no effect). ``dirpath`` is therefore None here.
    - ``step``: ``lightning.pytorch.callbacks.ModelCheckpoint`` (Polaris, ``polaris/03_make_config.py``): every 1000
      steps, ``save_top_k=1`` without a monitor (the newest step checkpoint), ``save_last``.
    """

    epoch: CheckpointCallback | None
    step: CheckpointCallback | None


@dataclass(frozen=True)
class DataConfig:
    train_path: str | None
    valid_path: str | None
    test_path: str | None
    batch_size: int
    num_workers: int
    num_train: int
    num_val: int
    num_test: int
    dataset_kwargs: dict  # CLICDataset kwargs other than filepath / num_events


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int | None
    raw: dict  # the merged config, model section normalised, scale dicts resolved
    sources: tuple[str, ...]
    model_section: dict  # input of maskformer.from_config (keeps the ``mtl`` check)
    task_specs: list[TaskSpec]
    matcher_n_jobs: int
    optimizer: OptimizerConfig
    quant: QuantConfig
    checkpoint: CheckpointConfig
    data: DataConfig
    trainer: dict  # the remaining trainer keys (max_epochs, precision, log_every_n_steps, devices, ...)
    callbacks: dict[str, dict] = field(default_factory=dict)  # short class name -> init_args

    @property
    def model_hash(self) -> str:
        """Hash of ``model_description``: equal for every form of one model config (raw, generated, saved, moved)."""
        return config_hash(model_description(self.model_section))

    def build_model(self) -> MaskFormer:
        """``maskformer.from_config`` of the model section (unbuilt; call it on one batch to build)."""
        return from_config(self.model_section)


def _callbacks(trainer: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for entry in trainer.get("callbacks") or []:
        name = _short(entry["class_path"]) if isinstance(entry, dict) else _short(str(entry))
        if name in out:
            raise ConfigError(f"callback {name} is registered twice")
        out[name] = dict(entry.get("init_args") or {}) if isinstance(entry, dict) else {}
    return out


def _optimizer(model: dict, trainer: dict) -> OptimizerConfig:
    given = model.get("optimizer", "AdamW")
    name = {n.lower(): n for n in OPTIMIZERS}.get(str(given).lower())  # the wrapper matches case-insensitively
    if name is None:
        raise ConfigError(f"optimizer {given!r} is not supported (supported: {sorted(OPTIMIZERS)})")
    if trainer.get("gradient_clip_algorithm") not in {None, "norm"}:
        raise ConfigError(f"gradient_clip_algorithm {trainer['gradient_clip_algorithm']!r}: only norm clipping is supported")
    lrs = model.get("lrs_config") or {}
    if "weight_decay" not in lrs:
        raise ConfigError("lrs_config has no weight_decay (the torch wrapper reads lrs_config['weight_decay'] without a default)")
    clip = trainer.get("gradient_clip_val")
    return OptimizerConfig(
        name=name,
        lr_initial=float(lrs["initial"]),
        lr_max=float(lrs["max"]),
        lr_end=float(lrs["end"]),
        pct_start=float(lrs["pct_start"]),
        weight_decay=float(lrs["weight_decay"]),
        skip_scheduler=bool(lrs.get("skip_scheduler", False)),
        clip_norm=None if clip is None else float(clip),
        accumulate_grad_batches=int(trainer.get("accumulate_grad_batches") or 1),
        quantizer_lr_scale=float(lrs.get("quantizer_lr_scale", 1.0)),
    )


# The hybrid callbacks' constructor defaults and types (hepattn/keras/callbacks.py; pinned by test_config.py, which
# reads them from that file's source). A raw config omits defaults, a saved one spells them out: both give the same dict.
BETA_PID_ARGS = {
    "target_ebops": (float, None),
    "init_beta": (float, None),
    "p": (float, 1.0),
    "i": (float, 0.5),
    "d": (float, 0.0),
    "warmup_epochs": (int, 2),
    "max_beta": (float, 1e-7),
    "min_beta": (float, 0.0),
    "damp_beta_on_target": (float, 0.0),
}
BETA_SCHEDULER_ARGS = {"beta_start": (float, 0.0), "beta_end": (float, 1e-5), "warmup_steps": (int, 1000)}


def _with_defaults(name: str, args: dict | None, table: dict) -> dict | None:
    """``args`` completed with the defaults of ``table`` and coerced to its types (YAML may give "1.0e15" as a string).

    Raises:
        ConfigError: For an unknown or a missing required argument.
    """
    if args is None:
        return None
    unknown = set(args) - set(table)
    missing = {k for k, (_, default) in table.items() if default is None and k not in args}
    if unknown or missing:
        raise ConfigError(f"{name}: unknown arguments {sorted(unknown)}, missing {sorted(missing)}")
    return {k: kind(args.get(k, default)) for k, (kind, default) in table.items()}


def _quant(init_args: dict, callbacks: dict[str, dict]) -> QuantConfig:
    present = BETA_CALLBACKS & set(callbacks)
    if len(present) > 1:
        raise ConfigError("BetaPID and BetaScheduler both write beta; register only one (as hepattn.keras.callbacks.BetaPID.setup)")
    every = int((callbacks.get("PeriodicEBOPs") or {}).get("every_n_steps", 1))
    if every < 1:
        raise ConfigError(f"PeriodicEBOPs every_n_steps must be >= 1, got {every}")
    return QuantConfig(
        spec=init_args.get("quant"),
        beta_pid=_with_defaults("BetaPID", callbacks.get("BetaPID"), BETA_PID_ARGS),
        beta_scheduler=_with_defaults("BetaScheduler", callbacks.get("BetaScheduler"), BETA_SCHEDULER_ARGS),
        ebops_every_n_steps=every,
    )


def _checkpoint_callback(name: str, args: dict, *, hepattn_checkpoint: bool) -> CheckpointCallback:
    """The effective settings of one ``ModelCheckpoint`` (see ``CheckpointCallback``).

    Raises:
        ConfigError: For a time-interval trigger, a save on train epoch end, or a combination Lightning rejects.
    """
    if args.get("train_time_interval") is not None or args.get("save_on_train_epoch_end") is not None:
        raise ConfigError(f"{name}: train_time_interval / save_on_train_epoch_end are not supported by the native driver")
    steps, epochs = args.get("every_n_train_steps"), args.get("every_n_epochs")
    if steps is None and epochs is None:
        steps, epochs = 0, 1
    steps, epochs = int(steps or 0), int(epochs or 0)
    if steps and epochs:
        raise ConfigError(f"{name}: every_n_train_steps and every_n_epochs are mutually exclusive (as in Lightning)")
    monitor = args.get("monitor", "val/loss") if hepattn_checkpoint else args.get("monitor")
    save_top_k = -1 if hepattn_checkpoint else int(args.get("save_top_k", 1))
    if monitor is None and save_top_k not in {-1, 0, 1}:
        raise ConfigError(f"{name}: save_top_k={save_top_k} needs a monitor (as in Lightning)")
    return CheckpointCallback(
        monitor=monitor,
        mode=args.get("mode", "min"),
        save_top_k=save_top_k,
        save_last=False if hepattn_checkpoint else bool(args.get("save_last")),  # "link" saves too
        every_n_train_steps=steps,
        every_n_epochs=epochs,
        dirpath=None if hepattn_checkpoint else args.get("dirpath"),
    )


def _checkpoint(callbacks: dict[str, dict]) -> CheckpointConfig:
    def build(name: str, *, hepattn_checkpoint: bool) -> CheckpointCallback | None:
        if name not in callbacks:
            return None
        return _checkpoint_callback(name, callbacks[name], hepattn_checkpoint=hepattn_checkpoint)

    return CheckpointConfig(epoch=build("Checkpoint", hepattn_checkpoint=True), step=build("ModelCheckpoint", hepattn_checkpoint=False))


DATA_LOADER_KEYS = {"train_path", "valid_path", "test_path", "batch_size", "num_workers", "num_train", "num_val", "num_test", "scale_dict_path"}
DATA_IGNORED_KEYS = {"pin_memory", "test_suff"}  # torch-loader / file-name options with no native meaning


def _data(data: dict) -> DataConfig:
    kwargs = {k: v for k, v in data.items() if k not in DATA_LOADER_KEYS | DATA_IGNORED_KEYS}
    kwargs["scale_dict_path"] = data["scale_dict_path"]
    return DataConfig(
        train_path=data.get("train_path"),
        valid_path=data.get("valid_path"),
        test_path=data.get("test_path"),
        batch_size=int(data["batch_size"]),
        num_workers=int(data.get("num_workers", 0)),
        num_train=int(data.get("num_train", -1)),
        num_val=int(data.get("num_val", -1)),
        num_test=int(data.get("num_test", -1)),
        dataset_kwargs=kwargs,
    )


TRAINER_KEYS_READ = {"callbacks", "gradient_clip_val", "gradient_clip_algorithm", "accumulate_grad_batches"}


def load_config(paths: str | Path | list[str | Path], *, search_dirs: list[str | Path] | None = None) -> RunConfig:
    """Read one config file or several (merged in order) into a ``RunConfig``.

    Args:
        paths: The YAML file(s): raw CLIC configs, the generated Polaris config, or a saved run ``config.yaml``.
        search_dirs: Extra directories (searched first) for relative ``scale_dict_path`` values.

    Raises:
        ConfigError: For an option the native driver does not run (optimizer, clipping, both beta callbacks, ...).
    """
    sources = [Path(p) for p in ([paths] if isinstance(paths, str | Path) else paths)]
    cfg = read_yaml(sources)
    dirs = [Path(d) for d in (search_dirs or [])] + _search_dirs(sources)
    if "model" not in cfg or "model" not in cfg["model"]:
        raise ConfigError("the config has no model.model section")
    cfg["model"] = _resolve_scale_dicts(normalise_model_section(cfg["model"]), dirs)
    if "data" in cfg:
        cfg["data"] = _resolve_scale_dicts(cfg["data"], dirs)
    trainer = cfg.get("trainer") or {}
    callbacks = _callbacks(trainer)
    entry = cfg["model"]["model"]
    init_args = entry.get("init_args", entry) if isinstance(entry, dict) else {}
    tasks = init_args["tasks"]
    task_entries = tasks["init_args"]["modules"] if isinstance(tasks, dict) else list(tasks)
    return RunConfig(
        name=str(cfg.get("name", "")),
        seed=cfg.get("seed_everything"),
        raw=cfg,
        sources=tuple(str(s) for s in sources),
        model_section=cfg["model"],
        task_specs=specs_from_config(task_entries),
        matcher_n_jobs=matcher_n_jobs_from_config(init_args["matcher"]) if "matcher" in init_args else 1,
        optimizer=_optimizer(cfg["model"], trainer),
        quant=_quant(init_args, callbacks),
        checkpoint=_checkpoint(callbacks),
        data=_data(cfg["data"]),
        trainer={k: v for k, v in trainer.items() if k not in TRAINER_KEYS_READ},
        callbacks=callbacks,
    )
