"""Checkpoints of the stateless training state ``(tv, ntv, ov)`` (plan section 3.9).

Format: one ``.npz`` file holds every array (``tv/<key>``, ``ntv/<key>``, ``ov/<key>``) and the metadata as JSON
(``__meta__``: step, epoch, beta-controller state, data position, config hash, free-form extras). A JSON sidecar
(same stem, ``.json``) repeats the metadata and the per-array shape / dtype, plus the sha256 of the ``.npz``. Both
files are written to a temporary name in the target directory, flushed, fsynced and renamed (``os.replace``), and the
directory is fsynced after each rename (so the rename survives a power loss); the sidecar is renamed last, so it marks
a complete checkpoint. ``save_checkpoint`` does not replace an existing checkpoint unless asked (``overwrite``):
between the two renames the new ``.npz`` would sit next to the old sidecar, a checkpoint that is rejected on load.
``CheckpointManager`` therefore never overwrites; a second save at one step gets a new name. ``load_checkpoint``
rejects a checkpoint whose sidecar is missing, whose ``.npz`` hash differs, or whose keys, shapes or dtypes differ
from the freshly built model's.

Keys. ``variable.path`` is not stable: HGQ2 quantizer sublayers are auto-named, so the same config built twice in one
process gives ``.../fixed_point_quantizer_kif/f`` and ``.../fixed_point_quantizer_kif_580/f`` (S3 open issue,
measured in docs/hgq/port_log/S6a.md). The key of a model variable is its path with a trailing ``_<digits>`` removed
from every segment, plus ``#<n>`` when that leaves several variables with the same name (``n`` counts them in
``model.variables`` order, which is the construction order and so the same for every build of one config). An
optimizer slot variable (keras names it ``<reference path with '/' -> '_'>_<slot>``) is keyed by the key of its
reference variable and the slot name; other optimizer variables (``iteration``, ...) by their own name.

Export: ``export_weights`` assigns a state into the variables and calls ``Model.save_weights`` (``.weights.h5``,
keyed by the layer tracking structure, not by paths); a freshly built model loads it with ``load_weights``.

Policy: the two Lightning callbacks of the CLIC configs (``config.CheckpointConfig``). ``CheckpointManager`` writes
- ``epoch-<epoch>-step-<step>.npz`` at each train epoch end, after that epoch's validation (``maybe_save_epoch``;
  hepattn ``Checkpoint``, ``save_top_k=-1``: never deleted), with ``best`` pointing at the one with the best finite
  ``val/loss``. Lightning saves in ``on_train_epoch_end`` when ``val_check_interval`` is 1.0 (every CLIC config), so
  an epoch cut short by ``max_steps`` is saved there too, with the last logged ``val/loss`` (S6a.md decision 10);
- ``step-<step>.npz`` every ``every_n_steps`` (``maybe_save_step``; Polaris ``ModelCheckpoint``, the newest
  ``keep_last_k`` = ``save_top_k``; -1 keeps all), and ``last-<step>.npz`` at an epoch end or the train end
  (``save_last``; Lightning rewrites ``last.ckpt`` there when ``save_last``), with ``last`` pointing at the newest of
  the two. ``CheckpointManager.from_callback`` maps a ``config.CheckpointCallback`` onto these;
- ``best-step-<step>.npz`` (``maybe_save_best``): only the best one is kept (``save_top_k=1`` with a monitor; no CLIC
  config uses it).
The pointer files ``last.json`` and ``best.json`` are each replaced atomically after the checkpoint they name is
complete. Files get the mode of a plain ``open(..., "w")`` (0o666 minus the umask), so a shared project area can read
them (``mkstemp`` alone gives 0o600).
"""

import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np

FORMAT_VERSION = 1
GROUPS = ("tv", "ntv", "ov")
META_KEY = "__meta__"
_AUTO_SUFFIX = re.compile(r"_\d+$")
_STEP_FILE = re.compile(r"step-(?P<step>\d+)(?:-r(?P<rev>\d+))?\.npz")
_LAST_FILE = re.compile(r"last-\d+(?:-r\d+)?\.npz")
_BEST_FILE = re.compile(r"best-step-\d+(?:-r\d+)?\.npz")


class CheckpointError(RuntimeError):
    """A checkpoint that is incomplete, corrupted, or does not fit the model."""


def _canonical(path: str) -> str:
    return "/".join(_AUTO_SUFFIX.sub("", segment) for segment in path.split("/"))


def _numbered(names: list[str]) -> list[str]:
    total = Counter(names)
    seen: Counter = Counter()
    keys = []
    for name in names:
        keys.append(name if total[name] == 1 else f"{name}#{seen[name]}")
        seen[name] += 1
    return keys


def variable_keys(model: keras.Model) -> dict[int, str]:
    """``id(variable) -> key`` for every variable of ``model`` (see the module docstring)."""
    variables = model.variables
    return dict(zip((id(v) for v in variables), _numbered([_canonical(v.path) for v in variables]), strict=True))


def optimizer_keys(optimizer: keras.optimizers.Optimizer, model: keras.Model) -> list[str]:
    """Keys of ``optimizer.variables``, in order: ``<reference key>::<slot>`` for slot variables, else the name.

    Raises:
        CheckpointError: If two optimizer variables get the same key.
    """
    by_id = variable_keys(model)
    refs = {v.path.replace("/", "_"): by_id[id(v)] for v in model.trainable_variables}
    by_length = sorted(refs, key=len, reverse=True)
    keys = []
    for var in optimizer.variables:
        name = var.name
        ref = next((r for r in by_length if name.startswith(r + "_")), None)
        keys.append(f"{refs[ref]}::{name[len(ref) + 1 :]}" if ref is not None else name)
    if len(set(keys)) != len(keys):
        dup = sorted(k for k, n in Counter(keys).items() if n > 1)
        raise CheckpointError(f"optimizer variables with the same key: {dup[:5]}")
    return keys


def state_keys(model: keras.Model, optimizer: keras.optimizers.Optimizer | None = None) -> dict[str, list[str]]:
    """Keys of ``(tv, ntv, ov)`` in the order of ``model.trainable_variables``, ``model.non_trainable_variables``, ``optimizer.variables``."""
    by_id = variable_keys(model)
    return {
        "tv": [by_id[id(v)] for v in model.trainable_variables],
        "ntv": [by_id[id(v)] for v in model.non_trainable_variables],
        "ov": optimizer_keys(optimizer, model) if optimizer is not None else [],
    }


def current_state(model: keras.Model, optimizer: keras.optimizers.Optimizer | None = None) -> tuple[list, list, list]:
    """Host copies of the variable values ``(tv, ntv, ov)``."""

    def host(variables):
        return [np.array(v.value) for v in variables]

    return host(model.trainable_variables), host(model.non_trainable_variables), host(optimizer.variables) if optimizer is not None else []


def assign_state(model: keras.Model, optimizer: keras.optimizers.Optimizer | None, state: Sequence) -> None:
    """Write ``(tv, ntv, ov)`` into the variables (e.g. before ``save_weights`` or an eager evaluation).

    Raises:
        CheckpointError: If a group has another number of arrays than the variables it is written to.
    """
    tv, ntv, ov = state
    pairs = [(model.trainable_variables, tv), (model.non_trainable_variables, ntv)]
    if optimizer is not None:
        pairs.append((optimizer.variables, ov))
    for variables, values in pairs:
        if len(variables) != len(values):
            raise CheckpointError(f"state has {len(values)} arrays for {len(variables)} variables")
        for var, value in zip(variables, values, strict=True):
            var.assign(value)


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _atomic(path: Path, mode: str = "wb"):
    """Open a temporary file next to ``path``; on success fsync it and rename it over ``path``.

    Yields:
        The open temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).chmod(0o666 & ~_umask())
        Path(tmp).replace(path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            Path(tmp).unlink()
        raise
    _fsync_directory(path.parent)


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _fsync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable (POSIX); a no-op where directories cannot be opened (Windows)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sidecar_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".json")


def save_checkpoint(
    path: str | Path,
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer | None,
    state: Sequence,
    meta: dict | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``state = (tv, ntv, ov)`` and ``meta`` to ``path`` (``.npz``) and its JSON sidecar. Returns ``path``.

    ``meta`` must be JSON-serialisable (numpy scalars are converted); conventional keys: ``step``, ``epoch``,
    ``beta_controller``, ``data_position``, ``config_hash``.

    Raises:
        ValueError: If ``path`` does not end in ``.npz``.
        FileExistsError: If ``path`` or its sidecar exists and ``overwrite`` is False.
        CheckpointError: If a state group does not match the model's / optimizer's variable count.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"checkpoint path must end in .npz, got {path}")
    if not overwrite and (path.exists() or sidecar_path(path).exists()):
        raise FileExistsError(f"{path} exists (a crash between the two renames of an overwrite leaves it unloadable)")
    keys = state_keys(model, optimizer)
    arrays: dict[str, np.ndarray] = {}
    layout: dict[str, dict[str, list]] = {}
    for group, values in zip(GROUPS, state, strict=True):
        if len(values) != len(keys[group]):
            raise CheckpointError(f"{group}: {len(values)} arrays for {len(keys[group])} variables")
        layout[group] = {}
        for key, value in zip(keys[group], values, strict=True):
            array = np.asarray(value)
            arrays[f"{group}/{key}"] = array
            layout[group][key] = [list(array.shape), array.dtype.str]
    meta_json = _dumps({"format_version": FORMAT_VERSION, **(meta or {})})
    arrays[META_KEY] = np.array(meta_json)
    with _atomic(path) as f:
        np.savez(f, **arrays)
    sidecar = {"format_version": FORMAT_VERSION, "npz": path.name, "sha256": _sha256(path), "layout": layout, "meta": json.loads(meta_json)}
    with _atomic(sidecar_path(path), "w") as f:
        f.write(json.dumps(sidecar, indent=1, sort_keys=True))
    return path


def read_meta(path: str | Path) -> dict:
    """The metadata from the sidecar (no array is read; the sidecar is not verified against the ``.npz``)."""
    return json.loads(sidecar_path(path).read_text())["meta"]


def load_checkpoint(
    path: str | Path,
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer | None = None,
    *,
    expect_config_hash: str | None = None,
) -> tuple[tuple[list, list, list], dict]:
    """Read a checkpoint for a freshly built ``model`` (and built ``optimizer``). Returns ``((tv, ntv, ov), meta)``.

    The arrays come back in the order of this model's variable lists, matched by key. Nothing is assigned; use
    ``assign_state`` for that.

    Raises:
        CheckpointError: Missing sidecar, hash mismatch, unreadable ``.npz``, key / shape / dtype mismatch, metadata
            mismatch, or a config hash other than ``expect_config_hash``.
    """
    path = Path(path)
    side = sidecar_path(path)
    if not path.is_file() or not side.is_file():
        raise CheckpointError(f"incomplete checkpoint: {path} exists={path.is_file()}, sidecar {side} exists={side.is_file()}")
    try:
        sidecar = json.loads(side.read_text())
    except json.JSONDecodeError as err:
        raise CheckpointError(f"{side}: corrupted sidecar ({err})") from err
    if sidecar.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"{side}: format version {sidecar.get('format_version')}, expected {FORMAT_VERSION}")
    digest = _sha256(path)
    if digest != sidecar.get("sha256"):
        raise CheckpointError(f"{path}: sha256 {digest} differs from the sidecar's {sidecar.get('sha256')} (corrupted or partial write)")
    keys = state_keys(model, optimizer)
    arrays, meta = _read_npz(path)
    groups = GROUPS if optimizer is not None else GROUPS[:2]
    for group in groups:
        stored = {name.split("/", 1)[1] for name in arrays if name.startswith(group + "/")}
        missing, extra = set(keys[group]) - stored, stored - set(keys[group])
        if missing or extra:
            raise CheckpointError(f"{group}: keys differ from the model's; missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
    out = []
    for group in GROUPS:
        values = []
        for var, key in zip(_variables(model, optimizer, group), keys[group], strict=True):
            array = arrays[f"{group}/{key}"]
            if tuple(array.shape) != tuple(var.shape) or np.dtype(array.dtype) != np.dtype(var.dtype):
                raise CheckpointError(f"{group}/{key}: stored {array.shape} {array.dtype}, model {tuple(var.shape)} {var.dtype}")
            if [list(array.shape), array.dtype.str] != sidecar["layout"][group][key]:
                raise CheckpointError(f"{group}/{key}: array differs from the sidecar layout")
            values.append(array)
        out.append(values)
    if meta != sidecar["meta"]:
        raise CheckpointError(f"{path}: metadata differs from the sidecar's")
    if expect_config_hash is not None and meta.get("config_hash") != expect_config_hash:
        raise CheckpointError(f"{path}: written for config {meta.get('config_hash')}, expected {expect_config_hash}")
    return (out[0], out[1], out[2]), meta


def _read_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Every array and the metadata of a checkpoint ``.npz``.

    Raises:
        CheckpointError: If the file cannot be read (zip or CRC errors of a damaged file).
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files if name != META_KEY}
            meta = json.loads(str(data[META_KEY]))
    except Exception as err:  # zipfile / numpy / json errors on a damaged file
        raise CheckpointError(f"{path}: cannot read the arrays ({type(err).__name__}: {err})") from err
    return arrays, meta


def _variables(model, optimizer, group: str) -> list:
    if group == "tv":
        return model.trainable_variables
    if group == "ntv":
        return model.non_trainable_variables
    return optimizer.variables if optimizer is not None else []


def export_weights(model: keras.Model, state: Sequence, path: str | Path) -> Path:
    """Assign ``state`` (the model part) into ``model`` and save it as ``.weights.h5``.

    Raises:
        ValueError: If ``path`` does not end in ``.weights.h5``.
    """
    path = Path(path)
    if not path.name.endswith(".weights.h5"):
        raise ValueError(f"export path must end in .weights.h5, got {path}")
    assign_state(model, None, (state[0], state[1], []))
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(path)
    return path


@dataclass
class CheckpointPolicy:
    """When to save: every ``every_n_steps`` optimizer steps, every ``every_n_epochs`` validated epochs, and on a new
    best ``monitor`` value (``mode`` min/max).
    """

    every_n_steps: int | None = 1000
    monitor: str | None = "val/loss"
    mode: str = "min"
    best_value: float | None = None
    every_n_epochs: int | None = 1

    def __post_init__(self):
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.every_n_steps is not None and self.every_n_steps < 1:
            raise ValueError(f"every_n_steps must be >= 1, got {self.every_n_steps}")
        if self.every_n_epochs is not None and self.every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {self.every_n_epochs}")

    def step_due(self, step: int) -> bool:
        """True at steps ``every_n_steps``, ``2 * every_n_steps``, ... (Lightning ``every_n_train_steps``)."""
        return self.every_n_steps is not None and step > 0 and step % self.every_n_steps == 0

    def epoch_due(self, epoch: int) -> bool:
        """True after epochs ``every_n_epochs - 1``, ``2 * every_n_epochs - 1``, ... (0-based, as Lightning's
        ``(current_epoch + 1) % every_n_epochs == 0``, ``model_checkpoint.py:328``).
        """
        return self.every_n_epochs is not None and (epoch + 1) % self.every_n_epochs == 0

    def is_better(self, metrics: dict) -> bool:
        """True if ``metrics[monitor]`` is finite and improves on the best so far (NaN never improves)."""
        if self.monitor is None or self.monitor not in metrics:
            return False
        value = float(metrics[self.monitor])
        if not math.isfinite(value):
            return False
        if self.best_value is None:
            return True
        return value < self.best_value if self.mode == "min" else value > self.best_value


class CheckpointManager:
    """Writes checkpoints under ``directory`` by a ``CheckpointPolicy``.

    ``keep_last_k`` is Lightning's ``save_top_k`` without a monitor (``model_checkpoint.py:92-93``): the newest ``k``
    step checkpoints are kept; -1 keeps every one; 0 writes none (with ``write_last``, ``maybe_save_step`` writes
    ``last-<step>.npz`` instead, as Lightning writes ``last.ckpt`` with ``save_top_k=0, save_last=True``).
    ``write_last`` records the callback's ``save_last``: the driver calls ``save_last`` at each train epoch end
    (after that epoch's validation) and at the train end when it is set. ``from_callback`` builds the manager of one ``config.CheckpointCallback``.

    Raises:
        ValueError: For ``keep_last_k`` other than an int >= -1.
    """

    def __init__(self, directory: str | Path, policy: CheckpointPolicy | None = None, keep_last_k: int = 1, *, write_last: bool = False) -> None:
        if isinstance(keep_last_k, bool) or not isinstance(keep_last_k, int) or keep_last_k < -1:
            raise ValueError(f"keep_last_k must be an int >= -1 (-1 keeps every step checkpoint), got {keep_last_k!r}")
        self.directory = Path(directory)
        self.policy = policy or CheckpointPolicy()
        self.keep_last_k = keep_last_k
        self.write_last = write_last
        best = self._pointer("best")
        if best is not None:
            self.policy.best_value = best.get("value")

    @classmethod
    def from_callback(cls, directory: str | Path, callback) -> "CheckpointManager":
        """The manager that does what one ``config.CheckpointCallback`` does.

        Step trigger (``every_n_train_steps``): ``maybe_save_step``, ``keep_last_k = save_top_k``. Epoch trigger:
        ``maybe_save_epoch`` at each train epoch end; every epoch checkpoint is kept (hepattn ``Checkpoint``,
        ``save_top_k=-1``).

        Raises:
            ValueError: For what the manager does not implement: top-k by a monitor (``save_top_k`` > 0 with a
                monitor on a step trigger, or other than -1 on an epoch trigger).
        """
        keep = callback.save_top_k
        if callback.every_n_train_steps:
            if callback.monitor is not None and keep > 0:
                raise ValueError(f"step checkpoints: save_top_k={keep} by monitor {callback.monitor!r} is not implemented")
            policy = CheckpointPolicy(every_n_steps=callback.every_n_train_steps, monitor=callback.monitor, mode=callback.mode, every_n_epochs=None)
        else:
            if keep != -1:
                raise ValueError(f"epoch checkpoints: only save_top_k=-1 (hepattn Checkpoint) is implemented, got {keep}")
            policy = CheckpointPolicy(every_n_steps=None, monitor=callback.monitor, mode=callback.mode, every_n_epochs=callback.every_n_epochs)
        return cls(directory, policy, keep_last_k=keep, write_last=callback.save_last)

    def _pointer(self, name: str) -> dict | None:
        path = self.directory / f"{name}.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def _set_pointer(self, name: str, target: Path, **extra) -> None:
        with _atomic(self.directory / f"{name}.json", "w") as f:
            f.write(json.dumps({"file": target.name, **extra}, sort_keys=True, default=_json_default))

    def _remove(self, path: Path) -> None:
        for p in (sidecar_path(path), path):  # sidecar first: a half-removed checkpoint reads as incomplete
            with contextlib.suppress(FileNotFoundError):
                p.unlink()

    def _new_path(self, prefix: str, step: int) -> Path:
        """``<prefix>-<step>.npz``, or ``<prefix>-<step>-r<n>.npz`` (next free ``n``) if a checkpoint of that step exists."""
        path = self.directory / f"{prefix}-{step:09d}.npz"
        revision = 0
        while path.exists() or sidecar_path(path).exists():
            revision += 1
            path = self.directory / f"{prefix}-{step:09d}-r{revision}.npz"
        return path

    def _step_files(self) -> list[Path]:
        """Step checkpoints, oldest first: by step, then revision."""
        found = []
        for path in self.directory.glob("step-*.npz"):
            match = _STEP_FILE.fullmatch(path.name)
            if match:
                found.append(((int(match["step"]), int(match["rev"] or 0)), path))
        return [path for _, path in sorted(found)]

    def _point_last(self, path: Path, step: int) -> None:
        """Point ``last`` at ``path``, then remove the ``last-*`` checkpoints it no longer names."""
        self._set_pointer("last", path, step=step)
        for old in self.directory.glob("last-*.npz"):
            if _LAST_FILE.fullmatch(old.name) and old != path:
                self._remove(old)

    def save(self, step: int, model, optimizer, state, meta: dict) -> Path | None:
        """Write ``step-<step>.npz`` (a new name if it exists), point ``last`` at it, keep the newest ``keep_last_k``.

        With ``keep_last_k`` 0: no step file; ``save_last`` if ``write_last``, else nothing (returns None).
        """
        if self.keep_last_k == 0:
            return self.save_last(step, model, optimizer, state, meta) if self.write_last else None
        path = save_checkpoint(self._new_path("step", step), model, optimizer, state, {**meta, "step": step})
        self._point_last(path, step)
        if self.keep_last_k > 0:
            protected = {p["file"] for p in (self._pointer("best"),) if p}
            steps = self._step_files()
            for old in steps[: max(0, len(steps) - self.keep_last_k)]:
                if old.name not in protected and old != path:
                    self._remove(old)
        return path

    def maybe_save_step(self, step: int, model, optimizer, state, meta: dict) -> Path | None:
        return self.save(step, model, optimizer, state, meta) if self.policy.step_due(step) else None

    def _monitored(self, metrics: dict) -> None:
        """Check that the monitor is in ``metrics``; Lightning raises once validation has run (``model_checkpoint.py:378-386``).

        Raises:
            KeyError: If the monitor is not in ``metrics``.
        """
        if self.policy.monitor is not None and self.policy.monitor not in metrics:
            raise KeyError(f"monitor {self.policy.monitor!r} is not in the metrics {sorted(metrics)}")

    def save_last(self, step: int, model, optimizer, state, meta: dict) -> Path:
        """``last`` at a train epoch end or the train end: a no-op if ``last`` already holds ``step``.

        Lightning writes ``last.ckpt`` there when ``save_last`` is set (and skips a step it already saved); call this
        when ``CheckpointCallback.save_last`` is true, else ``last`` is up to ``every_n_steps - 1`` steps old.
        """
        pointer = self._pointer("last")
        if pointer is not None and pointer.get("step") == step and (self.directory / pointer["file"]).is_file():
            return self.directory / pointer["file"]
        path = save_checkpoint(self._new_path("last", step), model, optimizer, state, {**meta, "step": step})
        self._point_last(path, step)
        return path

    def save_epoch(self, epoch: int, step: int, metrics: dict, model, optimizer, state, meta: dict) -> Path:
        """Write ``epoch-<epoch>-step-<step>.npz`` (kept for good); point ``best`` at it if ``metrics[monitor]`` improves.

        As hepattn ``Checkpoint`` (``save_top_k=-1``): every epoch is saved, a NaN monitor value included. A NaN is
        never the best; Lightning ranks it as +inf (min mode), so its ``best_model_path`` names a NaN epoch until a
        finite one comes (S6a.md decision 10). A monitor missing from ``metrics`` raises
        ``KeyError`` (``_monitored``).
        """
        self._monitored(metrics)
        value = metrics.get(self.policy.monitor) if self.policy.monitor else None
        if value is not None:  # NaN as a string: a NaN read back from JSON never equals the sidecar's
            value = float(value) if math.isfinite(float(value)) else str(float(value))
        extra = {"monitor": self.policy.monitor, "value": value}
        path = save_checkpoint(
            self._new_path(f"epoch-{epoch:03d}-step", step), model, optimizer, state, {**meta, "epoch": epoch, "step": step, **extra}
        )
        if self.policy.is_better(metrics):
            self._set_pointer("best", path, step=step, epoch=epoch, monitor=self.policy.monitor, value=value)
            self.policy.best_value = value
        return path

    def maybe_save_epoch(self, epoch: int, step: int, metrics: dict, model, optimizer, state, meta: dict) -> Path | None:
        """``save_epoch`` after the epochs ``policy.epoch_due`` names; None otherwise."""
        return self.save_epoch(epoch, step, metrics, model, optimizer, state, meta) if self.policy.epoch_due(epoch) else None

    def maybe_save_best(self, step: int, metrics: dict, model, optimizer, state, meta: dict) -> Path | None:
        """Write ``best-step-<step>.npz`` if ``metrics[monitor]`` improves; remove the previous ``best-step`` file.

        A monitor missing from ``metrics`` raises ``KeyError`` (``_monitored``).
        """
        self._monitored(metrics)
        if not self.policy.is_better(metrics):
            return None
        value = float(metrics[self.policy.monitor])
        previous = self._pointer("best")
        path = save_checkpoint(
            self._new_path("best-step", step),
            model,
            optimizer,
            state,
            {**meta, "step": step, "monitor": self.policy.monitor, "value": value},
        )
        self._set_pointer("best", path, step=step, monitor=self.policy.monitor, value=value)
        self.policy.best_value = value
        if previous is not None and previous["file"] != path.name and _BEST_FILE.fullmatch(previous["file"]):
            self._remove(self.directory / previous["file"])
        return path

    def last(self) -> Path | None:
        pointer = self._pointer("last")
        return self.directory / pointer["file"] if pointer else None

    def best(self) -> Path | None:
        pointer = self._pointer("best")
        return self.directory / pointer["file"] if pointer else None
