"""Host batches for the jitted step (plan section 3.8): collated loader batches -> numpy -> static shapes -> device.

Torch-free: the loader is any iterable of ``(inputs, targets)`` dicts of arrays; torch CPU tensors convert through
``np.asarray`` (the ``__array__`` protocol, as keras's own torch-DataLoader adapter does), so this module never
imports torch. The CLIC ``CLICDataset`` + torch ``DataLoader`` side is ``hepattn.experiments.clic.native_data``.

Per batch:
- every array becomes numpy; int64 -> int32 and float64 -> float32 (JAX runs with x64 off), with a range check, so an
  out-of-range integer raises instead of wrapping;
- ``host_keys`` (default ``event_number``) are kept on the host in their loader dtype and not sent to the device;
- ``pad_to``: a short batch is padded with invalid events to the fixed batch size (val / test; plan 3.8), and
  ``event_valid`` ([B] bool) marks the real events. An invalid event has every ``*_valid`` False, the null class in
  ``<target_object>_class``, NaN in the per-object float targets (as ``CLICDataset`` writes for invalid particles),
  and 0 elsewhere. Measured: the native forward and matched loss of such an event are finite (S6a log);
- every batch must have the shapes and dtypes of the first one (a new shape would compile a new executable).

NaN targets are passed through unchanged: the objective replaces them (``objective._nan_to_zero``).
Train loaders must drop the last short batch themselves (``DataLoader(drop_last=True)``); ``HostBatches`` raises
on a short batch when ``pad_to`` is None.
"""

import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import jax
import numpy as np

from hepattn.keras_native import profile

HOST_KEYS = ("event_number",)


@dataclass(frozen=True)
class Batch:
    inputs: dict  # device arrays
    targets: dict  # device arrays
    event_valid: np.ndarray  # [B] bool, host; False for padding events
    host: dict  # host_keys, numpy in the loader dtype (padding rows hold -1)

    @property
    def num_real(self) -> int:
        return int(self.event_valid.sum())


def _cast(key: str, array: np.ndarray) -> np.ndarray:
    if array.dtype in {np.dtype(np.int64), np.dtype(np.uint64)}:
        info = np.iinfo(np.int32)
        if array.size and (array.min() < info.min or array.max() > info.max):
            raise OverflowError(f"{key}: values outside int32 ({array.min()}..{array.max()})")
        return array.astype(np.int32)
    if array.dtype == np.float64:
        return array.astype(np.float32)
    return array


def to_numpy(tree: dict) -> dict[str, np.ndarray]:
    """``np.asarray`` of every value of a flat dict (torch CPU tensors included)."""
    return {key: np.asarray(value) for key, value in tree.items()}


def pad_value(key: str, array: np.ndarray, target_object: str, null_class: int):
    """The value of ``key`` in an invalid (padding) event (see the module docstring)."""
    if array.dtype == np.bool_:
        return False
    if key == f"{target_object}_class":
        return null_class
    if np.issubdtype(array.dtype, np.floating) and key.startswith(f"{target_object}_") and array.ndim == 2:
        return np.nan
    if key in HOST_KEYS:
        return -1
    return 0


def pad_events(tree: dict[str, np.ndarray], size: int, *, target_object: str = "particle", null_class: int = 5) -> dict[str, np.ndarray]:
    """Pad axis 0 of every array to ``size`` with invalid-event values.

    Raises:
        ValueError: If an array already has more than ``size`` events.
    """
    out = {}
    for key, array in tree.items():
        missing = size - array.shape[0]
        if missing < 0:
            raise ValueError(f"{key}: batch {array.shape[0]} is larger than {size}")
        fill = np.full((missing, *array.shape[1:]), pad_value(key, array, target_object, null_class), dtype=array.dtype)
        out[key] = np.concatenate([array, fill]) if missing else array
    return out


class HostBatches:
    """Iterate a loader of collated ``(inputs, targets)`` as static-shape ``Batch`` objects on the device.

    Args:
        loader: Iterable of ``(inputs, targets)`` dicts.
        batch_size: The fixed batch size every yielded batch has.
        pad_to_batch: Pad short batches with invalid events (val / test). If False, a short batch raises (train
            loaders must use ``drop_last``).
        host_keys: Keys kept on the host (looked up in inputs and targets).
        device_put: Put inputs and targets on the default device (False: numpy, for tests and host-side use).
        target_object, null_class: Fill values of padding events (``particle``, class 5 = null in CLIC).
    """

    def __init__(
        self,
        loader: Iterable,
        batch_size: int,
        *,
        pad_to_batch: bool,
        host_keys: tuple[str, ...] = HOST_KEYS,
        device_put: bool = True,
        target_object: str = "particle",
        null_class: int = 5,
    ) -> None:
        self.loader = loader
        self.batch_size = batch_size
        self.pad_to_batch = pad_to_batch
        self.host_keys = host_keys
        self.device_put = device_put
        self.target_object = target_object
        self.null_class = null_class
        self.signature: dict | None = None

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def convert(self, inputs: dict, targets: dict) -> Batch:
        prof = profile.current()
        with prof.phase("numpy"):
            inputs, targets = to_numpy(inputs), to_numpy(targets)
            size = next(iter(inputs.values())).shape[0]
            if size != self.batch_size and not self.pad_to_batch:
                raise ValueError(f"batch of {size} events, expected {self.batch_size}: use drop_last=True for training loaders")
            event_valid = np.arange(self.batch_size) < size
            if size != self.batch_size:
                pad = {"target_object": self.target_object, "null_class": self.null_class}
                inputs, targets = pad_events(inputs, self.batch_size, **pad), pad_events(targets, self.batch_size, **pad)
            host = {key: tree.pop(key) for tree in (inputs, targets) for key in list(tree) if key in self.host_keys}
            inputs = {key: _cast(key, value) for key, value in inputs.items()}
            targets = {key: _cast(key, value) for key, value in targets.items()}
            self._check_signature(inputs, targets)
        if self.device_put:
            with prof.phase("device_put"):
                inputs, targets = jax.device_put(inputs), jax.device_put(targets)
            prof.block((inputs, targets), "device_put_wait")
        return Batch(inputs=inputs, targets=targets, event_valid=event_valid, host=host)

    def _check_signature(self, inputs: dict, targets: dict) -> None:
        signature = {
            f"{group}/{key}": (value.shape, value.dtype.str) for group, tree in (("in", inputs), ("tgt", targets)) for key, value in tree.items()
        }
        if self.signature is None:
            self.signature = signature
        elif signature != self.signature:
            changed = sorted(k for k in signature.keys() | self.signature.keys() if signature.get(k) != self.signature.get(k))
            raise ValueError(f"batch shapes/dtypes changed (a new compile): {changed}")

    def __iter__(self) -> Iterator[Batch]:
        loader = iter(self.loader)
        while True:
            with profile.current().phase("loader"):
                item = next(loader, _END)
            if item is _END:
                return
            inputs, targets = item
            yield self.convert(inputs, targets)


_END = object()


@dataclass(frozen=True)
class _Raised:
    error: BaseException


def prefetch(iterable: Iterable, depth: int) -> Iterator:
    """Iterate ``iterable`` in a background thread, up to ``depth`` items ahead (same items, same order).

    For ``HostBatches``: the loader wait, the numpy conversion and the ``device_put`` of the next batches run while the
    main thread dispatches the step. Close the returned generator (or exhaust it) before iterating ``iterable``
    again: closing stops the thread after the item it is producing and drops the items it read ahead. An exception
    in the thread is raised in the consumer at its position in the sequence.

    Yields:
        The items of ``iterable``, in order.

    Raises:
        ValueError: If ``depth`` < 1.
    """
    if depth < 1:
        raise ValueError(f"prefetch depth must be >= 1, got {depth}")
    items: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                items.put(item, timeout=0.05)
            except queue.Full:
                continue
            else:
                return True
        return False

    def work() -> None:
        try:
            for item in iterable:
                if not put(item):
                    return
        except BaseException as error:  # noqa: BLE001  (re-raised in the consumer)
            put(_Raised(error))
            return
        put(_END)

    thread = threading.Thread(target=work, name="hepattn-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is _END:
                return
            if isinstance(item, _Raised):
                raise item.error
            yield item
    finally:
        stop.set()
        while thread.is_alive():  # the thread may be blocked in put(): keep the queue empty until it sees ``stop``
            thread.join(timeout=0.05)
            while not items.empty():
                items.get_nowait()
