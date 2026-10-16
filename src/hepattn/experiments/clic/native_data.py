"""CLIC data for the keras_native (JAX) driver: ``CLICDataset`` + torch ``DataLoader`` on the CPU as the host source.

This module imports torch (through ``pflow_data``); it lives outside ``hepattn.keras_native`` on purpose, because
that package must not import torch (``tests/keras_native/test_import_hygiene.py``). The torch-free half, conversion
to static-shape numpy / device batches and padding, is ``hepattn.keras_native.data``. Decision (plan section 3.8,
user decision 2026-10-16): torch DataLoader as the host data source for now; a numpy ``load_event`` is S9.

Order and resume: the train order of an epoch is a permutation drawn from ``torch.Generator().manual_seed(seed +
epoch)``, so ``(epoch, batch index)`` is the data position a checkpoint stores, and ``start_batch`` resumes inside an
epoch with the same order (``EpochSampler``). Val / test are not shuffled.

Workers: with ``num_workers > 0`` the loader starts its workers with ``spawn`` (``WORKER_START_METHOD``), never
``fork``, and keeps them for the whole run (``persistent_workers``; ``SplitBatches`` holds one loader per split and
moves the sampler to the next epoch with ``set_epoch``). The process is multithreaded once JAX has a backend, and
JAX warns at every ``os.fork()`` that this "will likely lead to a deadlock" (jax/_src/xla_bridge.py, ``_at_fork``);
``fork`` is the Python 3.12 default on Linux (Polaris), ``spawn`` on macOS. A spawned worker receives the dataset
pickled, once per run; its start cost with real CLIC sizes is to be measured on Polaris (S6a log).
"""

import multiprocessing
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Sampler

from hepattn.experiments.clic.pflow_data import CLICDataset
from hepattn.keras_native.config import DataConfig
from hepattn.keras_native.data import HostBatches

WORKER_START_METHOD = "spawn"
SPLITS = {"train": ("train_path", "num_train"), "val": ("valid_path", "num_val"), "test": ("test_path", "num_test")}


class EpochSampler(Sampler[int]):
    """Indices of one epoch, from batch ``start_batch`` on; shuffled with the seed ``seed + epoch`` if ``shuffle``."""

    def __init__(
        self, num_events: int, *, batch_size: int, shuffle: bool, seed: int, epoch: int = 0, start_batch: int = 0, drop_last: bool = False
    ) -> None:
        self.num_events = num_events
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.start_batch = start_batch
        self.drop_last = drop_last

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        """The next ``iter`` gives epoch ``epoch`` from batch ``start_batch`` (read in the main process, also with workers)."""
        self.epoch = epoch
        self.start_batch = start_batch

    def order(self) -> list[int]:
        if not self.shuffle:
            return list(range(self.num_events))
        return torch.randperm(self.num_events, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()

    def __iter__(self) -> Iterator[int]:
        order = self.order()
        if self.drop_last:
            order = order[: len(order) - len(order) % self.batch_size]
        return iter(order[self.start_batch * self.batch_size :])

    def __len__(self) -> int:
        full = self.num_events - (self.num_events % self.batch_size if self.drop_last else 0)
        return max(0, full - self.start_batch * self.batch_size)


def make_dataset(data: DataConfig, split: str, *, dummy_data: bool | None = None, num_events: int | None = None) -> CLICDataset:
    """``CLICDataset`` of ``split`` (train / val / test) from the data config."""
    path_key, num_key = SPLITS[split]
    kwargs = dict(data.dataset_kwargs)
    if dummy_data is not None:
        kwargs["dummy_data"] = dummy_data
    return CLICDataset(filepath=getattr(data, path_key), num_events=num_events if num_events is not None else getattr(data, num_key), **kwargs)


def make_loader(
    dataset: CLICDataset,
    split: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    epoch: int = 0,
    start_batch: int = 0,
    multiprocessing_context: str = WORKER_START_METHOD,
) -> DataLoader:
    """A torch ``DataLoader``: train shuffles and drops the last short batch, val / test keep the order and every event.

    With workers: persistent, started with ``multiprocessing_context`` (see the module docstring).

    Raises:
        ValueError: If ``multiprocessing_context`` is ``fork``.
    """
    train = split == "train"
    sampler = EpochSampler(len(dataset), batch_size=batch_size, shuffle=train, seed=seed, epoch=epoch, start_batch=start_batch, drop_last=train)
    workers = {}
    if num_workers > 0:
        if multiprocessing.get_context(multiprocessing_context).get_start_method() == "fork":
            raise ValueError("DataLoader workers must not be forked from the multithreaded JAX process (use spawn or forkserver)")
        workers = {"multiprocessing_context": multiprocessing_context, "persistent_workers": True}
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=train, num_workers=num_workers, **workers)


class SplitBatches:
    """One loader (and its workers) for a split for the whole run; ``epoch`` gives the static-shape batches of an epoch.

    Train drops the last short batch, val / test are padded with invalid events. One ``HostBatches`` serves every
    epoch, so a batch whose shapes or dtypes differ from the first batch of the run raises.
    """

    def __init__(
        self,
        data: DataConfig,
        split: str,
        *,
        dataset: CLICDataset | None = None,
        seed: int = 42,
        device_put: bool = True,
        multiprocessing_context: str = WORKER_START_METHOD,
    ) -> None:
        self.dataset = dataset if dataset is not None else make_dataset(data, split)
        self.loader = make_loader(
            self.dataset, split, batch_size=data.batch_size, num_workers=data.num_workers, seed=seed, multiprocessing_context=multiprocessing_context
        )
        self.batches = HostBatches(self.loader, data.batch_size, pad_to_batch=split != "train", device_put=device_put)

    def epoch(self, epoch: int = 0, start_batch: int = 0) -> HostBatches:
        self.loader.sampler.set_epoch(epoch, start_batch)
        return self.batches


def host_batches(
    data: DataConfig,
    split: str,
    *,
    dataset: CLICDataset | None = None,
    seed: int = 42,
    epoch: int = 0,
    start_batch: int = 0,
    device_put: bool = True,
) -> HostBatches:
    """Static-shape batches of one epoch of ``split`` (a new loader; a run keeps one ``SplitBatches`` per split)."""
    return SplitBatches(data, split, dataset=dataset, seed=seed, device_put=device_put).epoch(epoch, start_batch)
