"""CLIC data for the keras_native (JAX) driver: ``CLICDataset`` + torch ``DataLoader`` on the CPU as the host source.

This module imports torch (through ``pflow_data``); it lives outside ``hepattn.keras_native`` on purpose, because
that package must not import torch (``tests/keras_native/test_import_hygiene.py``). The torch-free half, conversion
to static-shape numpy / device batches and padding, is ``hepattn.keras_native.data``. Decision (plan section 3.8,
user decision 2026-10-16): torch DataLoader as the host data source for now; a numpy ``load_event`` is S9.

Order and resume: the train order of an epoch is a permutation drawn from ``torch.Generator().manual_seed(seed +
epoch)``, so ``(epoch, batch index)`` is the data position a checkpoint stores, and ``start_batch`` resumes inside an
epoch with the same order (``EpochSampler``). Val / test are not shuffled.
"""

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Sampler

from hepattn.experiments.clic.pflow_data import CLICDataset
from hepattn.keras_native.config import DataConfig
from hepattn.keras_native.data import HostBatches

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
) -> DataLoader:
    """A torch ``DataLoader``: train shuffles and drops the last short batch, val / test keep the order and every event."""
    train = split == "train"
    sampler = EpochSampler(len(dataset), batch_size=batch_size, shuffle=train, seed=seed, epoch=epoch, start_batch=start_batch, drop_last=train)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=train, num_workers=num_workers, persistent_workers=False)


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
    """Static-shape batches of ``split``: train with ``drop_last``, val / test padded with invalid events."""
    dataset = dataset if dataset is not None else make_dataset(data, split)
    loader = make_loader(dataset, split, batch_size=data.batch_size, num_workers=data.num_workers, seed=seed, epoch=epoch, start_batch=start_batch)
    return HostBatches(loader, data.batch_size, pad_to_batch=split != "train", device_put=device_put)
