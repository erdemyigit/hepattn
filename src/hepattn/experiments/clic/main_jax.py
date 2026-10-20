"""Entry point of the keras_native (JAX) CLIC driver: ``hepattn.keras_native.cli`` with the CLIC data source.

    KERAS_BACKEND=jax python -m hepattn.experiments.clic.main_jax fit --config polaris_hgq.yaml --run-dir runs/r1
    python -m hepattn.experiments.clic.main_jax test --config polaris_hgq.yaml --ckpt runs/r1/ckpts/<file>.npz

This module lives outside ``hepattn.keras_native`` because the data source (``CLICDataset`` + torch ``DataLoader`` on
the CPU, ``native_data``) imports torch. It sets ``KERAS_BACKEND=jax`` when unset (and refuses another backend)
before anything imports keras.

DataLoader workers are spawned (``native_data.WORKER_START_METHOD``) and re-import this module as ``__mp_main__``:
the module body only defines functions; all work, and every jax / keras import, is under ``main``.
"""

import os
import sys


class ClicSource:
    """One split of the CLIC data as a ``keras_native.loop.DataSource`` (one loader and its workers for the run)."""

    def __init__(self, rc, split: str, *, seed: int, dummy_data: bool | None = None) -> None:
        from hepattn.experiments.clic.native_data import SplitBatches, make_dataset  # noqa: PLC0415

        self.dataset = make_dataset(rc.data, split, dummy_data=dummy_data)
        self.split = SplitBatches(rc.data, split, dataset=self.dataset, seed=seed)
        self.num_events = len(self.dataset)
        size = rc.data.batch_size
        self.num_batches = self.num_events // size if split == "train" else -(-self.num_events // size)

    def epoch(self, epoch: int, start_batch: int = 0):
        return self.split.epoch(epoch, start_batch)


def clic_data(rc, split: str, *, seed: int, dummy_data: bool | None = None) -> ClicSource:
    return ClicSource(rc, split, seed=seed, dummy_data=dummy_data)


def main(argv: list[str] | None = None) -> int:
    """Run the driver.

    Raises:
        SystemExit: If ``KERAS_BACKEND`` names another backend.
    """
    backend = os.environ.setdefault("KERAS_BACKEND", "jax")
    if backend != "jax":
        raise SystemExit(f"main_jax needs KERAS_BACKEND=jax (or unset), found {backend!r}")
    from hepattn.keras_native import cli  # noqa: PLC0415

    return cli.main(argv, clic_data)


if __name__ == "__main__":
    sys.exit(main())
