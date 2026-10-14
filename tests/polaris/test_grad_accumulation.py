"""Gradient accumulation in Lightning averages, so LR and clip must NOT be rescaled by it.

polaris/03_make_config.py used to divide the peak LR by HGQ_ACCUM and multiply the clip
threshold by it, assuming Lightning SUMS accumulated micro-batch gradients. Lightning 2.5
divides each micro-batch loss by accumulate_grad_batches (loops/optimization/automatic.py),
so that compensation ran every HGQ2 Polaris job at 1/4 of the intended LR with 4x looser
clipping. These tests pin both the Lightning behaviour and the generator.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

pl = pytest.importorskip("lightning.pytorch")

REPO = pathlib.Path(__file__).resolve().parents[2]
MAKE_CONFIG = REPO / "polaris/03_make_config.py"


class _Linear(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.lin = torch.nn.Linear(3, 1)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.lin(x), y)  # mean-reduced, like hepattn's losses

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1.0)  # update == -clipped gradient


def _update(accum: int, micro: int, clip: float | None = None) -> torch.Tensor:
    """Parameter change after ONE optimizer step over the same 8 events."""
    gen = torch.Generator().manual_seed(0)
    x, y = torch.randn(8, 3, generator=gen), torch.randn(8, 1, generator=gen)
    model = _Linear()
    before = torch.cat([p.detach().clone().flatten() for p in model.parameters()])
    trainer = pl.Trainer(
        max_steps=1,
        accumulate_grad_batches=accum,
        gradient_clip_val=clip,
        gradient_clip_algorithm="norm",
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, DataLoader(TensorDataset(x, y), batch_size=micro, shuffle=False))
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    return after - before


def test_lightning_averages_accumulated_gradients():
    full = _update(accum=1, micro=8)
    accumulated = _update(accum=4, micro=2)
    assert full.abs().max() > 1e-3  # the step moved the weights
    torch.testing.assert_close(accumulated, full)
    # the summing hypothesis behind the old compensation predicts 4x the full-batch update
    assert (accumulated - 4 * full).abs().max() > 1e-2


def test_clip_applies_to_the_averaged_gradient():
    clip = 1e-3
    unclipped = _update(accum=1, micro=8)
    assert unclipped.norm() > 10 * clip  # so the clip is active, not decorative
    full = _update(accum=1, micro=8, clip=clip)
    accumulated = _update(accum=4, micro=2, clip=clip)
    torch.testing.assert_close(accumulated, full)
    torch.testing.assert_close(accumulated.norm(), torch.tensor(clip), rtol=1e-4, atol=0.0)
    # the old 'clip x accum' would let the same step through at 4x the norm
    assert _update(accum=4, micro=2, clip=4 * clip).norm() > 3 * clip


def _generated(tmp_path: pathlib.Path, accum: int) -> dict:
    out = tmp_path / f"polaris_hgq_accum{accum}.yaml"
    env = {**os.environ, "HGQ_ACCUM": str(accum), "HGQ_CONFIG_OUT": str(out)}
    subprocess.run([sys.executable, str(MAKE_CONFIG)], check=True, env=env, capture_output=True, text=True)
    return yaml.safe_load(out.read_text())


def test_make_config_lr_and_clip_do_not_depend_on_accum(tmp_path):
    one, four = _generated(tmp_path, 1), _generated(tmp_path, 4)
    # the env override took effect, so the comparison below is between two different configs
    assert (one["trainer"]["accumulate_grad_batches"], four["trainer"]["accumulate_grad_batches"]) == (1, 4)
    for cfg in (one, four):
        assert cfg["model"]["lrs_config"]["max"] == pytest.approx(5.0e-5)
        assert cfg["trainer"]["gradient_clip_val"] == pytest.approx(0.1)
