import math
import re
from pathlib import Path

import pytest
import yaml

from hepattn.experiments.clic import main

from ..utils import run_test  # noqa: TID252

CONFIG = Path("tests/experiments/clic/test_clic.yaml")
NUM_SECTORS, NODES_PER_SECTOR = 3, 53


def sector_config(loss_norm: str, out: Path) -> Path:
    """test_clic.yaml with the sectored decoder: 3 sectors of at most 53 nodes, 50 shared queries per sector."""
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["name"] = f"clic_test_sector_{loss_norm}"
    cfg["data"]["sectors"] = {
        "num_sectors": NUM_SECTORS,
        "scheme": "cap",
        "max_nodes_per_sector": NODES_PER_SECTOR,
        "track_phi": "calo",
        "overflow": "drop_lowest_energy",
    }
    model = cfg["model"]["model"]
    model["class_path"] = "hepattn.models.sector.SectoredMaskFormer"
    model["init_args"].update(num_sectors=NUM_SECTORS, max_nodes_per_sector=NODES_PER_SECTOR, loss_norm=loss_norm)
    model["init_args"]["decoder"]["num_queries"] = cfg["data"]["num_objects"] // NUM_SECTORS
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def test_clic() -> None:
    run_test(main, str(CONFIG))


def assert_finite_val_loss(run_dir: Path) -> None:
    """The checkpoint names carry the validation loss (callbacks/checkpoint.py), which must be finite."""
    val_losses = [float(re.search(r"val_loss=(.+)\.ckpt$", c.name).group(1)) for c in (run_dir / "ckpts").glob("epoch=*.ckpt")]
    assert val_losses
    assert all(math.isfinite(v) for v in val_losses), val_losses


@pytest.mark.parametrize("loss_norm", ["sector_mean", "event"])
def test_clic_sector(loss_norm, tmp_path) -> None:
    run_test(main, str(sector_config(loss_norm, tmp_path / "config.yaml")), after_fit=assert_finite_val_loss)
