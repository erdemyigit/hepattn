"""Training script for the Keras/HGQ2 CLIC pflow model.

A separate entrypoint from main.py because the LightningCLI pins the module class
(MPflowHGQ here) and because importing the keras stack must not affect torch-only runs.

    python -m hepattn.experiments.clic.main_hgq fit --config .../pflow_hgq.yaml
"""

import pathlib

import comet_ml  # noqa: F401
from lightning.pytorch.cli import ArgsType

from hepattn.experiments.clic.lightning_module_hgq import MPflowHGQ
from hepattn.experiments.clic.pflow_data import PflowDataModule
from hepattn.utils.cli import CLI

config_dir = pathlib.Path(__file__).parent / "configs"


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=MPflowHGQ,
        datamodule_class=PflowDataModule,
        args=args,
        parser_kwargs={"default_env": True, "fit": {"default_config_files": [f"{config_dir}/pflow_hgq.yaml"]}},
    )


if __name__ == "__main__":
    main()
