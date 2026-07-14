"""Lightning module for the Keras/HGQ2 CLIC pflow model.

Kept separate from lightning_module.py so torch-only runs never import keras
(importing keras pins its backend process-wide).
"""

import torch
from lion_pytorch import Lion
from torch import Tensor
from torch.optim import AdamW

from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.keras import set_keras_default_device
from hepattn.keras.maskformer import KerasMaskFormer


class MPflowHGQ(MPflow):
    """MPflow driving a KerasMaskFormer (float reference or HGQ2 quantization-aware).

    Adds on top of MPflow:
    - the HGQ2 EBOPs regularization term in the aggregated loss,
    - materialization of lazily-built quantized layers before the optimizer is
      created and before checkpoint state is restored (HGQ2 layers size their
      bitwidth variables from the first real batch's static shapes),
    - a quantizer parameter group without weight decay (decaying learned bitwidths
      would silently shrink precision), with the non-trainable beta excluded.
    """

    def setup(self, stage: str) -> None:
        super().setup(stage)
        assert isinstance(self.model, KerasMaskFormer), "MPflowHGQ requires a KerasMaskFormer model"
        # keras-torch auto-selects cuda/mps for NEW tensors independently of Lightning;
        # variables are created on cpu here and moved with the module by Lightning
        set_keras_default_device("cpu")
        self._materialize_keras_layers(stage)

    def _materialize_keras_layers(self, stage: str) -> None:
        datamodule = self.trainer.datamodule
        loader_fn = {
            "fit": datamodule.train_dataloader,
            "validate": datamodule.val_dataloader,
        }.get(stage, datamodule.test_dataloader)
        inputs, _ = next(iter(loader_fn()))
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)
        self.model.train(was_training)

    def aggregate_losses(self, losses: dict[str, dict[str, dict[str, Tensor]]], stage: str | None = None) -> Tensor:
        total_loss = super().aggregate_losses(losses, stage=stage)
        quant_loss = self.model.quant_losses()
        self.log(f"{stage}/quant_ebops_loss", quant_loss, sync_dist=True)
        return total_loss + quant_loss

    def configure_optimizers(self):
        # Mirrors ModelWrapper.configure_optimizers with quantizer-aware param groups.
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        decay_params, quantizer_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name.endswith("/beta"):
                # beta is the regularization strength read by HGQ2's add_loss — it must
                # never be optimized (its 'gradient' is just the EBOPs magnitude)
                continue
            if "quantizer" in name:
                quantizer_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [{"params": decay_params}, {"params": quantizer_params, "weight_decay": 0.0}]
        opt = optimizer(param_groups, lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            return [opt], [{"scheduler": sch, "interval": "step"}]

        return opt
