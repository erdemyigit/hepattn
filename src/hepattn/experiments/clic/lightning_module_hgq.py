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
        # Create keras variables on the RANK'S device, not cpu.
        #
        # The old comment here said variables are "created on cpu and moved with the module
        # by Lightning". Measured (polaris/10_ddp_device_probe.py): they are NOT. Module.to()
        # moves all 2027 registered parameters and buffers, and leaves the keras Variables
        # behind on cpu. Single-device runs survive that because keras reads them wherever
        # they are; DDP does not, because torch's _sync_module_states walks a wider set than
        # named_parameters()+named_buffers() and hands NCCL 684 cpu tensors, which fails with
        # "No backend type associated with device type cpu" (measured, run 7598422).
        #
        # self.device is still cpu at setup() -- Lightning has not moved the module yet -- so
        # take the device from the strategy, which setup_environment() has already resolved.
        set_keras_default_device(self._target_device())
        self._materialize_keras_layers(stage)

    def _target_device(self) -> str:
        strategy = getattr(self.trainer, "strategy", None)
        root = getattr(strategy, "root_device", None)
        return str(root) if root is not None else str(self.device)

    def _sync_keras_device(self) -> None:
        # setup() now builds on the strategy's root device, so this is normally a no-op.
        # Kept because self.device is authoritative once Lightning has moved the module, and
        # because test/predict can run without a fit having gone through setup() first.
        set_keras_default_device(str(self.device))

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._sync_keras_device()

    def on_validation_start(self) -> None:
        self._sync_keras_device()

    def on_test_start(self) -> None:
        self._sync_keras_device()

    def on_predict_start(self) -> None:
        self._sync_keras_device()

    def _materialize_keras_layers(self, stage: str) -> None:
        datamodule = self.trainer.datamodule
        loader_fn = {
            "fit": datamodule.train_dataloader,
            "validate": datamodule.val_dataloader,
        }.get(stage, datamodule.test_dataloader)
        inputs, _ = next(iter(loader_fn()))
        # the loader yields cpu tensors; the layers are being built on the target device
        device = self._target_device()
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        self.model.to(device)
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

        # NOT self.model.named_parameters(): keras layer weights are not registered on
        # the nn.Module, so that list omits every Dense kernel in the model. See
        # KerasMaskFormer.trainable_parameter_groups.
        decay_params, quantizer_params = self.model.trainable_parameter_groups()

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
