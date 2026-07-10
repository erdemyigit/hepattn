"""Keras 3 (torch backend) + HGQ2 quantization-aware components for hepattn.

Keras locks its backend at first import, so every module in hepattn must obtain
keras through this package (``from hepattn.keras import keras``) rather than
importing it directly. This guarantees the torch backend is selected before
keras initializes, which is what makes keras/HGQ2 layers genuine
``torch.nn.Module`` instances that can live inside the existing Lightning
training driver.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

if os.environ["KERAS_BACKEND"] != "torch":
    raise RuntimeError(
        f"hepattn.keras requires KERAS_BACKEND='torch', found '{os.environ['KERAS_BACKEND']}'. "
        "The hepattn Keras/HGQ2 integration embeds keras layers in the PyTorch Lightning "
        "training driver, which is only possible under the torch backend. "
        "Unset KERAS_BACKEND or set it to 'torch'."
    )

import keras

if keras.backend.backend() != "torch":
    raise RuntimeError(
        f"keras was already initialized with backend '{keras.backend.backend()}' before "
        "hepattn.keras was imported. Import hepattn.keras before any direct keras import, "
        "or set KERAS_BACKEND=torch in the environment."
    )

from keras.src.backend.common import global_state
from keras.src.backend.torch.core import get_device


def set_keras_default_device(device: str) -> None:
    """Set the device on which keras-torch creates new tensors and variables (e.g. "cpu", "cuda:0").

    The keras torch backend auto-selects cuda/mps when available, independently of where
    the surrounding torch code runs. Any driver that mixes keras layers with plain torch
    tensors (tests, the Lightning wrapper) must pin this to the device it actually uses.
    """
    global_state.set_global_attribute("torch_device", device)


def get_keras_default_device() -> str:
    return get_device()


__all__ = ["get_keras_default_device", "keras", "set_keras_default_device"]
