"""Re-export shim. The factory is backend-neutral and lives in ``hepattn.keras_native.factory``.

Importing it through ``hepattn.keras`` first runs ``hepattn/keras/__init__.py``, which pins the torch
backend before keras initializes, so hybrid callers keep their old import path and behaviour.

The float leaves ``EinsumOp`` and ``SoftmaxOp`` are registered as keras-serializable here unless something already
registered them. Lightning's ``save_hyperparameters`` deep-copies the model, and keras deep-copies a layer through
its config, so without the registration a float KerasMaskFormer cannot be built by the CLI
(``TypeError: Could not locate class 'EinsumOp'``, measured on pflow_keras_float.yaml; docs/hgq/port_log/R4a.md).
"""

from hepattn.keras import keras
from hepattn.keras_native.factory import EinsumOp, LayerFactory, QuantSpec, SoftmaxOp, apply_softmax

for _cls in (EinsumOp, SoftmaxOp):
    if keras.saving.get_registered_name(_cls) == _cls.__name__:  # not registered yet
        keras.saving.register_keras_serializable(package="hepattn")(_cls)

__all__ = ["EinsumOp", "LayerFactory", "QuantSpec", "SoftmaxOp", "apply_softmax"]
