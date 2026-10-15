"""Re-export shim. The factory is backend-neutral and lives in ``hepattn.keras_native.factory``.

Importing it through ``hepattn.keras`` first runs ``hepattn/keras/__init__.py``, which pins the torch
backend before keras initializes, so hybrid callers keep their old import path and behaviour.
"""

from hepattn.keras_native.factory import EinsumOp, LayerFactory, QuantSpec, SoftmaxOp, apply_softmax

__all__ = ["EinsumOp", "LayerFactory", "QuantSpec", "SoftmaxOp", "apply_softmax"]
