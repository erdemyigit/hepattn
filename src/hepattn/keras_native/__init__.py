"""Pure ``keras.ops`` + HGQ2 implementation of the CLIC MaskFormer, trained on the JAX backend.

Rules (enforced by ``tests/keras_native/test_import_hygiene.py``): no module in this package imports
torch, and this ``__init__`` imports no submodule. The backend is chosen by ``KERAS_BACKEND`` before
keras is first imported; this package does not set it, because ``factory`` is also re-exported to the
torch-backend hybrid (``hepattn.keras.factory``).
"""
