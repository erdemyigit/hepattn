"""Shared setup for hepattn.keras tests.

Pins the keras-torch default device to CPU so keras-created tensors live on the
same device as the plain-torch reference tensors used in parity tests (keras
would otherwise auto-select mps/cuda). Import errors are tolerated here so that
environments without the `hgq` dependency group still collect (and skip) the
test modules via their own importorskip guards.
"""

import pytest

try:
    import hepattn.keras as _hepattn_keras
except ImportError:
    _hepattn_keras = None

if _hepattn_keras is not None:
    _hepattn_keras.set_keras_default_device("cpu")


@pytest.fixture(autouse=True)
def _eager_torch_compile():
    """Run torch.compile-wrapped functions (hepattn loss/cost registries) eagerly.

    Inductor's C++ compilation is broken on this platform for paths containing
    spaces; the compiled and eager results are numerically identical, and compile
    behaviour itself is covered by the torch-side tests.
    """
    if _hepattn_keras is None:
        yield
        return
    import torch  # noqa: PLC0415

    torch.compiler.set_stance("force_eager")
    yield
    torch.compiler.set_stance("default")
