import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from hepattn.experiments.clic import main_hgq

from ..utils import run_test  # noqa: TID252


@pytest.fixture(autouse=True)
def _eager_torch_compile():
    """The torch.compile'd loss registries are exercised in eager mode here.

    Inductor's C++ compilation fails on paths containing spaces on some platforms;
    compile behaviour itself is covered by the torch-side integration test.
    """
    torch.compiler.set_stance("force_eager")
    yield
    torch.compiler.set_stance("default")


def test_clic_hgq() -> None:
    """Full fit + test cycle of the HGQ2 QAT model on dummy data.

    Exercises: lazy quantized-layer materialization before optimizer creation,
    EBOPs term in the training loss, beta scheduling, checkpoint save (fit) and
    restore (test subcommand reloads the best checkpoint via the CLI).
    """
    run_test(main_hgq, "tests/experiments/clic/test_clic_hgq.yaml")
