"""Write tests/keras/golden/linformer_backend.npz: the gate H0 cases under the JAX backend.

Run in the native environment (it has jax; the hybrid environment does not):

    KERAS_BACKEND=jax PYTHONPATH=src .venv-jax/bin/python tests/keras/make_linformer_backend_golden.py

test_linformer_parity.py runs the same cases under the torch backend and compares.
"""

import os
import pathlib
import sys

if os.environ.get("KERAS_BACKEND") != "jax":
    sys.exit("run with KERAS_BACKEND=jax: the golden is the JAX-backend output")

import jax
import keras
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import linformer_backend_cases as C

OUT = pathlib.Path(__file__).parent / "golden" / "linformer_backend.npz"


def main() -> None:
    assert keras.backend.backend() == "jax"
    arrays = {
        C.key(spec_name, site, mode): C.run_case(site, spec, training)
        for spec_name, spec in C.SPECS.items()
        for site in C.SITES
        for mode, training in C.MODES.items()
    }
    meta = {"inputs_sha256": C.inputs_digest(), "jax": jax.__version__, "keras": keras.__version__}
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **arrays, **{f"meta.{name}": np.array(value) for name, value in meta.items()})
    print(f"wrote {OUT} ({len(arrays)} outputs, {OUT.stat().st_size} bytes), {meta}")


if __name__ == "__main__":
    main()
