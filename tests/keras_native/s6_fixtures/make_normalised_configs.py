"""Write the jsonargparse-normalised run configs that ``test_config.py`` loads (S6a).

A saved run ``config.yaml`` (``SaveConfig``) is what LightningCLI prints with ``--print_config``: defaults expanded,
activations and the decoder in ``class_path`` / ``init_args`` form. This script prints it for three configs and
writes them next to itself:

- ``org_normalised.yaml``: ``main.py fit --config base.yaml --config linformer.yaml`` (org torch model);
- ``polaris_normalised.yaml``: ``main_hgq.py fit --config <polaris/03_make_config.py output>`` (the hybrid run config);
- ``pflow_hgq_normalised.yaml``: ``main_hgq.py fit`` (its default config, ``pflow_hgq.yaml``);
- ``org_eval_normalised.yaml``: ``main.py fit --config base.yaml --config linformer.yaml --config eval.yaml``;
- ``org_partial_normalised.yaml``: the same with ``overlay_partial_dicts.yaml`` (next to this script) as the third file.

The last two pin how LightningCLI merges several ``--config`` files (``config.merge_configs``).

The generated Polaris config holds absolute paths of this checkout; they are replaced by ``/polaris/home/hepattn``
so the fixture is machine-independent and looks like a config saved on another host.

Run in the hybrid torch env (``hepattn/.venv``), from anywhere:
    KERAS_BACKEND=torch PYTHONPATH=<worktree>/src <hepattn>/.venv/bin/python tests/keras_native/s6_fixtures/make_normalised_configs.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLIC = REPO / "src/hepattn/experiments/clic"
FAKE_REPO = "/polaris/home/hepattn"


def print_config(module: str, configs: list[str]) -> str:
    args = [sys.executable, "-m", module, "fit"]
    for config in configs:
        args += ["--config", config]
    env = {**os.environ, "KERAS_BACKEND": "torch", "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run([*args, "--print_config"], cwd=CLIC, env=env, capture_output=True, text=True, check=True)
    return proc.stdout


def main() -> None:
    org = ["configs/base.yaml", "configs/linformer.yaml"]
    (HERE / "org_normalised.yaml").write_text(print_config("hepattn.experiments.clic.main", org))
    (HERE / "org_eval_normalised.yaml").write_text(print_config("hepattn.experiments.clic.main", [*org, "configs/eval.yaml"]))
    partial = print_config("hepattn.experiments.clic.main", [*org, str(HERE / "overlay_partial_dicts.yaml")])
    (HERE / "org_partial_normalised.yaml").write_text(partial)
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "polaris_hgq.yaml"
        env = {
            **os.environ,
            "HGQ_CONFIG_OUT": str(generated),
            "DATA_DIR": "/eagle/hgcal-maskformer-fpga/clic_data",
            "CKPT_DIR": f"{FAKE_REPO}/hgq_ckpts",
        }
        subprocess.run([sys.executable, str(REPO / "polaris/03_make_config.py")], env=env, check=True, capture_output=True)
        text = print_config("hepattn.experiments.clic.main_hgq", [str(generated)])
    (HERE / "polaris_normalised.yaml").write_text(text.replace(str(REPO), FAKE_REPO))
    (HERE / "pflow_hgq_normalised.yaml").write_text(print_config("hepattn.experiments.clic.main_hgq", []))
    names = ("org_normalised.yaml", "org_eval_normalised.yaml", "org_partial_normalised.yaml", "polaris_normalised.yaml", "pflow_hgq_normalised.yaml")
    for name in names:
        assert str(REPO) not in (HERE / name).read_text(), name
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
