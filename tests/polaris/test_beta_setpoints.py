"""polaris/03_make_config.py beta defaults follow the re-derivation in docs/hgq/BETA_SETPOINTS.md (S5, PROVISIONAL).

The generated BetaPID / BetaScheduler arguments are pinned, and so are the rules that produced them: the 40%-cut
setpoint, the old 2.585 decades of headroom above and below the starting beta, the decade-per-ten-epochs integral gain,
a positive floor and ramp start (the JAX EBOPs read-out is penalty / beta), and the x EBOPS_EVERY scaling.
The measured inputs (MODEL_EBOPS, the gradient-ratio factor 0.333) are checked by tests/keras/test_quant_losses_dedupe.py
(EBOPs at init) and recorded in BETA_SETPOINTS.md (gradient ratios).
"""

import math
import os
import pathlib
import subprocess
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
MAKE_CONFIG = REPO / "polaris/03_make_config.py"
EVERY = 8


def generate(tmp_path, **env) -> dict:
    out = tmp_path / "polaris_hgq.yaml"
    subprocess.run(
        [sys.executable, str(MAKE_CONFIG)], env={**os.environ, "HGQ_CONFIG_OUT": str(out), **env}, check=True, capture_output=True, timeout=120
    )
    return yaml.safe_load(out.read_text())


def callback(cfg: dict, name: str) -> dict:
    found = [c for c in cfg["trainer"]["callbacks"] if c.get("class_path", "").endswith(name)]
    assert len(found) == 1, found
    return found[0]["init_args"]


def module_constants() -> dict:
    """Module-level float constants of the generator, without running it (it writes a file on import)."""
    import ast  # noqa: PLC0415

    values = {}
    for node in ast.parse(MAKE_CONFIG.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def test_pid_defaults_and_their_rules(tmp_path):
    pid = callback(generate(tmp_path), "BetaPID")
    target, init, ceiling, floor = pid["target_ebops"], pid["init_beta"] / EVERY, pid["max_beta"] / EVERY, pid["min_beta"] / EVERY
    assert (target, init, ceiling, floor, pid["i"]) == pytest.approx((1.5e11, 8.7e-11, 3.3e-8, 2.3e-13, 0.5), rel=1e-12)

    model_ebops = module_constants()["MODEL_EBOPS"]
    old_rule = 1.0e15 / 1.6755e15  # the pre-retarget setpoint over the pre-retarget EBOPs: a 40% cut
    assert target / model_ebops == pytest.approx(old_rule, abs=0.005)
    old_decades = math.log10(1.0e-7 / 2.6e-10)
    assert math.log10(ceiling / init) == pytest.approx(old_decades, abs=0.01)
    assert math.log10(init / floor) == pytest.approx(old_decades, abs=0.01)
    assert init == pytest.approx(2.6e-10 * 0.333, rel=0.01), "INIT_BETA no longer the gradient-ratio-preserving value"
    err = math.log10(model_ebops / target)
    epochs_per_decade = 1 / (pid["i"] * err)  # 8.9 here and for the old model (0.5 at an error of 0.224)
    assert 8 <= epochs_per_decade <= 12, f"one decade per ~ten epochs, got {epochs_per_decade:.1f}"
    assert floor > 0


def test_schedule_ramp_starts_above_zero(tmp_path):
    ramp = callback(generate(tmp_path, HGQ_BETA_MODE="schedule"), "BetaScheduler")
    assert ramp["beta_start"] == pytest.approx(1e-3 * 4.0e-15 * EVERY, rel=1e-12)
    assert 0 < ramp["beta_start"] < ramp["beta_end"]


def test_env_overrides_still_win(tmp_path):
    pid = callback(generate(tmp_path, HGQ_TARGET_EBOPS="2e11", HGQ_INIT_BETA="1e-10", HGQ_MIN_BETA="1e-14", HGQ_EBOPS_EVERY="1"), "BetaPID")
    assert (pid["target_ebops"], pid["init_beta"], pid["min_beta"]) == pytest.approx((2e11, 1e-10, 1e-14), rel=1e-12)
