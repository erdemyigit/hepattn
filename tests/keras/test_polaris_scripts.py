"""The Polaris shell scripts pick the clone from HEPATTN_BRANCH / HEPATTN_DIR (polaris/repo_dir.sh).

Jobs that are running use $WORK_ROOT/hepattn on keras-hgq2 (docs/hgq/RETARGET_ORG_MAIN.md section 2, item 6), so a
new branch must get its own clone, every job script must run in the clone the setup made, and the setup must never
switch the branch of an existing clone.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
POLARIS = REPO / "polaris"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def repo_dir(**env: str) -> tuple[str, str]:
    script = f'WORK_ROOT=/w; source "{POLARIS}/repo_dir.sh"; echo "$HEPATTN_BRANCH $HEPATTN_DIR"'
    clean = {k: v for k, v in os.environ.items() if k not in {"HEPATTN_BRANCH", "HEPATTN_DIR"}}
    out = subprocess.run([BASH, "-c", script], env={**clean, **env}, check=True, capture_output=True, text=True).stdout.split()
    return out[0], out[1]


def test_repo_dir_defaults_and_overrides():
    assert repo_dir() == ("keras-hgq2", "/w/hepattn")  # unchanged for the running keras-hgq2 jobs
    assert repo_dir(HEPATTN_BRANCH="keras-jax") == ("keras-jax", "/w/hepattn-keras-jax")
    assert repo_dir(HEPATTN_BRANCH="rt/integration") == ("rt/integration", "/w/hepattn-rt-integration")
    assert repo_dir(HEPATTN_BRANCH="keras-jax", HEPATTN_DIR="/x/kj") == ("keras-jax", "/x/kj")


@pytest.mark.parametrize("script", [*sorted(POLARIS.glob("*.pbs")), POLARIS / "01_setup_env.sh", POLARIS / "repo_dir.sh"], ids=lambda p: p.name)
def test_scripts_parse_and_use_the_configured_clone(script):
    subprocess.run([BASH, "-n", str(script)], check=True)
    text = script.read_text()
    assert 'REPO="$WORK_ROOT/hepattn"' not in text, "hard-coded keras-hgq2 clone"
    if script.suffix == ".pbs" and "REPO=" in text:
        assert 'source polaris/repo_dir.sh\nREPO="$HEPATTN_DIR"' in text


def run_setup(tmp_path: pathlib.Path, clone_branch: str) -> subprocess.CompletedProcess:
    """Run a copy of 01_setup_env.sh against an existing clone on ``clone_branch``; a stub uv keeps it offline."""
    pol = tmp_path / "polaris"
    pol.mkdir()
    for name in ("01_setup_env.sh", "repo_dir.sh"):
        shutil.copy(POLARIS / name, pol / name)
    work = tmp_path / "work"
    (pol / "env.sh").write_text(f'export WORK_ROOT="{work}"\nexport HEPATTN_BRANCH=keras-jax\n')
    clone = work / "hepattn-keras-jax"
    clone.mkdir(parents=True)
    git = ["git", "-C", str(clone), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", clone_branch], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text("#!/bin/sh\necho uv-stub\n")
    (bin_dir / "uv").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    env.pop("HEPATTN_DIR", None)
    env.pop("HEPATTN_BRANCH", None)
    return subprocess.run([BASH, str(pol / "01_setup_env.sh")], env=env, capture_output=True, text=True, check=False)


def test_setup_refuses_to_switch_the_branch_of_an_existing_clone(tmp_path):
    res = run_setup(tmp_path, clone_branch="keras-hgq2")
    assert res.returncode != 0
    assert "is on branch keras-hgq2, not keras-jax" in res.stdout
    assert "repo exists, updating" not in res.stdout


def test_control_setup_updates_a_clone_on_the_right_branch(tmp_path):
    res = run_setup(tmp_path, clone_branch="keras-jax")
    assert "repo exists, updating (branch keras-jax)" in res.stdout
    assert "is on branch" not in res.stdout  # it then fails at `git fetch origin` (no remote), past the guard
