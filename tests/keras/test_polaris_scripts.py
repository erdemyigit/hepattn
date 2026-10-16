"""The Polaris shell scripts pick the clone from HEPATTN_BRANCH / HEPATTN_DIR (polaris/repo_dir.sh).

Jobs that are running use $WORK_ROOT/hepattn on keras-hgq2 (docs/hgq/RETARGET_ORG_MAIN.md section 2, item 6), so a
new branch must get its own clone, every job script must run in the clone the setup made, and the setup must never
switch the branch of an existing clone.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
POLARIS = REPO / "polaris"
BASH = shutil.which("bash")
GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(BASH is None or GIT is None, reason="bash or git not available")


def git(where: pathlib.Path, *args: str) -> None:
    subprocess.run([GIT, "-C", str(where), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True, capture_output=True)


def checkout(where: pathlib.Path, branch: str | None, env_sh: str | None = None) -> pathlib.Path:
    """A directory with polaris/repo_dir.sh (and env.sh): a git clone on ``branch``, or no clone at all for None."""
    (where / "polaris").mkdir(parents=True)
    shutil.copy(POLARIS / "repo_dir.sh", where / "polaris" / "repo_dir.sh")
    if env_sh is not None:
        (where / "polaris" / "env.sh").write_text(env_sh)
    if branch is not None:
        git(where, "init", "-q", "-b", branch)
        git(where, "add", "-A")
        git(where, "commit", "-q", "-m", "x")
    return where


def repo_dir(where: pathlib.Path, **env: str) -> tuple[str, str]:
    """HEPATTN_BRANCH and HEPATTN_DIR as a job in ``where`` resolves them: cd there, source env.sh, then repo_dir.sh."""
    env_sh = "source polaris/env.sh; " if (where / "polaris" / "env.sh").exists() else "WORK_ROOT=/w; "
    script = f'cd "{where}"; {env_sh}source polaris/repo_dir.sh; echo "$HEPATTN_BRANCH $HEPATTN_DIR"'
    clean = {k: v for k, v in os.environ.items() if k not in {"HEPATTN_BRANCH", "HEPATTN_DIR"}}
    out = subprocess.run([BASH, "-c", script], env={**clean, **env}, check=True, capture_output=True, text=True).stdout.split()
    return out[0], out[1]


def test_repo_dir_defaults_and_overrides(tmp_path):
    bare = checkout(tmp_path / "bare", branch=None)  # not a clone: the keras-hgq2 default
    assert repo_dir(bare) == ("keras-hgq2", "/w/hepattn")  # unchanged for the running keras-hgq2 jobs
    assert repo_dir(bare, HEPATTN_BRANCH="keras-jax") == ("keras-jax", "/w/hepattn-keras-jax")
    assert repo_dir(bare, HEPATTN_BRANCH="rt/integration") == ("rt/integration", "/w/hepattn-rt-integration")
    assert repo_dir(bare, HEPATTN_BRANCH="keras-jax", HEPATTN_DIR="/x/kj") == ("keras-jax", "/x/kj")


def test_repo_dir_defaults_to_the_branch_of_its_clone(tmp_path):
    """A job submitted from a keras-jax clone runs that clone, though qsub passes it no environment."""
    clone = checkout(tmp_path / "kj", branch="keras-jax")
    assert repo_dir(clone) == ("keras-jax", "/w/hepattn-keras-jax")
    assert repo_dir(clone, HEPATTN_BRANCH="other") == ("other", "/w/hepattn-other")
    git(clone, "checkout", "-q", "--detach")
    assert repo_dir(clone) == ("keras-hgq2", "/w/hepattn")


EXAMPLE = (POLARIS / "env.sh.example").read_text()
EXAMPLE_WORK_ROOT = "/eagle/CHANGEME/hepattn-hgq"


@pytest.mark.parametrize("branch", [None, "keras-jax"])
def test_env_sh_example_keeps_the_branch_of_the_command_line_and_the_clone(tmp_path, branch):
    """An env.sh copied from the example must not reset HEPATTN_BRANCH to keras-hgq2 (review fix round 2)."""
    where = checkout(tmp_path / "c", branch=branch, env_sh=EXAMPLE)
    assert repo_dir(where, HEPATTN_BRANCH="keras-jax") == ("keras-jax", f"{EXAMPLE_WORK_ROOT}/hepattn-keras-jax")
    expected = "keras-jax" if branch else "keras-hgq2"
    assert repo_dir(where)[0] == expected  # a job: no HEPATTN_BRANCH in its environment


@pytest.mark.parametrize("script", [*sorted(POLARIS.glob("*.pbs")), POLARIS / "01_setup_env.sh", POLARIS / "repo_dir.sh"], ids=lambda p: p.name)
def test_scripts_parse_and_use_the_configured_clone(script):
    subprocess.run([BASH, "-n", str(script)], check=True)
    text = script.read_text()
    assert 'REPO="$WORK_ROOT/hepattn"' not in text, "hard-coded keras-hgq2 clone"
    if script.suffix == ".pbs" and "REPO=" in text:
        assert 'source polaris/repo_dir.sh\nREPO="$HEPATTN_DIR"' in text


# Stand-ins for the venv python of the verification step: just enough of each package for its import lines.
FAKE_MODULES = {
    "torch/__init__.py": "__version__ = 'fake'\nclass version: cuda = None\nclass cuda:\n    @staticmethod\n    def is_available(): return False\n",
    "keras/__init__.py": "__version__ = 'fake'\nclass backend:\n    @staticmethod\n    def backend(): return 'torch'\n",
    "hgq/__init__.py": "",
    "hgq/layers.py": "QDense = None\n",
    "hepattn/__init__.py": "",
    "hepattn/keras/__init__.py": "",
    "hepattn/keras/maskformer.py": "KerasMaskFormer = None\n",
}


def run_setup(
    tmp_path: pathlib.Path,
    clones: dict[str, str],
    env_sh: str,
    runner_branch: str | None = None,
    masked_linformer: bool = True,
    **env: str,
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run a copy of 01_setup_env.sh offline; returns its result and the argv of each ``uv sync`` call.

    ``clones`` maps directory names under $WORK_ROOT to the branch they are cloned on, from a local origin whose
    branches hold polaris/repo_dir.sh (keras-hgq2 has none, as on GitHub). The setup runs from ``runner_branch``
    (None: a directory that is not a clone). A stub uv records its arguments and its ``venv`` makes a python that
    runs the verification against FAKE_MODULES (plus masked_linformer when ``masked_linformer``).
    """
    work = tmp_path / "work"
    origin = tmp_path / "origin.git"
    subprocess.run([GIT, "init", "-q", "--bare", str(origin)], check=True)
    for branch in sorted(set(clones.values())):
        seed = tmp_path / f"seed-{branch}"
        (seed / "polaris").mkdir(parents=True)
        (seed / "polaris" / ".keep").write_text("")
        if branch != "keras-hgq2":
            shutil.copy(POLARIS / "repo_dir.sh", seed / "polaris" / "repo_dir.sh")
        git(seed, "init", "-q", "-b", branch)
        git(seed, "add", "-A")
        git(seed, "commit", "-q", "-m", "x")
        git(seed, "push", "-q", str(origin), f"HEAD:refs/heads/{branch}")
    for name, branch in clones.items():
        subprocess.run([GIT, "clone", "-q", "-b", branch, str(origin), str(work / name)], check=True, capture_output=True)
    runner = tmp_path / "runner"
    checkout(runner, runner_branch, env_sh=env_sh.replace("@WORK@", str(work)))
    shutil.copy(POLARIS / "01_setup_env.sh", runner / "polaris" / "01_setup_env.sh")
    fake = tmp_path / "fake"
    for rel, text in {**FAKE_MODULES, **({"masked_linformer.py": ""} if masked_linformer else {})}.items():
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text(text)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv.log"
    python = bin_dir / "venv_python"  # -S: no site-packages, so only the fakes import
    python.write_text(f'#!/bin/sh\nPYTHONPATH="{fake}" exec "{sys.executable}" -S "$@"\n')
    python.chmod(0o755)
    uv = f'#!/bin/sh\necho "$@" >> "{log}"\nif [ "$1" = venv ]; then mkdir -p .venv/bin && cp "{python}" .venv/bin/python; fi\n'
    (bin_dir / "uv").write_text(uv)
    (bin_dir / "uv").chmod(0o755)
    clean = {k: v for k, v in os.environ.items() if k not in {"HEPATTN_BRANCH", "HEPATTN_DIR", "PYTHONPATH"}}
    res = subprocess.run(
        [BASH, str(runner / "polaris" / "01_setup_env.sh")],
        env={**clean, "PATH": f"{bin_dir}:{os.environ['PATH']}", **env},
        capture_output=True,
        text=True,
        check=False,
    )
    syncs = [line for line in (log.read_text().splitlines() if log.exists() else []) if line.startswith("sync")]
    return res, syncs


ENV_PLAIN = 'export WORK_ROOT="@WORK@"\nexport DATA_ROOT=/data\n'
ENV_KJ = ENV_PLAIN + "export HEPATTN_BRANCH=keras-jax\n"


def test_setup_refuses_to_switch_the_branch_of_an_existing_clone(tmp_path):
    res, syncs = run_setup(tmp_path, {"hepattn-keras-jax": "keras-hgq2"}, ENV_KJ)
    assert res.returncode != 0
    assert "is on branch keras-hgq2, not keras-jax" in res.stdout
    assert "repo exists, updating" not in res.stdout
    assert not syncs


@pytest.mark.parametrize("masked_linformer", [True, False])
def test_control_setup_installs_a_branch_from_its_lock(tmp_path, masked_linformer):
    """keras-jax: the clone is updated, its jobs resolve to it, ``uv sync --locked``, and masked_linformer must import."""
    res, syncs = run_setup(tmp_path, {"hepattn-keras-jax": "keras-jax"}, ENV_KJ, masked_linformer=masked_linformer)
    assert "repo exists, updating (branch keras-jax)" in res.stdout
    assert f"jobs submitted from {tmp_path / 'work' / 'hepattn-keras-jax'} run in it" in res.stdout
    assert len(syncs) == 1 and "--locked" in syncs[0].split(), syncs
    assert ("ALL CHECKS PASSED" in res.stdout) is masked_linformer, res.stdout + res.stderr
    assert (res.returncode == 0) is masked_linformer
    if not masked_linformer:
        assert "No module named 'masked_linformer'" in res.stderr


def test_control_setup_keeps_keras_hgq2_unlocked(tmp_path):
    """keras-hgq2 keeps its old install (no --locked) and does not need masked_linformer."""
    res, syncs = run_setup(tmp_path, {"hepattn": "keras-hgq2"}, ENV_PLAIN, masked_linformer=False)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "repo exists, updating (branch keras-hgq2)" in res.stdout
    assert len(syncs) == 1 and "--locked" not in syncs[0].split(), syncs


@pytest.mark.parametrize("how", ["command_line", "runner_clone"])
def test_setup_with_the_example_env_sh_leaves_the_keras_hgq2_clone_alone(tmp_path, how):
    """The review case: env.sh copied from env.sh.example, keras-jax asked for on the command line or by running from
    a keras-jax checkout. The setup must update the keras-jax clone, never the keras-hgq2 one that jobs run from.
    """
    env_sh = EXAMPLE + '\nexport WORK_ROOT="@WORK@"\n'
    clones = {"hepattn": "keras-hgq2", "hepattn-keras-jax": "keras-jax"}
    env = {"HEPATTN_BRANCH": "keras-jax"} if how == "command_line" else {}
    res, syncs = run_setup(tmp_path, clones, env_sh, runner_branch="keras-jax" if how == "runner_clone" else None, **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "repo exists, updating (branch keras-jax)" in res.stdout
    assert (tmp_path / "work" / "hepattn-keras-jax" / "polaris" / "env.sh").exists()
    assert not (tmp_path / "work" / "hepattn" / "polaris" / "env.sh").exists()
    assert len(syncs) == 1 and "--locked" in syncs[0].split()


def test_setup_refuses_a_clone_its_jobs_would_not_run_in(tmp_path):
    """A custom HEPATTN_DIR on the command line only: jobs (which read only env.sh) would run elsewhere."""
    res, syncs = run_setup(tmp_path, {"custom": "keras-jax"}, ENV_PLAIN, HEPATTN_BRANCH="keras-jax", HEPATTN_DIR=str(tmp_path / "work" / "custom"))
    assert res.returncode != 0
    assert f"jobs submitted from {tmp_path / 'work' / 'custom'} would run in {tmp_path / 'work' / 'hepattn-keras-jax'}" in res.stdout
    assert not syncs
