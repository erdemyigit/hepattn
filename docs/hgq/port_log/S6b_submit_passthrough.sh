#!/usr/bin/env bash
# S6b fix round 2: polaris/submit.sh hands the job knobs (G3_JOBS, SMOKE_ARGS, ...) to the job. Local check with a fake
# qsub that behaves as PBS without -V: the job sees only the -v variables. The fake job runs the real .pbs lines from
# `set -uo pipefail` up to `source polaris/repo_dir.sh` and prints the knobs. Control: submit.sh of a given revision
# (default 8380d8a, before the fix) loses G3_JOBS.
#
#   bash docs/hgq/port_log/S6b_submit_passthrough.sh <scratch dir> [control revision]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
S="${1:?usage: S6b_submit_passthrough.sh <scratch dir> [control revision]}"
CONTROL="${2:-8380d8a}"
rm -rf "$S"; mkdir -p "$S/polaris" "$S/bin"
cp "$R/polaris/submit.sh" "$R/polaris/16_jax_smoke.pbs" "$R/polaris/17_g3_step.pbs" "$S/polaris/"
printf 'PBS_PROJECT=proj\nPBS_FILESYSTEMS=home:eagle\nWORK_ROOT=/w\nDATA_ROOT=/d\n' > "$S/polaris/env.sh"
cat > "$S/bin/qsub" <<'Q'
#!/usr/bin/env bash
vars=(); script=""
while [ $# -gt 0 ]; do case "$1" in -v) vars+=("$2"); shift 2;; -A|-l|-q|-o|-W) shift 2;; *) script="$1"; shift;; esac; done
prefix="$(sed -n '/^set -uo pipefail/,/^source polaris\/repo_dir.sh/p' "$script" | sed '$d')"
env -i HOME="$HOME" PATH="/usr/bin:/bin" PBS_O_WORKDIR="$(dirname "$(dirname "$script")")" "${vars[@]}" bash -c "$prefix"'
for v in G3_JOBS G3_ARGS G3_DATA_FILE SMOKE_ARGS SMOKE_XLA_FLAGS XLA_FLAGS JAX_COMPILATION_CACHE_DIR; do echo "JOB $v=[${!v-UNSET}]"; done'
Q
chmod +x "$S/bin/qsub"
cd "$S"
export PATH="$S/bin:$PATH"
echo "== 17 with G3_JOBS (commas) and G3_ARGS (spaces)"
G3_JOBS='jax-g1:32,jax-precision:32,hybrid-tf32:16' G3_ARGS='--iters 64 --warmup 8' bash polaris/submit.sh 17_g3_step.pbs 2>&1 | grep "^JOB"
echo "== 16 with SMOKE_XLA_FLAGS= (empty: drop the flag) and SMOKE_ARGS with = , and a quote"
SMOKE_XLA_FLAGS= SMOKE_ARGS="--steps 300 --tag a=b,c'd" bash polaris/submit.sh 16_jax_smoke.pbs 2>&1 | grep "^JOB"
echo "== 16 with no knobs (the defaults)"
bash polaris/submit.sh 16_jax_smoke.pbs 2>&1 | grep "^JOB\|job variables"
echo "== CONTROL: submit.sh at $CONTROL"
git -C "$R" show "$CONTROL:polaris/submit.sh" > polaris/submit.sh
G3_JOBS='jax-g1:32,jax-precision:32' bash polaris/submit.sh 17_g3_step.pbs 2>&1 | grep "^JOB G3_JOBS"
