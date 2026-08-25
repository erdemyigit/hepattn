#!/usr/bin/env bash
# Run this FIRST, on a Polaris login node. It changes nothing — it only prints the
# facts the other scripts need (project/allocation, queue limits, filesystems, GPUs).
# Copy the values it reports into polaris/env.sh.
set -uo pipefail

echo "=============== ACCOUNT / ALLOCATION ==============="
echo "Projects you can charge to (-A value):"
if command -v sbank >/dev/null 2>&1; then
  sbank-list-allocations 2>/dev/null || sbank 2>/dev/null || echo "  (sbank present but no output)"
fi
qstat -Qf 2>/dev/null | head -0   # warm up
echo "  groups: $(id -Gn)"
echo "  NOTE: your -A project is usually one of your unix groups above."

echo
echo "=============== QUEUES ==============="
qstat -Q 2>/dev/null || echo "  qstat not available on this host"
echo
echo "Per-queue limits (walltime / node counts):"
for q in debug debug-scaling prod preemptable small medium; do
  lim=$(qstat -Qf "$q" 2>/dev/null | grep -iE "resources_max|resources_min|max_walltime|enabled|started" | tr '\n' ' ')
  [ -n "$lim" ] && echo "  $q: $lim"
done

echo
echo "=============== FILESYSTEMS ==============="
for p in "$HOME" /eagle /grand /lus; do
  [ -d "$p" ] && echo "  $p exists"
done
echo "Your writable project dirs (look for one with space for ~15 GB data + checkpoints):"
ls -d /eagle/*/ 2>/dev/null | head -10
ls -d /grand/*/ 2>/dev/null | head -10
echo
echo "Quota on \$HOME:"; myquota 2>/dev/null || quota -s 2>/dev/null || echo "  (no quota command)"

echo
echo "=============== GPUS (login node may show none — that is normal) ==============="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "  no GPUs on this login node (expected)"

echo
echo "=============== MODULES / PYTHON ==============="
module list 2>&1 | head -20
echo "python3: $(command -v python3) -> $(python3 -V 2>&1)"
echo "uv:      $(command -v uv || echo 'not installed — 01_setup_env.sh will install it')"

echo
echo "=============== USER-SITE HAZARD CHECK ==============="
# This is the trap that cost the group days: a stale ~/.local site-packages shadows
# the environment's own torch and produces CUDA-version mismatch errors.
US="$HOME/.local/lib"
if [ -d "$US" ]; then
  echo "  WARNING: $US exists:"
  ls -d "$US"/python3.* 2>/dev/null
  echo "  All scripts here export PYTHONNOUSERSITE=1 so this is ignored."
  echo "  If you still see torch/CUDA version mismatches, move it aside:"
  echo "      mv $US $HOME/.local/lib.disabled"
else
  echo "  clean — no ~/.local/lib present"
fi

echo
echo "=============== NEXT ==============="
echo "  1. cp polaris/env.sh.example polaris/env.sh"
echo "  2. edit polaris/env.sh with the PROJECT / QUEUE / FILESYSTEMS / DATA_ROOT above"
echo "  3. bash polaris/01_setup_env.sh"
