#!/usr/bin/env bash
# Stage the CLIC ROOT files onto Polaris project space. Run on a Polaris LOGIN node.
#
#   bash polaris/02_stage_data.sh
#
# Pulls ~12 GB from the CMU cluster (falcon). If ANL cannot reach falcon inbound, use
# the PUSH alternative printed at the end (run that from falcon or your laptop).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

SRC_HOST="${SRC_HOST:-eertorer@falcon.phys.cmu.edu}"
SRC_DIR="${SRC_DIR:-~/data/clic}"

echo "==> destination: $DATA_ROOT"
mkdir -p "$DATA_ROOT"

# Only the two files training needs. train_clic_fix.root is ~11.5 GB, val ~0.3 GB.
# test_clic_common_infer.root is deliberately NOT staged: it carries -9999 sentinel
# indices that crash the loader, and the configs evaluate on the val file instead.
FILES="train_clic_fix.root val_clic_fix.root"

echo "==> pulling: $FILES"
for f in $FILES; do
  if [ -s "$DATA_ROOT/$f" ]; then
    echo "    $f already present ($(du -h "$DATA_ROOT/$f" | cut -f1)) — skipping"
    continue
  fi
  echo "    $f ..."
  rsync -h --progress --partial --inplace "$SRC_HOST:$SRC_DIR/$f" "$DATA_ROOT/$f"
done

echo
echo "==> staged:"
ls -la "$DATA_ROOT"/*.root | awk '{printf "    %.1f GB  %s\n", $5/1073741824, $NF}'

echo
echo "==> integrity check (compares sizes against the source)"
for f in $FILES; do
  loc=$(stat -c%s "$DATA_ROOT/$f" 2>/dev/null || stat -f%z "$DATA_ROOT/$f")
  rem=$(ssh "$SRC_HOST" "stat -c%s $SRC_DIR/$f" 2>/dev/null || echo "?")
  if [ "$loc" = "$rem" ]; then echo "    OK   $f ($loc bytes)"
  else echo "    MISMATCH $f: local=$loc remote=$rem — re-run this script"; fi
done

cat <<EOF

If the pull failed because ANL cannot reach CMU, PUSH instead — run this ON falcon
(or wherever the data lives):

  rsync -h --progress --partial \\
    ~/data/clic/{train_clic_fix.root,val_clic_fix.root} \\
    <you>@polaris.alcf.anl.gov:$DATA_ROOT/

Next: qsub polaris/04_smoke.pbs
EOF
