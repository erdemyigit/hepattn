#!/bin/zsh
# usage: run.sh <id> [pytest args...]   applies mutant <id> to a fresh copy of pristine and runs pytest there
S=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad
M=$S/r3fix/mut; id=$1; shift; D=$M/run/$id
rm -rf $D; mkdir -p $M/run; cp -R $M/pristine $D
/usr/bin/python3 - "$id" "$D" <<'PY'
import sys; sys.path.insert(0, "/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r3fix/mut")
from mutants import MUTANTS
mid, root = sys.argv[1], sys.argv[2]
f, old, new = MUTANTS[mid]
p = f"{root}/{f}"; s = open(p).read()
assert s.count(old) == 1, (mid, s.count(old)); open(p, "w").write(s.replace(old, new)); print("applied", mid)
PY
cd $D
KERAS_BACKEND=jax PYTHONPATH=$D/src HEPATTN_PARITY_LEDGER=0 /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax/.venv-jax/bin/python -m pytest -q -p no:cacheprovider -W ignore::UserWarning -W ignore::DeprecationWarning -rf "$@" > $M/$id.log 2>&1
echo "EXIT $?" >> $M/$id.log
rm -rf $D
