#!/bin/zsh
# usage: run_mutant.sh <id>   (applies mutant <id> to a fresh copy of pristine, runs the R3 test set)
S=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad
id=$1; D=$S/r3/mut/run/$id
rm -rf $D; mkdir -p $S/r3/mut/run; cp -R $S/r3/mut/pristine $D
/usr/bin/python3 - "$id" "$D" <<'PY'
import sys; sys.path.insert(0, "/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r3/mut")
from mutants import MUTANTS
mid, root = sys.argv[1], sys.argv[2]
(_, f, old, new), = [m for m in MUTANTS if m[0] == mid]
p = f"{root}/{f}"; s = open(p).read()
assert s.count(old) == 1, (mid, s.count(old)); open(p, "w").write(s.replace(old, new)); print("applied", mid)
PY
cd $D
KERAS_BACKEND=jax PYTHONPATH=$D/src HEPATTN_PARITY_LEDGER=0 /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax/.venv-jax/bin/python -m pytest -q -p no:cacheprovider -W ignore::UserWarning -W ignore::DeprecationWarning \
  tests/keras_native/test_pr8_layers.py tests/keras_native/test_helpers.py tests/keras_native/test_init.py tests/keras_native/test_maskformer.py -rf > $S/r3/mut/$id.log 2>&1
echo "EXIT $?" >> $S/r3/mut/$id.log
rm -rf $D
