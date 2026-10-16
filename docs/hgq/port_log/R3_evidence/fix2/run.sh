#!/bin/zsh
# usage: run.sh <id> [pytest args...]: copy the mutant's pristine root, apply the mutant, run pytest there, log in mut/<id>.log
S=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad
MD=$S/r3fix2/mut; id=$1; shift; D=$MD/run/$id
root=$(/usr/bin/python3 -c "import sys; sys.path.insert(0, '$MD'); from mutants import MUTANTS; print(MUTANTS['$id'][0])")
rm -rf $D; cp -R $MD/pristine_$root $D
/usr/bin/python3 - "$id" "$D" <<'PY'
import sys; sys.path.insert(0, "/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r3fix2/mut")
from mutants import MUTANTS
mid, root = sys.argv[1], sys.argv[2]
for f, old, new in MUTANTS[mid][1]:
    p = f"{root}/{f}"; s = open(p).read()
    assert s.count(old) == 1, (mid, f, s.count(old)); open(p, "w").write(s.replace(old, new))
print("applied", mid)
PY
cd $D
KERAS_BACKEND=jax PYTHONPATH=$D/src HEPATTN_PARITY_LEDGER=0 /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax/.venv-jax/bin/python -m pytest -q -p no:cacheprovider -W ignore::UserWarning -W ignore::DeprecationWarning -rf "$@" > $MD/$id.log 2>&1
echo "EXIT $?" >> $MD/$id.log
rm -rf $D
