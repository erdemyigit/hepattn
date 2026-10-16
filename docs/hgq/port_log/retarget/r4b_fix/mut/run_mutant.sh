#!/bin/zsh
# usage: run_mutant.sh <name>   (copies the integration worktree, applies one mutant, runs the R4b targets)
W=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax
M=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r4b_fix/mut
name=$1
C=$M/copies/$name
rm -rf $C; mkdir -p $C
rsync -a --exclude .git --exclude .venv-jax --exclude .serena --exclude 'tests/outputs' --exclude '__pycache__' --exclude 'docs' $W/ $C/
$W/.venv-jax/bin/python - "$C" "$name" <<'PY'
import sys, pathlib
sys.path.insert(0, "/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r4b_fix/mut")
from mutants import MUTANTS
root, name = sys.argv[1:]
if name != "m00_unmutated":
    f, old, new = MUTANTS[name]
    p = pathlib.Path(root) / f
    s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new))
PY
cd $C
KERAS_BACKEND=torch PYTHONPATH=$C/src HEPATTN_PARITY_LEDGER=0 $W/../hepattn/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/keras/test_linformer_parity.py tests/keras/test_linformer_state_dict.py tests/keras/test_beta_reaches_every_leaf.py tests/keras/test_effective_ebops.py tests/keras/test_beta_pid.py -W ignore \
  \
  > $M/$name.log 2>&1
echo "EXIT $?" >> $M/$name.log
