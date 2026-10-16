#!/bin/zsh
# usage: run_mutant.sh <name>; rsync copy of the worktree, apply, run the R5 native targets there
S=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad
W=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax
D=$S/r5/mut/tree_$1
rm -rf $D; mkdir -p $D
rsync -a --exclude .venv-jax --exclude .git --exclude tests/outputs --exclude logs $W/ $D/
$W/.venv-jax/bin/python $S/r5/mut/mutants.py $1 $D || exit 2
cd $D
PYTHONPATH=$D/src HEPATTN_PARITY_LEDGER=0 KERAS_BACKEND=jax $W/.venv-jax/bin/python -m pytest -q -p no:cacheprovider \
  tests/keras_native/test_quantized_golden.py tests/keras_native/test_org_model.py tests/keras_native/test_mf_golden.py \
  -k "not mf_torch and not chaotic and not test_full_model_paired_delta_reproduces_hybrid" --junitxml=$S/r5/mut/$1.xml > $S/r5/mut/$1.log 2>&1
echo "$1 $(tail -1 $S/r5/mut/$1.log)" >> $S/r5/mut/summary.txt
rm -rf $D
