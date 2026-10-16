#!/bin/zsh
# usage: run.sh <tag> native|hybrid|torch|matching   (runs in the integration worktree)
W=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax
L=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r4b_fix/logs
cd $W
tag=$1; suite=$2; shift 2
export PYTHONPATH=$W/src HEPATTN_PARITY_LEDGER=0
case $suite in
  native) KERAS_BACKEND=jax $W/.venv-jax/bin/python -m pytest -q -p no:cacheprovider tests/keras_native --junitxml=$L/$tag.native.xml "$@" > $L/$tag.native.log 2>&1;;
  hybrid) KERAS_BACKEND=torch ../hepattn/.venv/bin/python -m pytest -q -p no:cacheprovider tests/keras tests/polaris --junitxml=$L/$tag.hybrid.xml "$@" > $L/$tag.hybrid.log 2>&1;;
  torch) KERAS_BACKEND=torch ../hepattn/.venv/bin/python -m pytest -q -p no:cacheprovider -m 'not gpu and not requiresdata' tests/models tests/polaris tests/loss --junitxml=$L/$tag.torch.xml "$@" > $L/$tag.torch.log 2>&1;;
  matching) KERAS_BACKEND=torch ../hepattn/.venv/bin/python -m pytest -q -p no:cacheprovider -m 'not gpu and not requiresdata' tests/matching --junitxml=$L/$tag.matching.xml "$@" > $L/$tag.matching.log 2>&1;;
esac
echo "EXIT $?" >> $L/$tag.$suite.log
