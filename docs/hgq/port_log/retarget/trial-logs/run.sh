#!/bin/zsh
# usage: run.sh <tree> <tag>
T=$1; G=$2; L=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/trial-logs
cd $T
( KERAS_BACKEND=torch PYTHONPATH=$T/src /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn/.venv/bin/python -m pytest -p no:cacheprovider -q -rfEs tests/keras > $L/$G.hybrid.log 2>&1; echo EXIT $? >> $L/$G.hybrid.log ) &
( KERAS_BACKEND=jax PYTHONPATH=$T/src /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax/.venv-jax/bin/python -m pytest -p no:cacheprovider -q -rfEs tests/keras_native > $L/$G.native.log 2>&1; echo EXIT $? >> $L/$G.native.log ) &
( KERAS_BACKEND=torch PYTHONPATH=$T/src /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn/.venv/bin/python -m pytest -p no:cacheprovider -q -rfEs -m 'not gpu and not requiresdata' tests/models tests/polaris > $L/$G.models.log 2>&1; echo EXIT $? >> $L/$G.models.log ) &
wait
