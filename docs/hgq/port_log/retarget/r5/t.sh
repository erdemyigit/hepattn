#!/bin/zsh
# usage: t.sh <tag> native|hybrid <pytest args...>  (targeted runs; ledger to $S/r5/logs/<tag>.ledger.jsonl when LEDGER=1)
W=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax
L=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r5/logs
cd ${ROOT:-$W}
tag=$1; suite=$2; shift 2
export PYTHONPATH=${ROOT:-$W}/src HEPATTN_PARITY_LEDGER=${LEDGER:-0}
case $suite in
  native) KERAS_BACKEND=jax $W/.venv-jax/bin/python -m pytest -q -p no:cacheprovider "$@" > $L/$tag.log 2>&1;;
  hybrid) KERAS_BACKEND=torch $W/../hepattn/.venv/bin/python -m pytest -q -p no:cacheprovider "$@" > $L/$tag.log 2>&1;;
esac
echo "EXIT $?" >> $L/$tag.log
tail -2 $L/$tag.log
