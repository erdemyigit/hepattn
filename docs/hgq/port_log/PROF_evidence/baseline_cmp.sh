#!/bin/zsh
# Tiny 2-epoch fit with the S7 baseline src (111aede) and the prof/loop src (profiler off / async / sync+prefetch).
set -u
SP=$1; WT=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-prof
PY=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax/.venv-jax/bin/python
cd $WT
common=(fit --config tests/keras_native/s6b_fixtures/tiny_polaris.yaml --dummy-data --max-epochs 2 --ckpt-every-n-steps 2 --ckpt-save-top-k -1 --log-every-n-steps 1 --compile-cache=)
for name src extra in \
  s7 $SP/s7base/src "" \
  off $WT/src "" \
  async $WT/src "--profile-phases $SP/runs/async.csv" \
  sync_pf $WT/src "--profile-phases $SP/runs/sync.csv --profile-sync --prefetch 2"; do
  rm -rf $SP/runs/$name; mkdir -p $SP/runs
  echo "== $name ($src)"
  PYTHONPATH=$src KERAS_BACKEND=jax $PY -c "import hepattn,sys;print(hepattn.__file__)"
  PYTHONPATH=$src KERAS_BACKEND=jax $PY -m hepattn.experiments.clic.main_jax $common --run-dir $SP/runs/$name ${=extra} > $SP/runs/$name.log 2>&1; echo "exit $?"
done
