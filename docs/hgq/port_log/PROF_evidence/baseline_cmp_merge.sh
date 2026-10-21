#!/bin/zsh
# Tiny 2-epoch fit: pre-merge keras-jax src (fb45632, dir name s7 for baseline_cmp2.py) vs merged src (6c46fef), profiler off / async / sync+prefetch.
# Prepare: git archive fb45632 src | tar -x -C $SP/pre; git archive 6c46fef src | tar -x -C $SP/merged. Compare:
# cd $SP/runs; cp -R off ctl_bit; cp -R off ctl_meta; python baseline_controls.py; python baseline_cmp2.py $SP/runs off async sync_pf ctl_bit ctl_meta
set -u
SP=$1; WT=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax
PY=$WT/.venv-jax/bin/python
cd $WT
common=(fit --config tests/keras_native/s6b_fixtures/tiny_polaris.yaml --dummy-data --max-epochs 2 --ckpt-every-n-steps 2 --ckpt-save-top-k -1 --log-every-n-steps 1 --compile-cache=)
for name src extra in \
  s7 $SP/pre/src "" \
  off $SP/merged/src "" \
  async $SP/merged/src "--profile-phases $SP/runs/async.csv" \
  sync_pf $SP/merged/src "--profile-sync --profile-phases $SP/runs/sync.csv --prefetch 2"; do
  rm -rf $SP/runs/$name; mkdir -p $SP/runs
  echo "== $name ($src)"
  env -u KERAS_BACKEND PYTHONPATH=$src KERAS_BACKEND=jax $PY -c "import hepattn;print(hepattn.__file__)"
  env -u KERAS_BACKEND PYTHONPATH=$src KERAS_BACKEND=jax $PY -m hepattn.experiments.clic.main_jax $common --run-dir $SP/runs/$name ${=extra} > $SP/runs/$name.log 2>&1; echo "exit $?"
done
