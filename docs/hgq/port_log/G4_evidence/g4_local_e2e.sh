#!/usr/bin/env bash
# Local CPU end-to-end of polaris/19_g4_compare.py (tiny sizes): init, 4 arms uninterrupted (tag u), hybrid-a and jax
# killed and resumed (tag k, same init), reports of both. Run from the g4 worktree root.
set -u
S="$1"; J=../hepattn-keras-jax/.venv-jax/bin/python; H=../hepattn/.venv/bin/python
G="env -u KERAS_BACKEND $J polaris/19_g4_compare.py"
rm -rf "$S/u" "$S/k"; mkdir -p "$S"
$G init --tiny --root "$S" --tag u --k 12 --epoch-steps 4 --batch 4 --accum 2 --ckpt-every 5 --jax-ckpt-every 2 \
  --log-every 1 --num-val 8 --workers 0 --pid-warmup-epochs 1 --hybrid-python $H --jax-python $J > "$S/init_u.out" 2>&1 || { echo INIT-FAILED; exit 1; }
for a in ${ARMS:-hybrid-a hybrid-b jax jax-b}; do
  $G run --root "$S" --tag u --arm $a --hybrid-python $H --jax-python $J --compile-cache "$S/cache" | tail -1 | cut -c1-160
done
mkdir -p "$S/k"; cp "$S/u/manifest.json" "$S/k/"; ln -s "$S/u/hybrid-b" "$S/k/hybrid-b"
for kill in 5 10 ""; do G4_KILL_AT_STEP=$kill $G run --root "$S" --tag k --arm hybrid-a --hybrid-python $H --jax-python $J | tail -1 | cut -c1-160; done
for kill in 4 10 ""; do G4_KILL_AT_STEP=$kill $G run --root "$S" --tag k --arm jax --hybrid-python $H --jax-python $J --compile-cache "$S/cache" | tail -1 | cut -c1-160; done
$G run --root "$S" --tag k --arm jax --hybrid-python $H --jax-python $J | tail -1   # a finished arm: no-op
$G status --root "$S" --tag k | head -30
$G report --root "$S" --tag u --out "$S/g4_u.json"
$G report --root "$S" --tag k --out "$S/g4_k.json"
