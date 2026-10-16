#!/bin/zsh
R=$(dirname $0)
P=tests/keras_native
TAG=g $R/run.sh m0 $P/test_matching.py $P/test_masked_linformer.py $P/test_pr8_layers.py $P/test_variables.py
TAG=g $R/run.sh n1 $P/test_matching.py
TAG=g $R/run.sh n2 $P/test_masked_linformer.py
TAG=g $R/run.sh n7 $P/test_masked_linformer.py
for m in n3 n5 n6; do TAG=g $R/run.sh $m $P/test_pr8_layers.py; done
TAG=g $R/run.sh n4 $P/test_pr8_layers.py $P/test_variables.py
H="tests/keras/test_linformer_parity.py tests/keras/test_sorting.py tests/keras/test_org_model_parity.py"
for m in m0 h1 h2; do TAG=h $R/runh.sh $m ${=H}; done
echo ALLDONE
