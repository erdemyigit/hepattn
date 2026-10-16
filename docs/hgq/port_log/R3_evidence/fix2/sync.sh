#!/bin/zsh
# Rebuild the trial merge: rt/r3 f4f37db + rt/r2 (merge 757d49c), then the R3 worktree's changes since f4f37db
# (committed and uncommitted), then the float-parity patch as it is in the worktree.
WT=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-rt-r3
M=/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/r3fix2/merged
cd $M && git reset -q --hard 757d49c && git clean -qfd -e .venv && find . -name __pycache__ -prune -exec rm -rf {} +
git -C $WT diff f4f37db -- src tests docs/hgq/port_log/R3_evidence/proposed_test_float_parity.patch | git apply --3way || exit 1
git apply --3way docs/hgq/port_log/R3_evidence/proposed_test_float_parity.patch || exit 1
git status --short
