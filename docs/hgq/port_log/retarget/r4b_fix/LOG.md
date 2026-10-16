# R4b fixer log (start HEAD 0e8b026)
Issues: M1 lazy core state_dict (2 reviewers), M2 evaluate.py top-level walk, minors: 16-head backend row, train-mode hybrid row, optimizer-test beta walk, gitignore (R0/R5), uint32 note.
Plan: attention.py raise in _save_to_state_dict/_load_from_state_dict while core unbuilt; KerasMaskFormer.materialize(inputs);
evaluate.py via callbacks leaves; rewrite test_effective_ebops "slope far below total" (encodes old bug).
## progress
- attention.py: _save_to_state_dict/_load_from_state_dict raise on unbuilt linformer; maskformer.materialize(inputs).
- roundtrip: 4.489e-6 vs control 57.8, 132 core keys of 469 (logs/roundtrip.out).
- evaluate.py on callbacks leaves; floor at beta 0 (probes hid floor in fp32: calibration 7.593 vs 7.5).
  regions by shortest path (first path put task heads in decoder). E_eff = total + task heads to <1e-7 (evaluate_ebops2.out); linformer heads 10.8%.
- 16-head golden rows (2.49 MB): high ties up to 1.656e-4, 112 elements; prod 0; control f0-1 >= 57686 elements.
- train-mode test: train/eval from same state: high 4e-4, prod 37-45 (WRAP vs i-fit). state update checked.
- found: port_to_keras.build_keras_maskformer_from_config drops input_sort_field (open issue, not our file).
