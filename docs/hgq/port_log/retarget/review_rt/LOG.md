# critic log: RETARGET_ORG_MAIN.md (2026-10-15)
- Pin OK: git ls-remote org -> main ac0da8f (= PR #9 head linformer-phi lineage); masked-linformer main dce2e15 (gh api). linformer-float (Sep 17) / linformer-into-unified-main (Sep 24) are older.
- Reruns: probe_chain (8 blockers, k=256) OK; probe_losses 1.4038/1.309, 0.893919/0.892372, ce 0.5626/1.2e-7, 0.881/0.187 OK; probe_pr8_hybrid 7.15e-7, 0.407, 0.135, depth1 sa_norm_kv 4.8e-7 (= matched floor) OK; scipy nan/-inf ValueError, +inf ok; B2-c hand-calc -1.9696 / 0.0747 OK.
- WRONG: R0 gate (c) tests/matching -> 106 failed on trial-merge, 105 on 37ff127 (lap1015 namespace pkg, AttributeError lap_late) (review_rt/loss_matching*.log); tests/loss 0 failed.
- WRONG: 5.5 sort_index exact equality with padding: torch argsort unstable on +inf ties -> full equal False, valid prefix True, pad suffix same set.
- WRONG: 3.2 "deviation losses.py:5-6 goes away": regression L1 (task.py:1590-1595) and kl_div_loss (loss.py:271-277) still mean an empty selection -> NaN.
- INCONSISTENT: 5.1 shape gate vs 5.6 q_sa tail drop (torch q_sa proj (160,64), orgmain_forward.out); 5.1 attention edits in R2-owned file; R3 gate needs R2.
- OMITTED: maskformer.py:112-113 B=1 all-valid -> key_valid None -> no renorm (masked_linformer.py:16-18); production never (pflow_data.py:120 keeps <160 nodes), test fixtures can.
- Minor: WORKLOG line 156 not 155; option (c) G2 ref is KERAS_JAX_PLAN.md:468-471 not section 0.4 (that is G1 NO-GO).
