# planner log: RETARGET_ORG_MAIN.md (2026-10-15)
- Bug numbering verified: WORKLOG.hgq.md:153-154 (1 = a4263c1 cross-event mask norm, 2 = #3 a2bc9c7 double softmax).
- CORRECTION: WORKLOG.hgq.md:155 says cos/sin topo bug NOT on keras-jax. Wrong: keras-jax pflow_data.py:240-241 has
  `torch.cat([track_cosphi, topo_phi])` / sinphi same; identical on org/main ac0da8f. PR #5 open.
- NEW: merged uv.lock is stale. `uv lock --check` exit 1 on trial-merge (227 pkgs, +masked-linformer); exit 0 on baseline-kj
  (226). Logs trial-logs/uv_lock_check{,_baseline}.log. org/main uv.lock has 0 masked-linformer entries (org uses pixi).
  Polaris 01_setup_env.sh:68 `uv sync --group hgq --no-install-project` would re-lock on the login node -> regenerate in R0.
- lap1015 not importable in hepattn/.venv nor .venv-jax (ModuleNotFoundError, -I). Native matcher gate uses org Matcher(scipy).
- Polaris clone is on keras-hgq2 (01_setup_env.sh:38-41): keras-jax changes do not touch running jobs.
- G4/S7 gates in KERAS_JAX_PLAN.md compare against the hybrid -> hybrid must run the same model => recommend (a).
- keras_native/__init__ imports nothing; only matching.py imports jax -> new masked_linformer module can be shared with hybrid (gate H0).
- Double-softmax assignment fixture (2x2, logits [5,0..], dice diag .95 off .30): single -> [0,1], double -> [1,0]. Measured.
- Native SwiGLU: proposal = two QDense halves (x2 with silu inside QDense like other activations) + QMultiply; no new LUT.
- q_sa tail rows E[150:160] are dead in org (project mask=None uses proj[:n]); create tail only when renorm on.
- DONE: plan written + committed on keras-jax efb8eac (docs/hgq/RETARGET_ORG_MAIN.md, 475 lines). B2-a probe rerun: max|org-native(probs)|=0.5626, logits 1.2e-7. Helen-linformer deletions moved to R5 (5 test files import it).
