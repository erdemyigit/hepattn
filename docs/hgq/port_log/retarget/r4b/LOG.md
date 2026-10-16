# R4b working log (hybrid masked-Linformer core, gate H0)
Worktree hepattn-keras-jax, branch rt/integration (HEAD dc6a706 at start). Plan RETARGET_ORG_MAIN.md s4 (H0), s6 R4b row.
Hybrid env: hepattn/.venv (no jax: find_spec None; masked_linformer installed). KERAS_BACKEND=torch.

## Design notes
- KerasAttention gains attn_type=linformer: reuses q/k/v/out QDense, torch qkv norms, value residual; core = native
  AttendCore/ProjectCore built lazily at the first forward (shapes q_len/kv_len); proj pending-port until built.
- q_sa: native project_unmasked core has no rows 150-159; hybrid keeps them as a buffer for the torch state_dict.
- Decoder passes linformer_core per call site (q_ca/kv_ca attend; q_sa project_unmasked); encoder project.

## Measurements (macOS arm64 CPU, torch 2.10.0, hepattn/.venv)
- H0 float core probe (probes/h0_float.py, fixtures = test_masked_linformer.py): attend fwd q_ca 8.345e-7 kv_ca 6.557e-7;
  grads q<=7.15e-7 k<=3.87e-7 v 3.58e-7 E<=1.91e-5 F<=1.72e-5; padded grads 0.0; project enc 1.013e-6, enc_short 5.96e-7,
  q_sa 0.0. All <= the JAX-side measurements (R2: 8.345e-7; 8.345e-7/3.278e-7/3.576e-7/4.244e-5/5.913e-5; 1.013e-6/5.96e-7/2.98e-7).
- H0 backend probe (probes/h0_backend.py, H=16, B=2, seed 7): torch==jax exactly except high q_ca train 5.55e-5 (16 elems),
  high kv_ca eval 1.283e-4 / train 4.46e-5 (16 elems each); quantization effect (q - float) 4.6e-2..6.5e-2 (high).
- Full model linformer float parity (probes/model_parity.py seeds 0-15): outputs max 5.088e-6 (seed 3), losses max
  1.875e-7 (seed 8); 4/4 attn-mask booleans equal every seed. Worked with no change besides attention/porting/decoder.

## Summary (done)
Commits 4da3e5e..0e8b026 on rt/integration. H0 pass (no fallback). Hybrid 263p/8s/0xf; native 457p/50xf; torch 522p/173s.
Mutants m01-m15 all killed (m09 redefined after the explicit copy was found redundant). Callback fix: beta now reaches
nested HGQ2 leaves (QSoftmax tables, core leaves). Linformer model EBOPs at init 2.509e11 vs dense 1.907e15.
