# trial-merge log (hybrid impact)
- worktree scratchpad/trial-merge, branch scratch/trial-merge from keras-jax 37ff127; merged org/main ac0da8f
- conflicts: .gitignore (union), pixi.lock (theirs), attention.py (theirs first, measure)
- installed masked-linformer==0.1.0 @dce2e15 --no-deps into hepattn/.venv and keras-jax/.venv-jax (torch 2.10.0 both)
- accidental worktree ../x (731db27, clean) created by a typo and removed immediately
- baseline worktree scratchpad/baseline-kj detached at keras-jax 37ff127
## results (logs in this dir)
- base (keras-jax 37ff127): hybrid 104p/8s; native 352p/2s; tests/models+polaris 272p/173s
- merged 12f6097 (attention.py=theirs): hybrid 56p/18F/30E/8s, ONE root cause: KerasMaskFormerDecoderLayer lacks query_update_order (inherited torch forward reads it)
  native 311p/41F/2s: 29 TypeError linformer_proj_dim (torch refs built with Helen API), 11 objective asserts (PR#3 logits cost, a4263c1 per-event mask norm), 1 quantized SwiGLU (generated polaris yaml)
  tests/models+polaris 310p/173s/0F
- f110dcc (+18/-5 LOC: #8 kwargs in keras decoder/encoder, linformer_k alias, polaris rename): hybrid+polaris 107p/8s; native unchanged 41F
- Polaris generated config (merged): float builds; quant FAILS SwiGLU (keras/dense.py:107); encoder block hard-coded -> mixes old encoder with #8 decoder;
  matcher lap1015_late + PYTHONPATH=src -> lap1015 is an empty namespace pkg -> AttributeError lap_late at first match
- proto masked-linformer branch of KerasAttention (~55 LOC): float parity 9e-8 / 1e-7; discrim 9.8e-2; quant high 7.6e-3 (existing attn 7.7e-3), low 2.3e-1
