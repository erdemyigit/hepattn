# R5 integrate log (scratch; durable copy -> docs/hgq/port_log/R5.md)
Start HEAD 93eb295 on rt/integration.
## S2 layer goldens (done, uncommitted)
- cases.py: new lin_enc/lin_q_ca/lin_kv_ca (kind attn, org Attention linformer k=64, qkv_norm LayerNorm), enc_lin = PR#8 encoder + phi sort,
  dense_{ffn,innet}_swiglu. Helen lin_self/lin_cross + "mid" mode removed. Leaf rule select_leaves (adds QEinsumDense,
  QUnaryFunctionLUT, EinsumDense, SeqProjection.tail_q; drops nested tables).
- make_golden.py GOLDEN_OUT + case args. gen1 in $S/r5/gen1: kept 6 files byte-identical (cmp silent) and array-identical.
- measure_s2.out: all rows; lin high <= 8.8e-5; low 0 for lin; deltas ratio >= 36. controls_s2.out: >= 8.6e4 x tol.
- test_quantized_golden: 77 passed.
