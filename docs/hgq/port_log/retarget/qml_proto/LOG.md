# qml_proto working log (quantized masked-linformer design)

## Sources read
- masked-linformer dce2e15 (MIT, Lindsey Gray): project L39-53, _rescale L56-63 (clamp eps^2 before sqrt), attend L66-87.
- org/main ac0da8f attention.py _linformer_attention: kv_sort_idx gather; attn_mask None -> project(k/v, kv_mask) + SDPA unmasked; else attend(q,k,v,E,F, attn_mask & kv_mask). q_mask ignored.
- decoder org/main: q_ca/kv_ca pass attn_mask (attend); q_sa passes only q_mask -> project with mask None (plain E[:n], no renorm).
- keras_native linformer.py (Helen semantics): seqproj QEinsum 'bnd,nk->bkd' with float E as 2nd INPUT (datalane iq), mask in rank-k space (needs k>=kv_len).
- HGQ2 0.1.9: QEinsum ebops = einsum(bits...)->'' with batch dim forced to 1 for EVERY input (base.py:102); datalane default homogeneous_axis=(0,).
  QSoftmax mask multiplicative: mask*x-(1-mask)*1e9, exp*mask, inv(sum+eps) -> all-False row = 0.
  QUnaryFunctionLUT ebops = sum(2^bits_in * bits_out)*1e-4.
- hls4ml 1.3.0 keras_v3 handlers: QEinsum (2 inputs only), QEinsumDense, QSoftmax, QUnaryFunctionLUT (homogeneous table, N<=1e6, power-of-2 range).
  HGQ dais tracer handles Multiply/Einsum/GetItem/Reshape... NOT Where/Sqrt/Greater.
- FPGA_STATUS.md: QUnaryFunctionLUT on rank-3 inputs fails conversion (HGQ2 table-shape bug).
- prod quant spec pflow_hgq.yaml: weight kbi b0=8 i0=2; datalane kif i0=4 f0=8; table kif i0=2 f0=10.

## Results (2026-10-15)
- Float twin (qml_float.py, check_float.py): attend fwd 4.8e-7 (f32) / 7.8e-16 (f64); grads <=4.8e-6 / 1.1e-14; all-False rows exactly 0;
  grad into padded keys exactly 0; project + SDPA 3.6e-7. Mutants caught: mul-mask (NaN), full over E[:n], no final where (6.9e7 on all-False rows).
- Layer prototype (qml_layers.py, run_quant.py): float leaves vs torch attend 3.6e-7 (small) / 5.4e-7 (prod); explicit form 1.9e-7.
  High spec: 7.0e-3 (renorm lanes at deploy precision) -> 6.1e-4 with lanes raised. Mutants (no renorm, no tail) caught.
- Prod spec jit: cache 1, 0 zero-grad trainables, finite grads, EBOPs>0 for attend/project/helen/dense.
- EBOPs prod decoder q_ca shape (B=1): attend 1.050e10, project 5.14e9, helen 5.39e9, dense unmasked 5.74e9, dense masked 1.27e14 (QSoftmax -1e9 fill -> exp iq i=30).
  Core only: attend 6.60e9 vs dense 1.84e9 (3.6x), helen 1.49e9, project 1.24e9; explicit per-query form 7.62e10 vs implemented 6.51e9 (renorm off).
- hls4ml 1.3.0 codegen OK: QEinsumDense bhqn,nk / bhqk,nk; QEinsum bhqk,bqk; kept as QEinsumDense bqn,nk 1-bit lane; rsqrt LUT rank-2; QSoftmax. Rank-3 LUT FAILS (confirmed). Needs KERAS_TORCH_DEVICE=cpu on mac.
- Renorm precision (E on kbi 8/2 grid, step 1/64): r <= 63.5; lanes f=12 -> max rel err 18.5% at 1-2 valid keys, 0.9% at 20; f=16 -> 1.5% / 0.08%.
