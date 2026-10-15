"""Probe: native from_config on the effective org/main linformer config; record each failure."""
import copy, sys, traceback, yaml
base = yaml.safe_load(open(sys.argv[1])); over = yaml.safe_load(open(sys.argv[2]))
def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out
eff = merge(base, over)["model"]
ia = eff["model"]["init_args"]
print("effective encoder:", ia["encoder"]); print("effective decoder:", ia["decoder"]); print("input_sort_field:", ia.get("input_sort_field"))
from hepattn.keras_native.maskformer import from_config
from hepattn.keras_native.attention import build_attention
try:
    from_config(eff); print("from_config OK")
except Exception as e:
    print("from_config FAILS:", type(e).__name__, e)
# linformer_k key handling in build_attention
att = build_attention(256, attn_type="linformer", num_heads=16, linformer_seq_len=160, linformer_k=64, name="probe")
print("build_attention(linformer_k=64) -> LinformerAttention.k =", att.k, " seq_len =", att.seq_len)
