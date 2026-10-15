"""Probe: strip each org/main option the native from_config rejects, one at a time, to list every blocker."""
import copy, sys, yaml
base = yaml.safe_load(open(sys.argv[1])); over = yaml.safe_load(open(sys.argv[2]))
def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out
eff = merge(base, over)["model"]; ia = eff["model"]["init_args"]
from hepattn.keras_native.maskformer import from_config
patches = [
    ("drop input_sort_field", lambda ia: ia.pop("input_sort_field")),
    ("unwrap encoder class_path/init_args", lambda ia: ia.__setitem__("encoder", ia["encoder"]["init_args"])),
    ("drop encoder.dim (dup of MaskFormer dim)", lambda ia: ia["encoder"].pop("dim")),
    ("drop encoder.dense_norm_placement", lambda ia: ia["encoder"].pop("dense_norm_placement")),
    ("encoder.norm FastLayerNorm->LayerNorm", lambda ia: ia["encoder"].__setitem__("norm", "LayerNorm")),
    ("drop decoder dense_norm_placement", lambda ia: ia["decoder"]["decoder_layer_config"].pop("dense_norm_placement")),
    ("drop decoder query_update_order", lambda ia: ia["decoder"]["decoder_layer_config"].pop("query_update_order")),
    ("drop decoder self_attn_norm_kv", lambda ia: ia["decoder"]["decoder_layer_config"].pop("self_attn_norm_kv")),
    ("decoder norm FastLayerNorm->LayerNorm", lambda ia: ia["decoder"]["decoder_layer_config"].__setitem__("norm", "LayerNorm")),
]
for label, patch in [("as is", lambda ia: None), *patches]:
    patch(ia)
    try:
        m = from_config(copy.deepcopy(eff)); print(f"[{label}] from_config OK")
        break
    except Exception as e:
        print(f"[{label}] -> {type(e).__name__}: {str(e)[:160]}")
# quantized SwiGLU
try:
    q = copy.deepcopy(eff); q["model"]["init_args"]["quant"] = {"weight": {"default_q_type": "kbi", "b0": 8, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8}, "table": {"default_q_type": "kif", "i0": 2, "f0": 10}}
    from_config(q); print("[quant] OK")
except Exception as e:
    print(f"[quant] -> {type(e).__name__}: {str(e)[:160]}")
from hepattn.keras_native.objective import specs_from_config, matcher_n_jobs_from_config
print("specs_from_config OK:", [s.name for s in specs_from_config(ia["tasks"]["init_args"]["modules"])])
try:
    matcher_n_jobs_from_config(ia["matcher"]); print("matcher OK")
except Exception as e:
    print(f"[matcher] -> {type(e).__name__}: {str(e)[:200]}")
import os
os.chdir(sys.argv[3])
m = from_config(copy.deepcopy(eff))
enc_attn = m.encoder.layers[0].attn.fn; dec_attn = m.decoder.layers[0].q_ca.fn
print("built after patches; encoder attn:", type(enc_attn).__name__, "seq_len", enc_attn.seq_len, "k", enc_attn.k, "| decoder q_ca:", type(dec_attn).__name__, "seq_len", dec_attn.seq_len, "k", dec_attn.k)
