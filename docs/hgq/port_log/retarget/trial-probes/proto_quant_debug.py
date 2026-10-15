import runpy, sys, torch
sys.argv = ["x"]
ns = runpy.run_path(sys.argv[0] if False else "/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/trial-probes/proto_masked_linformer_hybrid.py", run_name="lib")
from hepattn.keras.attention import KerasAttention
from hepattn.keras.factory import LayerFactory
from hepattn.keras.porting import port_attention
from hepattn.models.attention import Attention
HIGH, q, kv, kvm2, am, sort, x, kvm = (ns[k] for k in ("HIGH", "q", "kv", "kvm2", "am", "sort", "x", "kvm"))
# baseline: the existing hybrid quantized torch-SDPA attention at the same spec
torch.manual_seed(0)
ref = Attention(32, num_heads=4).eval(); f = LayerFactory(HIGH)
with f.scopes():
    kat = KerasAttention(32, num_heads=4, factory=f).eval()
with torch.no_grad():
    kat(q, kv, kv, attn_mask=am, kv_mask=kvm2); port_attention(ref, kat)
    print("existing KerasAttention quant-high vs torch: %.2e" % (ref(q, kv, kv, attn_mask=am, kv_mask=kvm2) - kat(q, kv, kv, attn_mask=am, kv_mask=kvm2)).abs().max())
    for spec_name in ("HIGH", "LOW"):
        spec = ns[spec_name]
        ref, kat = ns["pair"](quant=spec); kat(x, kv_mask=kvm); port_attention(ref, kat)
        print(f"proto quant-{spec_name} encoder SA vs torch: %.2e" % (ref(x, kv_mask=kvm) - kat(x, kv_mask=kvm))[kvm].abs().max())
        ref, kat = ns["pair"](quant=spec); kat(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort); port_attention(ref, kat)
        print(f"proto quant-{spec_name} decoder CA vs torch: %.2e" % (ref(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort) - kat(q, kv, kv, attn_mask=am, kv_mask=kvm2, kv_sort_idx=sort)).abs().max())
