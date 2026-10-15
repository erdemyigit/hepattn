"""EBOPs of one decoder cross-attention (CLIC shapes: 150 queries x 160 constituents, dim 256, 16 heads) at the Polaris spec."""
import runpy, torch
ns = runpy.run_path("/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/trial-probes/proto_masked_linformer_hybrid.py", run_name="lib")
from hepattn.keras.attention import KerasAttention
from hepattn.keras.factory import LayerFactory
import keras
SPEC = {"weight": {"default_q_type": "kbi", "b0": 8, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
        "table": {"default_q_type": "kif", "i0": 2, "f0": 10}, "ebops": {"beta0": 0.0}}
g = torch.Generator().manual_seed(1)
q = torch.randn(2, 150, 256, generator=g); kv = torch.randn(2, 160, 256, generator=g)
kvm = torch.ones(2, 160, dtype=torch.bool); kvm[:, 140:] = False
am = torch.rand(2, 150, 160, generator=g) > 0.7

def ebops(mod):
    tot = {}
    for name, m in mod.named_modules():
        if isinstance(m, keras.layers.Layer) and hasattr(m, "ebops"):
            tot[name] = float(m.ebops)
    return tot

for tag in ("quadratic", "masked-linformer k=64"):
    f = LayerFactory(SPEC)
    with f.scopes():
        att = KerasAttention(256, num_heads=16, factory=f) if tag == "quadratic" else ns["KerasMaskedLinformer"](256, 16, 160, 64, f)
    att.train()
    att(q, kv, kv, attn_mask=am, kv_mask=kvm)
    e = ebops(att)
    core = {k: v for k, v in e.items() if not k.endswith("proj")}
    print(f"{tag}: total EBOPs {sum(e.values()):.3e}; attention core (non-projection) {sum(core.values()):.3e}: " + ", ".join(f"{k}={v:.2e}" for k, v in core.items()))
