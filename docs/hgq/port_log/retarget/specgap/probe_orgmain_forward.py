"""Probe org/main ac0da8f float model (base.yaml + linformer.yaml): build, trace attention call sites, run loss.

Matcher solver forced to scipy (lap1015 is not installed here); everything else is the effective config.
"""
import copy, importlib, sys, yaml
import torch
from torch import nn

base = yaml.safe_load(open(sys.argv[1])); over = yaml.safe_load(open(sys.argv[2]))
def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out
ia = merge(base, over)["model"]["model"]["init_args"]
ia["matcher"]["init_args"]["default_solver"] = "scipy"

def load(path):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)

def inst(spec):
    if isinstance(spec, dict) and "class_path" in spec:
        args = {k: inst(v) for k, v in (spec.get("init_args") or {}).items()}
        return load(spec["class_path"])(**args)
    if isinstance(spec, list):
        return [inst(v) for v in spec]
    if isinstance(spec, str) and spec.startswith("torch.nn."):
        return load(spec)()
    return spec

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models import attention as A
args = {k: (v if k == "decoder" else inst(v)) for k, v in ia.items()}
args["decoder"] = MaskFormerDecoder(**ia["decoder"])
model = load("hepattn.models.MaskFormer")(**args)
torch.manual_seed(0)

# --- static structure facts
enc = model.encoder; dec = model.decoder
for i, l in enumerate(enc.layers):
    print(f"enc L{i}: attn pre-norm={type(l.attn.norm).__name__} qkv_norm={l.attn.fn.qkv_norm} q_norm_affine={getattr(l.attn.fn,'q_norm',None) is not None and l.attn.fn.q_norm.elementwise_affine} "
          f"dense post_norm={l.dense.post_norm} dense norm={type(l.dense.norm).__name__} value_resid_mix={hasattr(l.attn.fn,'value_residual_mix')} proj_k={tuple(l.attn.fn.proj_k.shape)} ls={type(l.attn.ls).__name__} dp={type(l.attn.dp).__name__}")
for i, l in enumerate(dec.decoder_layers):
    print(f"dec L{i}: q_ca norm={type(l.q_ca.norm).__name__} q_sa norm={type(l.q_sa.norm).__name__} q_dense post={l.q_dense.post_norm} kv_dense post={l.kv_dense.post_norm} order={l.query_update_order} sa_norm_kv={l.self_attn_norm_kv} "
          f"proj_k q_ca/q_sa/kv_ca={tuple(l.q_ca.fn.proj_k.shape)}/{tuple(l.q_sa.fn.proj_k.shape)}/{tuple(l.kv_ca.fn.proj_k.shape)} value_residual={l.q_ca.fn.value_residual}")
print("encoder Dense:", enc.layers[0].dense.fn.net)
print("innet:", model.input_nets[0].net.net)
for t in model.tasks:
    nets = {k: getattr(t, k) for k in ("net", "object_net", "constituent_net", "node_net") if getattr(t, k, None) is not None}
    print(t.name, "run@layer0..3:", [t.should_run_at_layer(i) for i in range(4)], {k: [m for m in v.net if isinstance(m, nn.Linear)] for k, v in nets.items() if hasattr(v, 'net')})

# --- trace linformer call sites
calls = []
orig = A.Attention._linformer_attention
names = {}
for i, l in enumerate(enc.layers): names[id(l.attn.fn)] = f"enc{i}.sa"
for i, l in enumerate(dec.decoder_layers):
    names[id(l.q_ca.fn)] = f"dec{i}.q_ca"; names[id(l.q_sa.fn)] = f"dec{i}.q_sa"; names[id(l.kv_ca.fn)] = f"dec{i}.kv_ca"
def traced(self, q, k, v, kv_mask, attn_mask, kv_sort_idx):
    calls.append((names[id(self)], tuple(q.shape), tuple(k.shape), None if kv_mask is None else tuple(kv_mask.shape),
                  None if attn_mask is None else tuple(attn_mask.shape), kv_sort_idx is not None, "attend" if attn_mask is not None else "project+sdpa"))
    return orig(self, q, k, v, kv_mask, attn_mask, kv_sort_idx)
A.Attention._linformer_attention = traced

B, N, P = 3, 160, 150
g = torch.Generator().manual_seed(1)
nv = torch.tensor([120, 37, 159]); npart = torch.tensor([40, 9, 60])
valid = torch.arange(N)[None] < nv[:, None]
inputs = {"node_features": torch.randn(B, N, 27, generator=g), "node_valid": valid,
          "node_eta": torch.randn(B, N, generator=g), "node_phi": (torch.rand(B, N, generator=g) * 6.28 - 3.14) * valid,
          "node_e": torch.rand(B, N, generator=g) * 10 * valid, "node_pt": torch.rand(B, N, generator=g) * 5 * valid,
          "node_is_track": (torch.rand(B, N, generator=g) < 0.4).float() * valid,
          "node_sinphi": torch.zeros(B, N), "node_cosphi": torch.zeros(B, N)}
pvalid = torch.arange(P)[None] < npart[:, None]
inc = torch.rand(B, P, N, generator=g) * pvalid[..., None] * valid[:, None]
inc = inc / inc.sum(1, keepdim=True).clamp_min(1e-6)
targets = {"particle_valid": pvalid, "node_valid": valid, "particle_class": torch.where(pvalid, torch.randint(0, 5, (B, P), generator=g), 5),
           "particle_node_valid": inc > 0.01, "particle_incidence": inc}
for f in ["e", "pt", "eta", "sinphi", "cosphi"]:
    targets[f"particle_{f}"] = torch.where(pvalid, torch.randn(B, P, generator=g), torch.nan)
model.eval()
with torch.no_grad():
    out = model(inputs)
    print("output layers:", list(out), {k: list(v) for k, v in out.items()})
    seen = {}
    for c in calls:
        seen.setdefault(c[0].split(".")[1] if "." in c[0] else c[0], set()).add(c[1:])
    for site, s in seen.items():
        print("site", site, "->", sorted(s))
    _, _, losses = model.loss(out, targets)
    print({lay: {t: {n: round(float(v), 4) for n, v in d.items()} for t, d in tl.items()} for lay, tl in losses.items()})
    # sort facts
    ks = model.encoder  # key_sort_idx recomputed for display
    from hepattn.models.encoder import sort_index
    si = sort_index(inputs["node_phi"], valid)
    print("sort_index padding last:", all(bool(valid[b].gather(0, si[b])[: int(nv[b])].all()) and not bool(valid[b].gather(0, si[b])[int(nv[b]):].any()) for b in range(B)))
