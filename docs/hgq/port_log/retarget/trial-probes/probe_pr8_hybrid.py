"""Float parity of the hybrid decoder/encoder layers under the PR #8 options, plus a discrimination check."""
import torch
from hepattn.keras import set_keras_default_device
set_keras_default_device("cpu")
from hepattn.keras.decoder import KerasMaskFormerDecoderLayer
from hepattn.keras.encoder import KerasEncoderLayer
from hepattn.keras.porting import port_attention, port_dense, port_residual, port_encoder_layer
from hepattn.models.decoder import MaskFormerDecoderLayer
from hepattn.models.encoder import EncoderLayer

DIM = 32
PR8 = {"dim": DIM, "norm": "FastLayerNorm", "hybrid_norm": True, "dense_norm_placement": "legacy",
       "query_update_order": "ca_sa_dense", "self_attn_norm_kv": True,
       "attn_kwargs": {"num_heads": 16}, "dense_kwargs": {"activation": "SwiGLU"}}
OLD = {k: v for k, v in PR8.items() if k not in {"dense_norm_placement", "query_update_order", "self_attn_norm_kv"}}

def dec_pair(tcfg, kcfg, depth):
    torch.manual_seed(depth)
    t = MaskFormerDecoderLayer(depth=depth, **tcfg).eval()
    k = KerasMaskFormerDecoderLayer(depth=depth, **kcfg).eval()
    for part, fn in (("q_ca", port_attention), ("q_sa", port_attention), ("q_dense", port_dense), ("kv_ca", port_attention), ("kv_dense", port_dense)):
        port_residual(getattr(t, part), getattr(k, part), fn)
    g = torch.Generator().manual_seed(7)
    q = torch.randn(2, 150, DIM, generator=g); kv = torch.randn(2, 40, DIM, generator=g)
    am = torch.rand(2, 150, 40, generator=g) > 0.5; am[..., 0] = True
    km = torch.ones(2, 40, dtype=torch.bool); km[:, 35:] = False
    with torch.no_grad():
        qt, kvt = t(q, kv, attn_mask=am, kv_mask=km)
        qk, kvk = k(q, kv, attn_mask=am, kv_mask=km)
    return (qt - qk).abs().max().item(), ((kvt - kvk)[km]).abs().max().item()

def enc_pair(tcfg, kcfg, depth):
    torch.manual_seed(depth)
    t = EncoderLayer(depth=depth, **tcfg).eval()
    k = KerasEncoderLayer(depth=depth, **kcfg).eval()
    port_encoder_layer(t, k)
    x = torch.randn(2, 40, DIM, generator=torch.Generator().manual_seed(3))
    km = torch.ones(2, 40, dtype=torch.bool); km[:, 35:] = False
    with torch.no_grad():
        yt = t(x, kv_mask=km); yk = k(x, kv_mask=km)
    return ((yt - yk)[km]).abs().max().item()

ENC8 = {"dim": DIM, "norm": "FastLayerNorm", "hybrid_norm": True, "dense_norm_placement": "legacy", "attn_kwargs": {"num_heads": 16}, "dense_kwargs": {"activation": "SwiGLU"}}
ENC_OLD = {k: v for k, v in ENC8.items() if k != "dense_norm_placement"}
def safe(f, *a):
    try:
        return f(*a)
    except AssertionError as e:
        return f"port refused: {e}"

for d in (0, 1):
    print(f"decoder depth {d}: matched PR8 cfg  max|dq|,max|dkv| =", dec_pair(PR8, PR8, d))
    print(f"decoder depth {d}: keras w/o PR8 options (discrimination) =", safe(dec_pair, PR8, OLD, d))
    print(f"encoder depth {d}: matched PR8 cfg  max|dy| =", enc_pair(ENC8, ENC8, d))
    print(f"encoder depth {d}: keras w/o legacy placement (discrimination) =", safe(enc_pair, ENC8, ENC_OLD, d))
for d in (0, 1):
    print(f"decoder depth {d}: keras order=ca_dense_sa (discrimination) =", dec_pair(PR8, {**PR8, "query_update_order": "ca_dense_sa"}, d))
    print(f"decoder depth {d}: keras self_attn_norm_kv=False (discrimination) =", dec_pair(PR8, {**PR8, "self_attn_norm_kv": False}, d))
