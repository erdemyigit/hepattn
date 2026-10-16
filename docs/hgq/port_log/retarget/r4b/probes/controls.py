import sys, pathlib
W = pathlib.Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
sys.path.insert(0, str(W / "tests/keras"))
import conftest, torch, numpy as np
import test_linformer_parity as T
import masked_linformer as ml, torch.nn.functional as F
t = T.t
core, (q, k, v, pk, pv), _, am, _ = T.attend_case("q_ca")
with torch.no_grad():
    out = core(t(q), t(k), t(v), t(am)); nr = ml.attend(t(q), t(k), t(v), t(pk), t(pv), t(am), renorm=False)
print("renorm attend", float((out - nr).nan_to_num().abs().max()), "nan in nr", bool(torch.isnan(nr).any()))
out, (q, k, v, pk, pv, m) = T.project_case("enc_short")
nr = F.scaled_dot_product_attention(q, ml.project(k, pk, m, renorm=False), ml.project(v, pv, m, renorm=False))
print("renorm project", float((out - nr).abs().max()))
for shape in ("q_ca", "kv_ca"):
    tattn = T.torch_attention(shape, 25); x = T.make_inputs(shape, 26, sort=shape == "q_ca"); kattn = T.hybrid_attention(shape, tattn, x)
    masks = {n: t(x[n]) for n in ("kv_mask", "attn_mask", "kv_sort_idx")}
    with torch.no_grad():
        ref = tattn(t(x["q"]), t(x["kv"]), t(np.random.default_rng(27).standard_normal(x["kv"].shape).astype(np.float32)), **masks)
        same = kattn(t(x["q"]), t(x["kv"]), t(x["kv"]), **masks)
    print("v=k", shape, float((same - ref).abs().max()))
tattn = T.torch_attention("q_ca", 7); x = T.make_inputs("q_ca", 8, sort=True); kattn = T.hybrid_attention("q_ca", tattn, x); ref = T.call(tattn, x)
print("nosort", float((T.call(kattn, {**x, "kv_sort_idx": None}) - ref).abs().max()))
only = {**x, "kv_mask": x["kv_mask"][np.arange(2)[:, None], np.argsort(x["kv_sort_idx"])]}
print("only_kv", float((T.call(kattn, only) - ref).abs().max()))
x = T.make_inputs("enc", 11)
t0 = T.torch_attention("enc", 9, value_residual=True, is_first_layer=True); t1 = T.torch_attention("enc", 10, value_residual=True, is_first_layer=False)
k0 = T.hybrid_attention("enc", t0, x, name="a0", value_residual=True, is_first_layer=True); k1 = T.hybrid_attention("enc", t1, x, name="a1", value_residual=True, is_first_layer=False)
iv = {}; mask = t(x["kv_mask"])
with torch.no_grad():
    h = k0(t(x["q"]), kv_mask=mask, initial_values=iv); o = k1(h, kv_mask=mask, initial_values=iv); nm = k1(h, kv_mask=mask, initial_values=None)
print("nomix", float((nm - o).abs().max()))
for shape in ("enc_short", "enc"):
    tattn = T.torch_attention(shape, 14); x = T.make_inputs(shape, 15)
    one = {**x, "q": x["q"][:1], "kv_mask": np.ones((1, T.SHAPES[shape][1]), bool)}; one["kv"] = one["q"]
    kattn = T.hybrid_attention(shape, tattn, one)
    print("b1 gap", shape, float((T.call(kattn, one) - T.call(kattn, {**one, "kv_mask": None})).abs().max()))
tattn = T.torch_attention("q_ca", 0, qkv_norm=True, norm="LayerNorm")
with torch.no_grad(): tattn.in_proj_bias.zero_(); tattn.out_proj.bias.zero_()
x = T.make_inputs("q_ca", 22, sort=True); ref = T.call(tattn, x)
hi = T.call(T.hybrid_attention("q_ca", tattn, x, spec=T.HIGH_SPEC, qkv_norm=True, norm="LayerNorm"), x)
co = T.call(T.hybrid_attention("q_ca", tattn, x, spec=T.COARSE_SPEC, qkv_norm=True, norm="LayerNorm"), x)
print("paired", float((hi - ref).abs().max()), float((co - ref).abs().max()), float(co.std(dim=1).mean()))
