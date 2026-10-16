import sys, copy, pathlib
W = pathlib.Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
sys.path.insert(0, str(W / "tests/keras"))
import conftest, torch
torch.compiler.set_stance("force_eager")
import org_model_utils as U
import test_org_model_parity as P
gen = U.generate_polaris_config(pathlib.Path(sys.argv[1]))
org = U.torch_reference_config(U.merge_org_configs())
def show(tag, hyb):
    res = P.run_pair(org, hyb, 0)
    print(tag, f"outs {max(res['outs'].values()):.3e} losses {max(res['losses'].values()):.3e} bools {sum(res['bools'].values())}/{len(res['bools'])}", flush=True)
h = copy.deepcopy(gen); h["model"]["model"]["init_args"].pop("input_sort_field"); show("no_sort", h)
for key, value in [("query_update_order", "ca_dense_sa"), ("self_attn_norm_kv", False)]:
    h = copy.deepcopy(gen); h["model"]["model"]["init_args"]["decoder"]["decoder_layer_config"][key] = value; show(key, h)
