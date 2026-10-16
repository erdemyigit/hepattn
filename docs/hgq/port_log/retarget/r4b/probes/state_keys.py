import sys, pathlib
W = pathlib.Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
sys.path.insert(0, str(W / "tests/keras"))
import conftest, torch
torch.compiler.set_stance("force_eager")
import org_model_utils as U
q = sys.argv[2] == "q"
gen = U.generate_polaris_config(pathlib.Path(sys.argv[1]))
m = U.instantiate(gen, quantized=q)
inputs, _ = U.clic_dummy_batch()
with torch.no_grad(): m(inputs)
m.register_keras_parameters()
sd = m.state_dict()
for k, v in sd.items():
    if "decoder_layers.0.q_sa" in k or "decoder_layers.0.kv_ca" in k:
        if "proj" in k or "dead" in k or "tail" in k or "attend" in k or "project" in k:
            print(k, tuple(v.shape), v.data_ptr())
