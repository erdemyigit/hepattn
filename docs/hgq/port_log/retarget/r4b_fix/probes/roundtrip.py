import sys, pathlib, tempfile, torch
import hepattn.keras; hepattn.keras.set_keras_default_device("cpu")
sys.path.insert(0, sys.argv[1] + "/tests/keras")
import test_linformer_state_dict as T
from hepattn.keras.porting import port_maskformer
gen = T.U.generate_polaris_config(pathlib.Path(tempfile.mkdtemp()) / "g.yaml")
org = T.U.torch_reference_config(T.U.merge_org_configs()); torch.manual_seed(0); tm = T.U.instantiate(org)
inputs, _ = T.U.clic_dummy_batch(2, seed=0)
with torch.no_grad(): ref = T.flatten_outputs(tm(inputs))
m = T.fresh(gen, 1); port_maskformer(tm, m); st = m.materialize(inputs).state_dict()
tg = T.fresh(gen, 2).materialize(inputs); c = T.max_err(tg, inputs, ref); tg.load_state_dict(st)
print("roundtrip err", T.max_err(tg, inputs, ref), "control", c, "core keys", len(T.core_keys(st)), "total keys", len(st))
