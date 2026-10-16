import sys, time, pathlib
W = pathlib.Path("/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax")
sys.path.insert(0, str(W / "tests/keras"))
import conftest  # noqa: pins cpu
import torch
torch.compiler.set_stance("force_eager")
import org_model_utils as U
import test_org_model_parity as P
import hepattn; assert str(W) in hepattn.__file__, hepattn.__file__
gen = U.generate_polaris_config(pathlib.Path(sys.argv[1]))
org = U.torch_reference_config(U.merge_org_configs())
seeds = [int(s) for s in sys.argv[2].split(",")]
for seed in seeds:
    t0 = time.time()
    res = P.run_pair(org, gen, seed)
    o = max(res["outs"], key=res["outs"].get); l = max(res["losses"], key=res["losses"].get)
    print(f"seed {seed}: bools {all(res['bools'].values())} ({len(res['bools'])}) outs {res['outs'][o]:.3e} ({o}) losses {res['losses'][l]:.3e} ({l}) {time.time()-t0:.1f}s", flush=True)
