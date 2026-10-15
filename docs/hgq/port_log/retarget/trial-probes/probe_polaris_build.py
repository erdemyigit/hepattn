"""Build KerasMaskFormer from a 03_make_config.py output (float, then quantized) and run one forward on dummy CLIC events."""
import importlib, sys, traceback
import torch, yaml
from hepattn.keras import set_keras_default_device
set_keras_default_device("cpu")

def inst(node):
    if isinstance(node, dict) and "class_path" in node:
        mod, _, cls = node["class_path"].rpartition(".")
        c = getattr(importlib.import_module(mod), cls)
        args = {k: inst(v) for k, v in (node.get("init_args") or {}).items()}
        if c is torch.nn.ModuleList:
            return torch.nn.ModuleList(args["modules"])
        return c(**args)
    if isinstance(node, list):
        return [inst(v) for v in node]
    if isinstance(node, dict):
        return {k: inst(v) for k, v in node.items()}
    if isinstance(node, str) and node.startswith("torch.nn."):
        return getattr(torch.nn, node.split(".")[-1])()
    return node

cfg = yaml.safe_load(open(sys.argv[1]))
ia = cfg["model"]["model"]["init_args"]
from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.experiments.clic.pflow_data import CLICDataset
d = cfg["data"]
for quant in (None, ia["quant"]):
    tag = "float" if quant is None else "quant"
    try:
        kw = {k: (v if k in ("encoder", "decoder") else inst(v)) for k, v in ia.items() if k != "quant"}
        m = KerasMaskFormer(quant=quant, **kw).eval()
        ds = CLICDataset(filepath="dummy", inputs=d["inputs"], targets=d["targets"], scale_dict_path=d["scale_dict_path"],
                         num_events=2, num_objects=d.get("num_objects", 150), max_nodes=d.get("max_nodes", 160), dummy_data=True)
        ev = [ds[i] for i in range(2)]
        x = {k: torch.stack([e[0][k] for e in ev]) for k in ev[0][0]}
        with torch.no_grad():
            out = m(x)
        print(tag, "OK; encoder attn", type(m.encoder.layers[0].attn.fn).__name__, "decoder order", getattr(m.decoder.decoder_layers[0], "query_update_order", "n/a"), "pflow keys", sorted(out["final"]) if "final" in out else list(out)[:3])
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(tag, "FAIL", type(e).__name__, str(e)[:200], f"@ {tb.filename.split('src/')[-1]}:{tb.lineno}")
