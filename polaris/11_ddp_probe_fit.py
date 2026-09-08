"""Name the CPU tensor that DDP refuses to broadcast, from inside the real fit.

10_ddp_device_probe showed a single-process model is clean: all 2027 parameters and
buffers land on cuda after .to(), and the 500 keras _ebops/_beta Variables that stay on
cpu are plain attributes DDP never looks at. Yet the real DDP run dies inside
DistributedDataParallel.__init__ -> _sync_module_states -> _broadcast_coalesced with
"No backend type associated with device type cpu", which only happens if a tensor in
module.parameters() + module.buffers() is on cpu at wrap time.

So stop reasoning and look: patch DDP.__init__ to enumerate the module exactly as
_sync_module_states does, print every non-cuda tensor BY NAME, then delegate. Run it
instead of main_hgq and the traceback becomes a list of names.

    PYTHONPATH=src python polaris/11_ddp_probe_fit.py fit --config polaris_hgq.yaml \
        --trainer.devices 2 --trainer.strategy ddp ...
"""

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import torch.nn.parallel.distributed as ddp_mod

_orig_init = ddp_mod.DistributedDataParallel.__init__


def _probing_init(self, module, *args, **kwargs):
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "?"))
    named = list(module.named_parameters()) + list(module.named_buffers())
    devs: dict[str, int] = {}
    bad: list[tuple[str, str]] = []
    for name, tensor in named:
        d = str(tensor.device)
        devs[d] = devs.get(d, 0) + 1
        if tensor.device.type != "cuda":
            bad.append((name, d))
    print(f"[DDP-PROBE rank={rank}] {len(named)} params+buffers, devices={devs}", flush=True)
    print(f"[DDP-PROBE rank={rank}] non-cuda: {len(bad)}", flush=True)
    for name, d in bad[:25]:
        print(f"[DDP-PROBE rank={rank}]   {name} -> {d}", flush=True)
    if len(bad) > 25:
        print(f"[DDP-PROBE rank={rank}]   ... and {len(bad) - 25} more", flush=True)
    return _orig_init(self, module, *args, **kwargs)


ddp_mod.DistributedDataParallel.__init__ = _probing_init
print(f"[DDP-PROBE] patched DDP.__init__ (torch {torch.__version__})", flush=True)

from hepattn.experiments.clic.main_hgq import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
