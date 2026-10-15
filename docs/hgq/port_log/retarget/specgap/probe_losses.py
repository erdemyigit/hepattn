"""Native keras_native losses/costs vs org/main hepattn.models.loss on identical tensors (B=3 events with different padding)."""
import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("org_loss", sys.argv[1]); OL = importlib.util.module_from_spec(spec); spec.loader.exec_module(OL)
from hepattn.keras_native import losses as NL
rng = np.random.default_rng(0)
B, P, N = 3, 150, 160
nv = np.array([120, 37, 159]); npart = np.array([40, 9, 60])
pad = np.arange(N)[None] < nv[:, None]
ov = np.arange(P)[None] < npart[:, None]
logits = rng.normal(size=(B, P, N)).astype(np.float32) * 3
logits = np.where(pad[:, None], logits, np.finfo(np.float32).min)  # as ObjectHitMaskTask.forward
tgt = ((rng.random((B, P, N)) < 0.05) & pad[:, None] & ov[..., None]).astype(np.float32)
t = lambda a: torch.from_numpy(np.asarray(a))
org_bce = float(OL.mask_bce_loss(t(logits), t(tgt), object_valid_mask=t(ov), input_pad_mask=t(pad), sample_weight=t(np.ones_like(tgt))))
nat_bce = float(NL.mask_bce_loss(logits, tgt, ov, pad, sample_weight=np.ones_like(tgt)))
org_dice = float(OL.mask_dice_loss(t(logits), t(tgt), object_valid_mask=t(ov), input_pad_mask=t(pad)))
nat_dice = float(NL.mask_dice_loss(logits, tgt, ov, pad))
print(f"mask_bce  org/main={org_bce:.6g} native={nat_bce:.6g}")
print(f"mask_dice org/main={org_dice:.6g} native={nat_dice:.6g}")
# per-event reference written from the a4263c1 formula (independent of both implementations)
sig = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -700, 700)))
bce = np.maximum(logits, 0) * 0 + (1 - tgt) * logits.astype(np.float64) + np.logaddexp(0, -logits.astype(np.float64))
per = (bce * pad[:, None]).sum(-1) / np.maximum(pad.sum(-1), 1)[:, None]
print(f"mask_bce  numpy per-event ref={(per * ov).sum() / ov.sum():.6g}")
cl = rng.normal(size=(B, P, 6)).astype(np.float32); tc = np.where(ov, rng.integers(0, 5, (B, P)), 5)
org_cost = OL.object_ce_cost(t(cl), t(tc)).numpy()
nat_cost_probs = np.asarray(NL.object_ce_cost(torch.softmax(t(cl), -1).numpy(), tc))
nat_cost_logits = np.asarray(NL.object_ce_cost(cl, tc))
print(f"object_ce_cost: max|org(logits) - native(probs, as objective.py:131)| = {np.abs(org_cost - nat_cost_probs).max():.4g}; "
      f"max|org - native(logits)| = {np.abs(org_cost - nat_cost_logits).max():.2g}; range org={np.ptp(org_cost):.3f} native(probs)={np.ptp(nat_cost_probs):.3f}")
