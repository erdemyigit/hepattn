"""Regenerate docs/hgq/PARITY.md from the measured parity ledger.

Usage (from the repo root):
    rm -f tests/keras/.parity_ledger.jsonl
    PYTHONPATH=src uv run pytest tests/keras -q
    PYTHONPATH=src uv run python tests/keras/gen_parity_report.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "tests/keras/.parity_ledger.jsonl"
OUT = REPO / "docs/hgq/PARITY.md"

HEADER = """# PyTorch → Keras numerical parity report

This report is GENERATED from measurements recorded while running `tests/keras`
(see `tests/keras/gen_parity_report.py`); do not edit the tables by hand.

## Methodology

Every Keras component is weight-ported from its torch twin and compared elementwise
in fp32 against the torch reference (attention backend `torch`, i.e.
`scaled_dot_product_attention` semantics — the portable reference obtained via
`change_attn_backends(model, "torch")`). The acceptance criterion is
`|keras - torch| <= atol + rtol * |torch|` evaluated on valid (non-padded) slots;
each row below reports the measured maxima. Boolean outputs (thresholded predictions
and the per-decoder-layer mask-attention masks) are asserted to be EXACTLY equal —
for the self-referential mask-attention graph this is a stronger and more meaningful
criterion than any float tolerance.

## Known, deliberate numerical semantics

- **Bitwise equality across frameworks is not achievable**: a single keras `Dense`
  (matmul+add) vs torch `Linear` (fused addmm) already differs by up to ~3.6e-07 abs
  (measured in `tests/keras/test_env_smoke.py`). All further deviations compound from
  this kernel-level floor.
- **Fully-masked attention rows produce zeros**, matching the torch fused SDPA
  kernels the reference runs on. A textbook softmax would yield NaN there and poison
  valid outputs downstream through `0 * NaN` in the values contraction (padded
  decoder keys in the bidirectional cross-attention hit exactly this case).
- **Large max-relative errors on near-zero elements are expected** (the rel column is
  dominated by elements where the reference is ~0); the combined atol+rtol criterion
  is the pass/fail bar, and the abs column is the physically meaningful one there.
- **`mask_bce` can be `inf` on randomly-initialized models** (the mask task pads
  invalid-key logits to `finfo.min`, which saturates BCE): the torch reference itself
  is `inf` and the keras twin reproduces it exactly. Trained models do not sit in
  this regime.
- Regression outputs in `mode=scale` are exponentially scaled, so their max-abs error
  scales with the output magnitude; the relative error (~1e-5, i.e. fp32 precision)
  is the meaningful metric.

## Measured float parity (torch fp32 reference vs weight-ported float Keras twin)

| Component / tag | Configuration | Max abs err | Max rel err | atol | rtol |
|---|---|---|---|---|---|
"""

FOOTER = """
## Quantized deltas (HGQ2 mode vs float twin)

Populated in Stage 4 (quantized mode): high-bitwidth configs are asserted close to
the float twin while aggressively low-bitwidth configs are asserted to differ, so the
quantizers are demonstrably active and demonstrably faithful.
"""


def main() -> None:
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    # keep the worst (max abs err) record per (tag, config)
    worst: dict[tuple[str, str], dict] = {}
    for rec in records:
        key = (rec["tag"], rec["config"])
        if key not in worst or rec["max_abs_err"] > worst[key]["max_abs_err"]:
            worst[key] = rec

    rows = []
    for (tag, config), rec in sorted(worst.items()):
        rows.append(f"| `{tag}` | `{config}` | {rec['max_abs_err']:.2e} | {rec['max_rel_err']:.2e} | {rec['atol']:.0e} | {rec['rtol']:.0e} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEADER + "\n".join(rows) + "\n" + FOOTER)
    print(f"wrote {OUT} with {len(rows)} rows from {len(records)} measurements")


if __name__ == "__main__":
    main()
