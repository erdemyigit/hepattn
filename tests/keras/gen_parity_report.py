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

The rows tagged `quantized.delta.*` above compare the SAME weights run through the
float twin and the HGQ2-quantized twin on layer-0 mask logits (computed before any
mask-attention feedback, so deltas are not amplified by threshold flips). The paired
criterion is what makes this non-vacuous: the high-bitwidth config must stay close to
float (fails if the config scopes silently did not apply) while the low-bitwidth
config must differ by more than 10x the high-bitwidth delta (fails if the quantizers
are inert under the torch backend).

Measured: high-bitwidth (20-bit weights, 24-bit datalane, 18-bit softmax tables)
deviates ~0.30 abs on logits of scale ~15 (~2%) — dominated by the quantized-softmax
exp/inv lookup tables accumulated across the attention pipeline; the `table`
quantizer place bounds softmax accuracy INDEPENDENTLY of weight/datalane bitwidths
(without configuring it the delta is ~2.7 regardless of the other bitwidths).
Low-bitwidth (4-bit) deviates ~31 — a factor ~100 separation.
"""


def main() -> None:
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    # keep the worst (max abs err) record per (tag, config)
    worst: dict[tuple[str, str], dict] = {}
    for rec in records:
        key = (rec["tag"], rec["config"])
        if key not in worst or rec["max_abs_err"] > worst[key]["max_abs_err"]:
            worst[key] = rec

    def fmt(x: float) -> str:
        return "—" if x != x else f"{x:.2e}"  # noqa: PLR0124  (NaN check)

    rows = []
    for (tag, config), rec in sorted(worst.items()):
        rows.append(f"| `{tag}` | `{config}` | {fmt(rec['max_abs_err'])} | {fmt(rec['max_rel_err'])} | {fmt(rec['atol'])} | {fmt(rec['rtol'])} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEADER + "\n".join(rows) + "\n" + FOOTER)
    print(f"wrote {OUT} with {len(rows)} rows from {len(records)} measurements")


if __name__ == "__main__":
    main()
