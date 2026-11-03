"""Markdown tables from a measure_sectors.py json (usage: python tables.py result.json)."""

import json
import sys
from pathlib import Path


def pct(x):
    return f"{100 * x:.2f}"


def main():
    d = json.loads(Path(sys.argv[1]).read_text())
    print(f"events {d['n_events']}; nodes/event {d['nodes_per_event']}; valid targets/event {d['valid_targets_per_event']}")
    print(f"valid targets without any node: {pct(d['invisible_valid_frac'])} %\n")

    print("### Padding: hits and (a)-targets per sector (per-event maximum over sectors)\n")
    print(
        "| track phi | S | edges | max hits q50 / q99 / q99.9 / max | max targets q99 / q99.9 / max | "
        "events with a sector > Q_s targets (%) | runs in vtx sort (mean) |"
    )
    print("|---|---|---|---|---|---|---|")
    for key, r in d.items():
        if "/" not in key:
            continue
        v, s, sch = key.split("/")
        h, t = r["max_hits_per_sector"], r["max_targets_a_per_sector"]
        print(
            f"| {v} | {s[1:]} | {sch} | {h['q50']:.0f} / {h['q99']:.0f} / {h['q99.9']:.0f} / {h['max']:.0f} | "
            f"{t['q99']:.0f} / {t['q99.9']:.0f} / {t['max']:.0f} | {pct(r['frac_events_targets_a_over_q'])} (Q_s={r['queries_per_sector']}) | "
            f"{r['runs_in_vtx_sort_mean']:.2f} |"
        )

    print("\n### Boundary splits, option (a) majority sector (% of valid targets, or of their total energy)\n")
    print("| track phi | S | edges | split (any) | >10 % outside | >30 % outside | energy outside | mask nodes in >1 sector |")
    print("|---|---|---|---|---|---|---|---|")
    for key, r in d.items():
        if "/" not in key or key.endswith("S1/fixed"):
            continue
        v, s, sch = key.split("/")
        print(
            f"| {v} | {s[1:]} | {sch} | {pct(r['a_frac_particles_split_any'])} | {pct(r['a_frac_particles_out_gt_0.1'])} | "
            f"{pct(r['a_frac_particles_out_gt_0.3'])} | {pct(r['a_energy_out_frac'])} | {pct(r['mask_split_frac'])} |"
        )

    print("\n### Option (b): target in every sector with share >= x; option (c): overlap band w (rad) each side\n")
    print(
        "| track phi | S | edges | (b) x=0.1 dup % / E uncov % / max tgt / ev>Q % | x=0.2 | x=0.3 | (c) w=0.1 E uncov % / max hits | w=0.2 | w=0.4 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for key, r in d.items():
        if "/" not in key or key.endswith("S1/fixed"):
            continue
        v, s, sch = key.split("/")
        b = [
            f"{pct(r[f'b{x}_dup_frac'])} / {pct(r[f'b{x}_energy_uncovered_frac'])} / {r[f'b{x}_max_targets_per_sector']['max']:.0f} / "
            f"{pct(r[f'b{x}_frac_events_targets_over_q'])}"
            for x in (0.1, 0.2, 0.3)
        ]
        c = [f"{pct(r[f'c{w}_energy_uncovered_frac'])} / {r[f'c{w}_max_hits_per_sector']['max']:.0f}" for w in (0.1, 0.2, 0.4)]
        print(f"| {v} | {s[1:]} | {sch} | " + " | ".join(b + c) + " |")


if __name__ == "__main__":
    main()
