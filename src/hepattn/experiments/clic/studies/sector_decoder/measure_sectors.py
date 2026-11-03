"""Per-sector multiplicities and boundary splits of CLIC events, to size a sectored decoder (issue #20).

Selection, target validity and incidence follow CLICDataset (pflow_data.py). A particle's weight on a node is
particle_e for its track(s) and topo2particle_energy for each topocluster it deposits in; its share of a sector is
the weight there over its total weight. Sectors are S cyclic phi intervals given by boundary angles per event:
  fixed  edges at -pi + 2 pi j / S
  gap    equal-width sectors, rotated per event to maximise the smallest distance from an edge to a node (input-only)
  count  equal-count chunks of the phi-sorted nodes, cyclic start chosen to maximise the smallest boundary gap (input-only)
  cap    contiguous chunks of the phi-sorted nodes of at most MCAP[S] nodes each, cut at the gaps that maximise the
         smallest boundary gap (input-only)
Node phi for tracks: vertex phi (`vtx`, the model's sort field) or calorimeter-entry phi_int (`calo`).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from hepattn.experiments.clic.pflow_data import CLICDataset

TWO_PI = 2 * np.pi
BRANCHES = [
    "track_phi", "track_phi_int", "topo_phi", "topo_e", "particle_e", "particle_phi", "particle_pdgid",
    "track_particle_idx", "topo2particle_topo_idx", "topo2particle_particle_idx", "topo2particle_energy",
]  # fmt: skip
SECTORS = (1, 2, 3, 4, 6)
SCHEMES = ("fixed", "gap", "count", "cap")
MCAP = {2: 96, 3: 64, 4: 48, 6: 32}  # about 1.2 x ceil(159 / S)
VARIANTS = ("vtx", "calo")
SHARE_CUTS = (0.1, 0.2, 0.3)
BANDS = (0.1, 0.2, 0.4)
MAX_NODES, NUM_OBJECTS, INCIDENCE_CUT = 160, 150, 0.01


def wrap(x):
    return np.mod(x, TWO_PI)


def _gap_cuts(gaps, s, mcap):
    """Cut positions (cut i = between sorted nodes i and i+1, cyclic) giving s chunks of <= mcap nodes, maximising the
    smallest cut gap. Binary search on how many of the largest gaps may be cut; greedy farthest jumps cover the cycle.
    """
    n = len(gaps)
    order = np.argsort(-gaps, kind="stable")

    def solve(k):
        allowed = order[:k]
        if n <= mcap:
            return allowed[:s]
        a = np.sort(allowed)
        m = len(a)
        a2 = np.concatenate([a, a + n])
        far = np.searchsorted(a2, a2 + mcap, side="right") - 1
        for i0 in range(m):
            pos, cuts = i0, [a[i0]]
            while a[i0] + n - a2[pos] > mcap:
                nxt = far[pos]
                if nxt == pos or len(cuts) == s:
                    break
                pos = nxt
                cuts.append(a2[pos] % n)
            else:
                rest = [c for c in allowed if c not in set(cuts)][: s - len(cuts)]
                return np.array(cuts + rest)
        return None

    lo, hi = s, n  # solve(n) succeeds whenever mcap >= ceil(n / s)
    while lo < hi:
        mid = (lo + hi) // 2
        if solve(mid) is None:
            lo = mid + 1
        else:
            hi = mid
    return solve(lo)


def boundaries(phi, s, scheme):
    """Return S boundary angles (sector j = [b_j, b_{j+1}) cyclic), increasing modulo 2 pi from b_0."""
    if s == 1:
        return np.array([-np.pi])
    width = TWO_PI / s
    if scheme == "fixed":
        return -np.pi + width * np.arange(s)
    if scheme == "gap":
        offsets = np.linspace(0, width, 181)[:-1]
        edges = offsets[:, None] + width * np.arange(s)[None]  # (n_off, S)
        d = np.abs(np.angle(np.exp(1j * (edges[..., None] - phi[None, None]))))  # (n_off, S, N)
        best = d.min(-1).min(-1).argmax()
        return -np.pi + edges[best]
    p = np.sort(phi)
    if scheme == "cap" and len(p) >= s:
        gaps = wrap(np.roll(p, -1) - p)
        cut = _gap_cuts(gaps, s, MCAP[s])
        b = np.angle(np.exp(1j * (p[cut] + gaps[cut] / 2)))
        return b[np.argsort(wrap(b - b[0]))]
    # equal-count chunks of the sorted nodes; boundary at the middle of the gap between chunks
    n = len(p)
    gaps = wrap(np.roll(p, -1) - p)  # gaps[i] between sorted node i and i+1 (cyclic)
    sizes = np.full(s, n // s) + (np.arange(s) < n % s)
    ends = np.cumsum(sizes) - 1  # last index of each chunk, relative to start
    starts = np.arange(n)
    cut = (starts[:, None] + ends[None]) % n  # (n_start, S): node index before each boundary
    best = gaps[cut].min(1).argmax()
    b = p[cut[best]] + gaps[cut[best]] / 2
    b = np.angle(np.exp(1j * b))
    return b[np.argsort(wrap(b - b[0]))]


def sector_index(phi, b):
    u = wrap(phi - b[0])
    bb = wrap(b - b[0])
    return np.searchsorted(bb, u, side="right") - 1


def in_window(phi, b, w):
    """(S, N) bool: node inside sector j extended by w on each side."""
    s = len(b)
    if s == 1:
        return np.ones((1, len(phi)), dtype=bool)
    width = wrap(np.roll(b, -1) - b)
    return wrap(phi[None] - b[:, None] + w) < (width[:, None] + 2 * w)


def runs_cyclic(sec_sorted):
    """Number of contiguous runs of sector ids along the sorted order, counting the wrap (head==tail) as one run."""
    if len(sec_sorted) == 0:
        return 0
    changes = int((sec_sorted[1:] != sec_sorted[:-1]).sum()) + 1
    if changes > 1 and sec_sorted[0] == sec_sorted[-1]:
        changes -= 1
    return changes


def event_records(ev, class_labels):
    n_tr, n_tp, n_p = len(ev["track_phi"]), len(ev["topo_phi"]), len(ev["particle_e"])
    tpi = np.asarray(ev["track_particle_idx"], dtype=np.int64)
    if not (n_tr + n_tp < MAX_NODES and n_p < NUM_OBJECTS and len(tpi) == n_tr):
        return None
    e_p = np.asarray(ev["particle_e"], dtype=np.float64)
    cls = np.array([class_labels[int(x)] for x in ev["particle_pdgid"]])
    has_track = np.zeros(n_p, bool)
    has_track[tpi] = True
    cls = np.where(~has_track & (cls < 2), cls + 3, cls)
    cls = np.where(~has_track & (cls == 2), 3, cls)
    valid = cls < 5

    n = n_tr + n_tp
    # weights (n_p, n): energy content of particle p on node n
    w = np.zeros((n_p, n))
    w[tpi, np.arange(n_tr)] = e_p[tpi]
    t_idx = np.asarray(ev["topo2particle_topo_idx"], dtype=np.int64)
    p_idx = np.asarray(ev["topo2particle_particle_idx"], dtype=np.int64)
    # assignment, not a sum: CLICDataset.load_event keeps the last of repeated (topo, particle) pairs
    w[p_idx, t_idx + n_tr] = np.asarray(ev["topo2particle_energy"], dtype=np.float64)
    # incidence as in CLICDataset.load_event (tracks 1, topos energy-normalised per node)
    inc = np.zeros((n_p, n))
    inc[tpi, np.arange(n_tr)] = 1.0
    inc[p_idx, t_idx + n_tr] = np.asarray(ev["topo2particle_energy"], dtype=np.float64)
    inc = inc / np.clip(inc.sum(0, keepdims=True), 1e-6, None)
    mask_node = inc > INCIDENCE_CUT

    wv, mv, ev_p, charged = w[valid], mask_node[valid], e_p[valid], has_track[valid]
    wtot = wv.sum(1)
    visible = wtot > 0
    out = {"n_nodes": n, "n_valid": int(valid.sum()), "n_invisible": int((~visible).sum())}
    phi_topo = np.asarray(ev["topo_phi"], dtype=np.float64)
    pphi = np.asarray(ev["particle_phi"], dtype=np.float64)[valid]
    vtx = np.concatenate([np.asarray(ev["track_phi"], dtype=np.float64), phi_topo])
    for variant in VARIANTS:
        phi = vtx if variant == "vtx" else np.concatenate([np.asarray(ev["track_phi_int"], dtype=np.float64), phi_topo])
        order_vtx = np.argsort(vtx)  # the model's sort (padding is absent here)
        for s in SECTORS:
            for scheme in SCHEMES if s > 1 else ("fixed",):
                b = boundaries(phi, s, scheme)
                sec = sector_index(phi, b)
                onehot = np.eye(s, dtype=bool)[sec].T  # (S, n)
                share = (wv @ onehot.T) / np.where(visible, wtot, 1.0)[:, None]  # (P, S)
                psec_truth = sector_index(pphi, b)
                share[~visible] = np.eye(s)[psec_truth[~visible]]
                a = share.argmax(1)
                fa = share[np.arange(len(a)), a]
                rec = {
                    "hits": np.bincount(sec, minlength=s),
                    "targets_a": np.bincount(a, minlength=s),
                    "out_share_a": 1 - fa,
                    "e": ev_p,
                    "charged": charged,
                    "mask_split": (mv[:, None, :] & onehot[None]).any(-1).sum(1) > 1,
                    "runs_in_vtx_sort": runs_cyclic(sec[order_vtx]),
                }
                for x in SHARE_CUTS:
                    tgt = (share >= x) | (np.arange(s)[None] == a[:, None])
                    rec[f"targets_b{x}"] = tgt.sum(0)
                    rec[f"dup_b{x}"] = tgt.sum(1) > 1
                    rec[f"uncov_b{x}"] = (share * ~tgt).sum(1)
                for band in BANDS:
                    win = in_window(phi, b, band)  # (S, n)
                    cov = (wv * win[a]).sum(1) / np.where(visible, wtot, 1.0)
                    cov[~visible] = 1.0
                    rec[f"uncov_c{band}"] = 1 - cov
                    rec[f"hits_c{band}"] = win.sum(1)
                out[variant, s, scheme] = rec
    return out


def process(args):
    path, start, stop = args
    d = CLICDataset.__new__(CLICDataset)
    d.init_label_dicts()
    arrays = uproot.open(path)["EventTree"].arrays(BRANCHES, entry_start=start, entry_stop=stop, library="ak")
    recs = []
    for i in range(len(arrays)):
        ev = {k: ak.to_numpy(arrays[k][i]) for k in BRANCHES}
        r = event_records(ev, d.class_labels)
        if r is not None:
            recs.append(r)
    return recs


def q(x, qs=(0.5, 0.9, 0.99, 0.999)):
    x = np.asarray(x)
    return {f"q{int(1000 * k) / 10:g}": float(np.quantile(x, k)) for k in qs} | {"max": float(x.max()), "mean": float(x.mean())}


def summarise(recs):
    out = {
        "n_events": len(recs),
        "nodes_per_event": q([r["n_nodes"] for r in recs]),
        "valid_targets_per_event": q([r["n_valid"] for r in recs]),
        "invisible_valid_frac": sum(r["n_invisible"] for r in recs) / sum(r["n_valid"] for r in recs),
    }
    for variant in VARIANTS:
        for s in SECTORS:
            qs_per_sector = NUM_OBJECTS // s
            for scheme in SCHEMES if s > 1 else ("fixed",):
                rs = [r[variant, s, scheme] for r in recs]
                e = np.concatenate([r["e"] for r in rs])
                oa = np.concatenate([r["out_share_a"] for r in rs])
                ch = np.concatenate([r["charged"] for r in rs])
                maxhit = np.array([r["hits"].max() for r in rs])
                maxtgt = np.array([r["targets_a"].max() for r in rs])
                res = {
                    "queries_per_sector": qs_per_sector,
                    "hits_per_sector_all": q(np.concatenate([r["hits"] for r in rs])),
                    "max_hits_per_sector": q(maxhit),
                    "targets_a_per_sector_all": q(np.concatenate([r["targets_a"] for r in rs])),
                    "max_targets_a_per_sector": q(maxtgt),
                    "frac_events_targets_a_over_q": float((maxtgt > qs_per_sector).mean()),
                    "a_frac_particles_split_any": float((oa > 1e-9).mean()),
                    "a_frac_particles_out_gt_0.1": float((oa > 0.1).mean()),
                    "a_frac_particles_out_gt_0.3": float((oa > 0.3).mean()),
                    "a_energy_out_frac": float((e * oa).sum() / e.sum()),
                    "a_energy_frac_of_split_particles": float(e[oa > 1e-9].sum() / e.sum()),
                    "mask_split_frac": float(np.concatenate([r["mask_split"] for r in rs]).mean()),
                    **{
                        f"a_{name}_{stat}": val
                        for name, sel in (("charged", ch), ("neutral", ~ch))
                        for stat, val in (
                            ("frac", float(sel.mean())),
                            ("out_gt_0.1", float((oa[sel] > 0.1).mean())),
                            ("energy_out_frac", float((e[sel] * oa[sel]).sum() / e[sel].sum())),
                        )
                    },
                    "runs_in_vtx_sort_mean": float(np.mean([r["runs_in_vtx_sort"] for r in rs])),
                    "frac_events_one_run_per_sector": float(np.mean([r["runs_in_vtx_sort"] <= s for r in rs])),
                }
                for x in SHARE_CUTS:
                    kt, ku, kd = f"targets_b{x}", f"uncov_b{x}", f"dup_b{x}"  # hoisted: formatting per record is slow
                    mt = np.array([r[kt].max() for r in rs])
                    unc = np.concatenate([r[ku] for r in rs])
                    res[f"b{x}_dup_frac"] = float(np.concatenate([r[kd] for r in rs]).mean())
                    res[f"b{x}_energy_uncovered_frac"] = float((e * unc).sum() / e.sum())
                    res[f"b{x}_max_targets_per_sector"] = q(mt)
                    res[f"b{x}_frac_events_targets_over_q"] = float((mt > qs_per_sector).mean())
                for band in BANDS:
                    ku, kh = f"uncov_c{band}", f"hits_c{band}"
                    unc = np.concatenate([r[ku] for r in rs])
                    res[f"c{band}_energy_uncovered_frac"] = float((e * unc).sum() / e.sum())
                    res[f"c{band}_frac_particles_not_covered"] = float((unc > 1e-9).mean())
                    res[f"c{band}_max_hits_per_sector"] = q(np.array([r[kh].max() for r in rs]))
                out[f"{variant}/S{s}/{scheme}"] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--entries", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = uproot.open(a.path)["EventTree"].num_entries
    n = n if a.entries < 0 else min(n, a.entries)
    jobs = [(a.path, i, min(i + a.chunk, n)) for i in range(0, n, a.chunk)]
    with ProcessPoolExecutor(a.workers) as pool:
        recs = [r for part in pool.map(process, jobs) for r in part]
    res = summarise(recs)
    res["source"] = {"path": a.path, "entries_read": n}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}: {len(recs)} events")


if __name__ == "__main__":
    main()
