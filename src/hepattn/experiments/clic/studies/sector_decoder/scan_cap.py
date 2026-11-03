"""Scan the per-sector node cap of the `cap` partition (calorimeter phi for tracks), option (a) targets.

For each cap: split fractions as in measure_sectors.py, and the targets a sector holds beyond its query count
(queries_per_sector), dropped lowest-energy first: the fraction of targets and of target energy lost that way.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import measure_sectors as ms
import numpy as np
import uproot

from hepattn.experiments.clic.pflow_data import CLICDataset


def event_scan(ev, class_labels, s, caps, q):
    n_tr, n_tp, n_p = len(ev["track_phi"]), len(ev["topo_phi"]), len(ev["particle_e"])
    tpi = np.asarray(ev["track_particle_idx"], dtype=np.int64)
    if not (n_tr + n_tp < ms.MAX_NODES and n_p < ms.NUM_OBJECTS and len(tpi) == n_tr):
        return None
    e_p = np.asarray(ev["particle_e"], dtype=np.float64)
    cls = np.array([class_labels[int(x)] for x in ev["particle_pdgid"]])
    has_track = np.zeros(n_p, bool)
    has_track[tpi] = True
    cls = np.where(~has_track & (cls < 2), cls + 3, cls)
    cls = np.where(~has_track & (cls == 2), 3, cls)
    valid = cls < 5
    n = n_tr + n_tp
    w = np.zeros((n_p, n))
    w[tpi, np.arange(n_tr)] = e_p[tpi]
    t_idx = np.asarray(ev["topo2particle_topo_idx"], dtype=np.int64)
    p_idx = np.asarray(ev["topo2particle_particle_idx"], dtype=np.int64)
    w[p_idx, t_idx + n_tr] = np.asarray(ev["topo2particle_energy"], dtype=np.float64)
    wv, ev_p, charged = w[valid], e_p[valid], has_track[valid]
    wtot = wv.sum(1)
    phi = np.concatenate([np.asarray(ev["track_phi_int"], dtype=np.float64), np.asarray(ev["topo_phi"], dtype=np.float64)])
    out = {"e": ev_p, "charged": charged}
    for cap in caps:
        ms.MCAP[s] = cap
        b = ms.boundaries(phi, s, "cap")
        sec = ms.sector_index(phi, b)
        share = (wv @ np.eye(s)[sec]) / np.where(wtot > 0, wtot, 1.0)[:, None]
        a = share.argmax(1)
        oa = 1 - share[np.arange(len(a)), a]
        lost = np.zeros(len(a), bool)
        for j in range(s):
            members = np.flatnonzero(a == j)
            if len(members) > q:
                lost[members[np.argsort(ev_p[members])[: len(members) - q]]] = True
        out[cap] = {"oa": oa, "lost": lost, "max_hits": int(np.bincount(sec, minlength=s).max()), "overflow": bool(lost.any())}
    return out


def process(args):
    path, start, stop, s, caps, q = args
    d = CLICDataset.__new__(CLICDataset)
    d.init_label_dicts()
    arrays = uproot.open(path)["EventTree"].arrays(ms.BRANCHES, entry_start=start, entry_stop=stop, library="ak")
    recs = []
    for i in range(len(arrays)):
        r = event_scan({k: ak.to_numpy(arrays[k][i]) for k in ms.BRANCHES}, d.class_labels, s, caps, q)
        if r is not None:
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--sectors", type=int, default=3)
    ap.add_argument("--caps", type=int, nargs="+", default=[53, 56, 60, 64, 72])
    ap.add_argument("--queries-per-sector", type=int, default=50)
    ap.add_argument("--entries", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = uproot.open(a.path)["EventTree"].num_entries
    n = n if a.entries < 0 else min(n, a.entries)
    jobs = [(a.path, i, min(i + a.chunk, n), a.sectors, a.caps, a.queries_per_sector) for i in range(0, n, a.chunk)]
    with ProcessPoolExecutor(a.workers) as pool:
        recs = [r for part in pool.map(process, jobs) for r in part]
    e = np.concatenate([r["e"] for r in recs])
    ch = np.concatenate([r["charged"] for r in recs])
    res = {"n_events": len(recs), "sectors": a.sectors, "queries_per_sector": a.queries_per_sector}
    for cap in a.caps:
        oa = np.concatenate([r[cap]["oa"] for r in recs])
        lost = np.concatenate([r[cap]["lost"] for r in recs])
        res[str(cap)] = {
            "max_hits": int(max(r[cap]["max_hits"] for r in recs)),
            "frac_events_overflow": float(np.mean([r[cap]["overflow"] for r in recs])),
            "frac_targets_lost": float(lost.mean()),
            "frac_energy_lost": float(e[lost].sum() / e.sum()),
            "split_out_gt_0.1": float((oa > 0.1).mean()),
            "energy_out_frac": float((e * oa).sum() / e.sum()),
            "neutral_energy_out_frac": float((e[~ch] * oa[~ch]).sum() / e[~ch].sum()),
            "charged_out_gt_0.1": float((oa[ch] > 0.1).mean()),
        }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
