"""polaris/24_ws_eval.py summaries on hand-made inputs: jet-E response IQR / median per truth-E bin, particle efficiency and
fake rate with the ``plot_helper_particle.plot_eff_fr_purity`` definitions.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("ws24", REPO / "polaris/24_ws_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jet_summary_quantiles_per_bin(ev):
    # bin 20-40 GeV: response -0.1 .. 0.1 uniform (IQR 0.1, median 0); bin 100-120: shifted by +0.5, twice as wide
    rel_a = np.linspace(-0.1, 0.1, 2001)
    rel_b = 0.5 + np.linspace(-0.2, 0.2, 1001)
    ref_e = np.concatenate([np.full(rel_a.size, 30.0), np.full(rel_b.size, 110.0)])
    rel = np.concatenate([rel_a, rel_b])
    out = ev.jet_summary(ref_e, rel, boot=20)
    q25, q50, q75 = np.percentile(rel, [25, 50, 75])
    assert out["iqr"] == pytest.approx(q75 - q25, abs=1e-12) and out["median"] == pytest.approx(q50, abs=1e-12)
    bins = {(b["e_lo"], b["e_hi"]): b for b in out["per_bin"]}
    assert bins[20, 40]["n"] == 2001 and bins[100, 120]["n"] == 1001 and bins[0, 20]["n"] == 0
    assert bins[20, 40]["iqr"] == pytest.approx(0.1, abs=1e-12) and bins[20, 40]["median"] == pytest.approx(0.0, abs=1e-12)
    assert bins[100, 120]["iqr"] == pytest.approx(0.2, abs=1e-12) and bins[100, 120]["median"] == pytest.approx(0.5, abs=1e-12)
    assert np.isnan(bins[0, 20]["iqr"])
    assert out["n_jets"] == 3002 and 0 < out["iqr_sigma"] < 0.05
    assert ev.jet_summary([], [], boot=5) == {"n_jets": 0}


def test_particle_rates_follow_the_plot_definitions(ev):
    # truth: 6 photons matched, 2 unmatched; reco: 6 matched photons + 3 unmatched photons + 1 unmatched neutral hadron
    ref = {"pt": np.full(6, 5.0), "class": np.full(6, 4)}
    comp = {"pt": np.full(6, 5.0), "class": np.full(6, 4)}
    ref_un = {"pt": np.full(2, 5.0), "class": np.full(2, 4)}
    comp_un = {"pt": np.full(4, 5.0), "class": np.array([4, 4, 4, 3])}
    out = ev.particle_summary((ref, comp, ref_un, comp_un))
    assert out["photon"]["eff"]["value"] == pytest.approx(6 / 8)
    assert out["photon"]["fake"]["value"] == pytest.approx(3 / 9)
    assert out["neut had"]["fake"]["value"] == pytest.approx(1.0) and out["neut had"]["eff"]["n"] == 0
    assert out["all"]["eff"]["value"] == pytest.approx(6 / 8) and out["all"]["fake"]["value"] == pytest.approx(4 / 10)
    photon_bins = out["photon per pt"]
    hit = [b for b in photon_bins if b["pt_lo"] < 5.0 <= b["pt_hi"]]
    assert len(hit) == 1 and hit[0]["eff"]["value"] == pytest.approx(6 / 8) and hit[0]["fake"]["value"] == pytest.approx(3 / 9)
    assert sum(b["eff"]["n"] for b in photon_bins) == 8
