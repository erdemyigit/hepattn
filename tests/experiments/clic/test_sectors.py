import numpy as np
import pytest
import torch

from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.experiments.clic.pflow_data import CLICDataset
from hepattn.experiments.clic.sectors import partition, wrap


def random_events(seed: int, num: int, max_n: int = 159):
    rng = np.random.default_rng(seed)
    for _ in range(num):
        yield rng.uniform(-np.pi, np.pi, rng.integers(8, max_n + 1))


def cut_gaps(phi: np.ndarray, sector: np.ndarray) -> np.ndarray:
    """Phi gaps between neighbouring sorted nodes that sit in different sectors."""
    order = np.argsort(phi)
    p, s = phi[order], sector[order]
    gaps = wrap(np.roll(p, -1) - p)
    return gaps[s != np.roll(s, -1)]


def equal_count_min_gap(phi: np.ndarray, num_sectors: int) -> float:
    """Smallest cut gap of the best equal-count partition (sizes differ by at most one, best cyclic start)."""
    p = np.sort(phi)
    n = len(p)
    gaps = wrap(np.roll(p, -1) - p)
    sizes = np.full(num_sectors, n // num_sectors) + (np.arange(num_sectors) < n % num_sectors)
    ends = np.cumsum(sizes) - 1
    return max(gaps[(start + ends) % n].min() for start in range(n))


@pytest.mark.parametrize("num_sectors", [2, 3, 4, 6])
def test_sizes_and_each_node_once(num_sectors):
    rng = np.random.default_rng(num_sectors)
    for phi in random_events(num_sectors, 300):
        n = len(phi)
        max_nodes = -(-n // num_sectors) + int(rng.integers(0, 10))
        sector, rank, edges = partition(phi, num_sectors, max_nodes)
        sizes = np.bincount(sector, minlength=num_sectors)
        assert len(sizes) == num_sectors
        assert sizes.min() >= 1
        assert sizes.max() <= max_nodes
        assert len(edges) == num_sectors
        assert (np.diff(edges) > 0).all()
        # every (sector, rank) slot holds one node: the ranks of a sector are 0 .. size - 1
        for j in range(num_sectors):
            assert sorted(rank[sector == j]) == list(range(sizes[j]))


def test_two_clusters_cut_in_the_gaps():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.05, 30)
    b = rng.normal(np.pi - 0.1, 0.05, 30)
    phi = wrap(np.concatenate([a, b]) + np.pi) - np.pi
    sector, _, _ = partition(phi, 2, 60)
    assert len(set(sector[:30])) == 1
    assert len(set(sector[30:])) == 1
    assert sector[0] != sector[30]


@pytest.mark.parametrize("num_sectors", [2, 3, 4, 6])
def test_no_cap_cuts_the_largest_gaps(num_sectors):
    for phi in random_events(10 + num_sectors, 100):
        sector, _, _ = partition(phi, num_sectors, len(phi))
        p = np.sort(phi)
        largest = np.sort(wrap(np.roll(p, -1) - p))[-num_sectors:]
        np.testing.assert_allclose(np.sort(cut_gaps(phi, sector)), largest, atol=1e-12)


@pytest.mark.parametrize("num_sectors", [2, 3, 4, 6])
def test_tight_cap_not_worse_than_equal_count(num_sectors):
    for phi in random_events(20 + num_sectors, 100):
        max_nodes = -(-len(phi) // num_sectors)
        sector, _, _ = partition(phi, num_sectors, max_nodes)
        assert np.bincount(sector).max() <= max_nodes
        assert cut_gaps(phi, sector).min() >= equal_count_min_gap(phi, num_sectors) - 1e-12


def test_rank_wraps_at_pi():
    # Sectors {-1.0, -0.9}, {1.0, 1.1} and {3.1, -3.1}; the last one straddles +-pi and starts near 2.1
    phi = np.array([3.1, -3.1, -1.0, -0.9, 1.0, 1.1])
    sector, rank, edges = partition(phi, 3, 2)
    assert sector[0] == sector[1] == 2
    assert edges[2] == pytest.approx(2.1)
    assert rank[0] == 0
    assert rank[1] == 1


def test_one_sector_ranks_by_phi():
    phi = np.random.default_rng(3).uniform(-np.pi, np.pi, 100)
    sector, rank, edges = partition(phi, 1, 160)
    assert (sector == 0).all()
    np.testing.assert_array_equal(edges, [-np.pi])
    np.testing.assert_array_equal(rank, np.argsort(np.argsort(phi)))


def test_permuting_nodes_permutes_the_partition():
    rng = np.random.default_rng(4)
    for phi in random_events(5, 50):
        perm = rng.permutation(len(phi))
        sector, rank, edges = partition(phi, 3, -(-len(phi) // 3) + 2)
        sector_p, rank_p, edges_p = partition(phi[perm], 3, -(-len(phi) // 3) + 2)
        np.testing.assert_array_equal(sector_p, sector[perm])
        np.testing.assert_array_equal(rank_p, rank[perm])
        np.testing.assert_array_equal(edges_p, edges)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_fewer_nodes_than_sectors(n):
    phi = np.linspace(-1.0, 1.0, n)
    sector, rank, _ = partition(phi, 3, 53)
    assert sorted(sector) == list(range(n))
    assert (rank == 0).all()


def test_too_many_nodes_raises():
    with pytest.raises(ValueError, match="do not fit"):
        partition(np.linspace(-3, 3, 160), 3, 53)


def synthetic_event(energies):
    """One event of 54 topoclusters, one neutral particle each: 52 in a tight cluster near phi 0, one at +2, one at -2.
    The `cap` partition (S = 3, M = 53) makes the cluster one sector, so that sector owns 52 targets for 50 slots.
    """
    ds = CLICDataset.__new__(CLICDataset)
    n = len(energies)
    phi = np.concatenate([np.linspace(0.0, 0.1, n - 2), [2.0, -2.0]])
    ds.n_tracks, ds.n_topos, ds.n_particles = np.array([0]), np.array([n]), np.array([n])
    ds.track_cumsum, ds.topo_cumsum, ds.particle_cumsum = [0, 0], [0, n], [0, n]
    ds.full_data_array = {
        "track_phi_int": torch.zeros(0),
        "track_phi": torch.zeros(0),
        "topo_phi": torch.tensor(phi, dtype=torch.float32),
        "particle_e": torch.tensor(energies, dtype=torch.float32),
        "particle_phi": torch.tensor(phi, dtype=torch.float32),
    }
    ds.aux_data_array = {
        "track_particle_idx": [np.zeros(0, dtype=np.int64)],
        "topo2particle_topo_idx": [np.arange(n)],
        "topo2particle_particle_idx": [np.arange(n)],
        "topo2particle_energy": [np.asarray(energies)],
    }
    ds.num_objects, ds.max_nodes = 150, 160
    ds.sectors = {"num_sectors": 3, "scheme": "cap", "max_nodes_per_sector": 53, "track_phi": "calo", "overflow": "drop_lowest_energy"}
    labels = {
        "particle_valid": torch.arange(150) < n,
        "particle_class": torch.where(torch.arange(150) < n, 3, 5),
        "particle_e": torch.where(torch.arange(150) < n, torch.arange(150).float(), torch.nan),
        "particle_node_valid": torch.eye(150, 160, dtype=torch.bool) & (torch.arange(150) < n)[:, None],
        "particle_incidence": torch.eye(150, 160) * (torch.arange(150) < n)[:, None],
    }
    inputs = {}
    ds.sectorise(0, inputs, labels)
    return inputs, labels


def test_overflow_drops_the_lowest_energy_targets_and_keeps_the_event():
    energies = np.random.default_rng(0).permutation(np.arange(1.0, 55.0))
    inputs, labels = synthetic_event(energies)
    assert labels["sector_dropped_targets"] == 2
    # sectors by lower edge: [-pi, ..) holds the node at -2, the cluster starts near -1, the node at +2 near 1.05
    np.testing.assert_array_equal(inputs["node_sector"][:54].numpy(), [1] * 52 + [2, 0])
    cluster = np.arange(52)
    kept = np.sort(cluster[np.argsort(energies[:52])[2:]])
    # particle_e holds the particle index here: block 1 = the 50 most energetic cluster particles, in particle order
    np.testing.assert_array_equal(labels["particle_e"][50:100].numpy(), kept)
    np.testing.assert_array_equal(labels["particle_e"][0].numpy(), 53)
    np.testing.assert_array_equal(labels["particle_e"][100].numpy(), 52)
    assert labels["particle_valid"].sum() == 52
    for j in range(3):
        block = labels["particle_valid"][50 * j : 50 * (j + 1)]
        n = int(block.sum())
        assert block[:n].all()
        assert (labels["particle_class"][50 * j + n : 50 * (j + 1)] == 5).all()
    # the restricted masks keep each target's own node, which is inside its sector
    assert (labels["particle_node_valid"].sum(-1) == labels["particle_valid"]).all()


def test_dropped_targets_are_logged_as_an_epoch_sum():
    logged = []

    class Recorder:
        def log(self, name, value, **kwargs):
            logged.append((name, float(value), kwargs))

    batch = ({}, {"sector_dropped_targets": torch.tensor([0, 2, 1])})
    MPflow.log_sector_drops(Recorder(), batch, "val")
    MPflow.log_sector_drops(Recorder(), ({}, {}), "val")
    assert len(logged) == 1
    name, value, kwargs = logged[0]
    assert name == "val/sector_dropped_targets"
    assert value == 3
    assert kwargs["reduce_fx"] == "sum"
    assert kwargs["on_epoch"]
    assert kwargs["sync_dist"]
