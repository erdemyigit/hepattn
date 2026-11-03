"""Partition of an event's nodes into phi sectors for the sectored decoder.

The `cap` scheme sorts the nodes by phi and cuts the cycle into contiguous chunks of at most `max_nodes` nodes,
placing the cuts in the largest phi gaps that allow it (the smallest cut gap is maximised). The partition reads
only the inputs, so training and inference use the same rule.
"""

import numpy as np

TWO_PI = 2 * np.pi


def wrap(x: np.ndarray) -> np.ndarray:
    """Map angles to [0, 2 pi)."""
    return np.mod(x, TWO_PI)


def gap_cuts(gaps: np.ndarray, num_sectors: int, max_nodes: int) -> np.ndarray:
    """Cut positions giving `num_sectors` cyclic chunks of at most `max_nodes` nodes, maximising the smallest cut gap.

    `gaps[i]` is the phi gap between sorted nodes i and i + 1 (cyclic); a cut at i ends a chunk at node i.
    Binary search on how many of the largest gaps may be cut; from each allowed start, greedy farthest jumps try to
    cover the cycle. If fewer cuts than `num_sectors` suffice, the largest unused allowed gaps are cut as well.

    Raises:
        ValueError: If no partition exists (`num_sectors * max_nodes < len(gaps)`) or `num_sectors > len(gaps)`.
    """
    n = len(gaps)
    if num_sectors > n or num_sectors * max_nodes < n:
        raise ValueError(f"cannot cut {n} nodes into {num_sectors} chunks of at most {max_nodes}")
    order = np.argsort(-gaps, kind="stable")

    def solve(k: int) -> np.ndarray | None:
        allowed = order[:k]
        if n <= max_nodes:
            return allowed[:num_sectors]
        a = np.sort(allowed)
        a2 = np.concatenate([a, a + n])
        far = np.searchsorted(a2, a2 + max_nodes, side="right") - 1
        for i0 in range(len(a)):
            pos, cuts = i0, [a[i0]]
            while a[i0] + n - a2[pos] > max_nodes:
                nxt = far[pos]
                if nxt == pos or len(cuts) == num_sectors:
                    break
                pos = nxt
                cuts.append(a2[pos] % n)
            else:
                used = set(cuts)
                rest = [c for c in allowed if c not in used][: num_sectors - len(cuts)]
                return np.array(cuts + rest)
        return None

    lo, hi = num_sectors, n  # solve(n) succeeds whenever max_nodes >= ceil(n / num_sectors)
    while lo < hi:
        mid = (lo + hi) // 2
        if solve(mid) is None:
            lo = mid + 1
        else:
            hi = mid
    cuts = solve(lo)
    assert cuts is not None
    return cuts


def partition(phi: np.ndarray, num_sectors: int, max_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the nodes of one event into `num_sectors` phi sectors of at most `max_nodes` nodes (`cap` scheme).

    Args:
        phi: (n,) node phi of the valid nodes, in radians.
        num_sectors: Number of sectors S.
        max_nodes: Largest number of nodes in a sector, M.

    Returns:
        sector: (n,) int64 sector of each node. Sectors are numbered by their lower edge, ascending in [-pi, pi);
            the last one may straddle +-pi. With fewer than S nodes the highest sectors are empty.
        rank: (n,) int64 position of each node in its sector, by cyclic phi from the sector's lower edge.
        edges: (S',) lower edges of the S' = min(S, n) non-empty sectors, in [-pi, pi). S = 1 gives [-pi].

    Raises:
        ValueError: If S x M < n.
    """
    phi = np.asarray(phi, dtype=np.float64)
    n = len(phi)
    if num_sectors * max_nodes < n:
        raise ValueError(f"{n} nodes do not fit in {num_sectors} sectors of {max_nodes}")
    sector = np.zeros(n, dtype=np.int64)
    if num_sectors == 1 or n <= 1:
        edges = np.array([-np.pi])
    else:
        order = np.argsort(phi, kind="stable")
        p = phi[order]
        gaps = wrap(np.roll(p, -1) - p)
        cuts = np.sort(gap_cuts(gaps, min(num_sectors, n), max_nodes))
        # chunk j holds the sorted nodes after cut j up to cut j + 1 (cyclic); its lower edge is mid-gap at cut j
        lower = wrap(p[cuts] + gaps[cuts] / 2 + np.pi) - np.pi
        chunk = (np.searchsorted(cuts, np.arange(n), side="left") - 1) % len(cuts)
        # number the chunks by their lower edge
        by_edge = np.argsort(lower, kind="stable")
        sector[order] = np.argsort(by_edge)[chunk]
        edges = lower[by_edge]
    rank = np.zeros(n, dtype=np.int64)
    for j in range(len(edges)):
        members = np.flatnonzero(sector == j)
        rank[members[np.argsort(wrap(phi[members] - edges[j]), kind="stable")]] = np.arange(len(members))
    return sector, rank, edges
