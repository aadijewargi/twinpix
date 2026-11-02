import random

import numpy as np
import pytest

from twinpix import index
from twinpix.index import BKTree, bktree_pairs, brute_force_pairs, connected_groups, find_pairs


def naive_pairs(hashes, r):
    """The obviously-correct O(n^2) reference."""
    out = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = bin(hashes[i] ^ hashes[j]).count("1")
            if d <= r:
                out.append((i, j, d))
    return out


def clustered_hashes(n, n_clusters, flips, seed):
    """Random 64-bit centres with nearby variants, like real photo libraries."""
    rng = random.Random(seed)
    centres = [rng.getrandbits(64) for _ in range(n_clusters)]
    out = []
    for _ in range(n):
        h = rng.choice(centres)
        for _ in range(rng.randint(0, flips)):
            h ^= 1 << rng.randrange(64)
        out.append(h)
    return out


@pytest.mark.parametrize("radius", [0, 1, 3, 6, 10, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bktree_matches_naive_on_random(radius, seed):
    rng = random.Random(seed)
    hashes = [rng.getrandbits(64) for _ in range(300)]
    hashes += [h ^ (1 << rng.randrange(64)) for h in hashes[:50]]  # guarantee near pairs
    expected = naive_pairs(hashes, radius)
    assert bktree_pairs(hashes, radius)[0] == expected
    assert brute_force_pairs(hashes, radius, block=64) == expected


@pytest.mark.parametrize("radius", [2, 8, 12])
def test_bktree_matches_naive_on_clustered(radius):
    hashes = clustered_hashes(400, 40, 8, seed=radius)
    expected = naive_pairs(hashes, radius)
    assert bktree_pairs(hashes, radius)[0] == expected
    assert find_pairs(hashes, radius, "auto") == expected


def test_exact_duplicates_share_a_node():
    tree = BKTree()
    for i in range(5):
        tree.add(0xDEADBEEF, i)
    assert len(tree) == 5
    assert sorted(tree.query(0xDEADBEEF, 0)) == [(i, 0) for i in range(5)]
    assert bktree_pairs([7] * 4, 0)[0] == [(i, j, 0) for i in range(4) for j in range(i + 1, 4)]


def test_empty_and_single_inputs():
    for fn in (lambda h, r: bktree_pairs(h, r)[0], brute_force_pairs):
        assert fn([], 5) == []
        assert fn([123], 5) == []
    assert BKTree().query(1, 64) == []


def test_extreme_hash_values():
    hashes = [0, (1 << 64) - 1, 1 << 63, 1]
    for r in (0, 1, 63, 64):
        assert brute_force_pairs(hashes, r) == naive_pairs(hashes, r)
        assert bktree_pairs(hashes, r)[0] == naive_pairs(hashes, r)


def test_bktree_prunes_for_small_radius():
    hashes = clustered_hashes(3000, 300, 4, seed=9)
    n = len(hashes)
    _, calls = bktree_pairs(hashes, 4)
    assert calls < 0.5 * n * (n - 1) / 2


def test_popcount_fallback_matches(monkeypatch):
    x = np.array([0, 1, 3, (1 << 64) - 1, 0xF0F0], dtype=np.uint64)
    fast = index._popcount64(x)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    slow = index._popcount64(x)
    assert fast.tolist() == slow.tolist() == [0, 1, 2, 64, 8]


def test_connected_groups_merges_chains():
    groups = connected_groups(7, [(0, 1), (1, 2), (4, 5)])
    assert groups == [[0, 1, 2], [4, 5]]  # singletons 3 and 6 are dropped


def test_unknown_method():
    with pytest.raises(ValueError):
        find_pairs([1, 2], 3, method="magic")
