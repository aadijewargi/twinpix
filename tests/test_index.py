import random

import numpy as np
import pytest

from twinpix import index
from twinpix.index import (
    BKTree, bktree_pairs, brute_force_pairs, choose_chunks, connected_groups, find_pairs, mih_pairs,
)


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
    assert mih_pairs(hashes, radius)[0] == expected


@pytest.mark.parametrize("radius", [2, 8, 12])
def test_bktree_matches_naive_on_clustered(radius):
    hashes = clustered_hashes(400, 40, 8, seed=radius)
    expected = naive_pairs(hashes, radius)
    assert bktree_pairs(hashes, radius)[0] == expected
    assert find_pairs(hashes, radius, "auto") == expected
    for m in (3, 4, 5, 8, 16):  # any chunk count is exact, only speed changes
        assert mih_pairs(hashes, radius, m=m)[0] == expected


def test_exact_duplicates_share_a_node():
    tree = BKTree()
    for i in range(5):
        tree.add(0xDEADBEEF, i)
    assert len(tree) == 5
    assert sorted(tree.query(0xDEADBEEF, 0)) == [(i, 0) for i in range(5)]
    assert bktree_pairs([7] * 4, 0)[0] == [(i, j, 0) for i in range(4) for j in range(i + 1, 4)]


def test_empty_and_single_inputs():
    for fn in (lambda h, r: bktree_pairs(h, r)[0], brute_force_pairs, lambda h, r: mih_pairs(h, r)[0]):
        assert fn([], 5) == []
        assert fn([123], 5) == []
    assert BKTree().query(1, 64) == []


def test_extreme_hash_values():
    hashes = [0, (1 << 64) - 1, 1 << 63, 1]
    for r in (0, 1, 63, 64):
        assert brute_force_pairs(hashes, r) == naive_pairs(hashes, r)
        assert bktree_pairs(hashes, r)[0] == naive_pairs(hashes, r)
        assert mih_pairs(hashes, r)[0] == naive_pairs(hashes, r)


def test_bktree_prunes_for_small_radius():
    hashes = clustered_hashes(3000, 300, 4, seed=9)
    n = len(hashes)
    _, calls = bktree_pairs(hashes, 4)
    assert calls < 0.5 * n * (n - 1) / 2


def test_mih_checks_a_tiny_fraction_of_pairs():
    rng = random.Random(1)
    hashes = clustered_hashes(20000, 2000, 4, seed=4) + [rng.getrandbits(64) for _ in range(20000)]
    n = len(hashes)
    pairs, checked = mih_pairs(hashes, 10)
    assert checked < 0.02 * n * (n - 1) / 2
    assert len(pairs) > 0


def test_mih_survives_a_huge_bucket(monkeypatch):
    from twinpix import index as ix

    monkeypatch.setattr(ix, "_BATCH", 1000)  # force many small batches
    hashes = [0xABCD] * 300 + clustered_hashes(500, 50, 3, seed=2)
    assert mih_pairs(hashes, 5)[0] == naive_pairs(hashes, 5)


def test_auto_picks_brute_force_when_mih_cannot_prune():
    m, cost = choose_chunks(500, 30)
    assert cost >= 500 / 4  # radius 30 of 64: pigeonhole bound is useless


@pytest.mark.parametrize("method", ["auto", "mih", "brute", "bktree"])
def test_every_method_agrees(method):
    hashes = clustered_hashes(1500, 100, 6, seed=21)
    assert find_pairs(hashes, 8, method) == naive_pairs(hashes, 8)


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
