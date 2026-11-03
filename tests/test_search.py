import numpy as np
import pytest

from twinshot.bktree import BKTree, all_pairs_brute_force, all_pairs_within
from twinshot.search import _chunk_layout, all_pairs, as_uint64, pairs_mih

METHODS = ["brute", "mih", "bktree", "auto"]


def planted(n, seed=0, dup_frac=0.3, max_flips=14):
    """Random 64-bit keys plus noisy near-copies, so every radius has matches."""
    rng = np.random.default_rng(seed)
    keys = []
    while len(keys) < n:
        k = int(rng.integers(0, 2**63, dtype=np.int64)) << 1 | int(rng.integers(0, 2))
        keys.append(k)
        if rng.random() < dup_frac:
            c = k
            for b in rng.choice(64, rng.integers(0, max_flips + 1), replace=False):
                c ^= 1 << int(b)
            keys.append(c)
    return keys[:n]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("radius", [0, 1, 3, 8, 12, 20])
def test_methods_match_brute_force_reference(method, radius):
    keys = planted(400, seed=radius)
    expected = all_pairs_brute_force(keys, radius)
    assert all_pairs(keys, radius, method) == expected


@pytest.mark.parametrize("method", METHODS)
def test_edge_sizes(method):
    assert all_pairs([], 5, method) == []
    assert all_pairs([123], 5, method) == []
    assert all_pairs([1, 1], 0, method) == [(0, 1, 0)]


@pytest.mark.parametrize("method", METHODS)
def test_identical_keys_all_pair_up(method):
    keys = [42] * 5 + [2**64 - 1]
    got = all_pairs(keys, 0, method)
    assert got == [(i, j, 0) for i in range(5) for j in range(i + 1, 5)]


@pytest.mark.parametrize("method", ["brute", "mih"])
def test_radius_64_returns_every_pair(method):
    keys = planted(30, seed=3)
    assert len(all_pairs(keys, 64, method)) == 30 * 29 // 2


def test_negative_radius_rejected():
    with pytest.raises(ValueError):
        all_pairs([1, 2], -1)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        all_pairs([1, 2], 1, "magic")


@pytest.mark.parametrize("m", range(2, 9))
def test_mih_any_chunk_count_is_exact(m):
    keys = planted(300, seed=m)
    h = as_uint64(keys)
    i, j, d = pairs_mih(h, 12, m=m)
    assert list(zip(i.tolist(), j.tolist(), d.tolist())) == all_pairs_brute_force(keys, 12)


def test_mih_bounded_memory_slicing_is_exact():
    keys = planted(500, seed=9)
    h = as_uint64(keys)
    i, j, d = pairs_mih(h, 12, m=4, max_candidates=7)
    assert list(zip(i.tolist(), j.tolist(), d.tolist())) == all_pairs_brute_force(keys, 12)


@pytest.mark.parametrize("m", range(1, 14))
def test_chunk_layout_covers_all_bits(m):
    layout = _chunk_layout(m)
    assert sum(b for _, b in layout) == 64
    assert [o for o, _ in layout] == list(np.cumsum([0] + [b for _, b in layout[:-1]]))


def test_bktree_query_matches_linear_scan():
    keys = planted(300, seed=5)
    tree = BKTree()
    for i, k in enumerate(keys):
        tree.add(k, i)
    assert len(tree) == 300
    for q in keys[:25]:
        for r in (0, 4, 10):
            got = sorted(tree.query(q, r))
            want = sorted((bin(q ^ k).count("1"), i) for i, k in enumerate(keys) if bin(q ^ k).count("1") <= r)
            assert got == want


def test_bktree_all_pairs_matches_brute():
    keys = planted(250, seed=11)
    assert all_pairs_within(keys, 9) == all_pairs_brute_force(keys, 9)


def test_bktree_empty_query():
    assert BKTree().query(5, 3) == []


def test_popcount_fallback_without_numpy_bitwise_count(monkeypatch):
    # NumPy < 2.0 has no bitwise_count; the lookup-table path must agree.
    from twinshot import search

    keys = planted(200, seed=21)
    expected = all_pairs_brute_force(keys, 10)
    monkeypatch.delattr(search.np, "bitwise_count", raising=False)
    assert all_pairs(keys, 10, "brute") == expected
    assert all_pairs(keys, 10, "mih") == expected
