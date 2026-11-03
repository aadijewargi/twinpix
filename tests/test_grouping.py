import itertools

import pytest

from twinshot.grouping import UnionFind, _complete_linkage, find_groups, rank_by_quality
from twinshot.hashing import Fingerprint, fingerprint
from twinshot.scanner import iter_image_files


def fp(path, phash=0, dhash=0, w=100, h=100, size=1000, mtime=0.0, sharp=10.0, color=20.0):
    return Fingerprint(path, size, mtime, w, h, phash, dhash, sharp, color)


def test_union_find_groups():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    assert not uf.union(0, 2)
    groups = sorted(sorted(g) for g in uf.groups())
    assert groups == [[0, 1, 2], [3], [4, 5]]


def test_complete_linkage_breaks_chains():
    # A~B, B~C but A !~ C: a chain, not a clique.
    adj = {0: {1}, 1: {0, 2}, 2: {1}}
    clusters = _complete_linkage([0, 1, 2], adj)
    for c in clusters:
        for a, b in itertools.combinations(c, 2):
            assert b in adj[a]
    assert sorted(len(c) for c in clusters) == [1, 2]


def chain_fps():
    # pHash distances: A-B = 8, B-C = 8, A-C = 16 (radius 12 links A-B and B-C only)
    a, b, c = 0, (1 << 8) - 1, (1 << 16) - 1
    return [fp("A", a), fp("B", b), fp("C", c)]


def test_single_vs_complete_linkage():
    single = find_groups(chain_fps(), 12, 64, linkage="single", check_exact=False)
    assert len(single) == 1 and len(single[0].members) == 3
    complete = find_groups(chain_fps(), 12, 64, linkage="complete", check_exact=False)
    assert len(complete) == 1 and len(complete[0].members) == 2


def test_dhash_confirmation_filters_pairs():
    fps = [fp("A", 0, 0), fp("B", 1, (1 << 20) - 1)]
    assert find_groups(fps, 12, 64, check_exact=False)
    assert not find_groups(fps, 12, 14, check_exact=False)


def test_bad_linkage():
    with pytest.raises(ValueError):
        find_groups([], linkage="average")


def test_no_fingerprints():
    assert find_groups([]) == []


@pytest.mark.parametrize(
    "best, other, reason",
    [
        (dict(w=4000, h=3000), dict(w=1600, h=1200), "highest resolution"),
        (dict(color=25.0), dict(color=0.5), "in colour"),
        (dict(sharp=30.0), dict(sharp=10.0), "sharpest"),
        (dict(size=5000), dict(size=1000), "least compressed"),
        (dict(mtime=1.0), dict(mtime=2.0), "oldest (likely original)"),
    ],
)
def test_keeper_rules(best, other, reason):
    a, b = fp("/x/best.jpg", **best), fp("/x/other.jpg", **other)
    for pair in ([a, b], [b, a]):
        order, why = rank_by_quality(pair)
        assert pair[order[0]] is a and why == reason


def test_keeper_rule_priority_resolution_beats_sharpness():
    big_soft = fp("/big.jpg", w=4000, h=3000, sharp=5)
    small_sharp = fp("/small.jpg", w=1000, h=750, sharp=50)
    order, why = rank_by_quality([small_sharp, big_soft])
    assert order[0] == 1 and why == "highest resolution"


def test_keeper_tie_break_is_deterministic():
    a, b = fp("/a/x.jpg"), fp("/a/longer.jpg")
    assert rank_by_quality([b, a])[0][0] == 1
    assert rank_by_quality([a, b])[0][0] == 0


def test_groups_sorted_by_reclaimable_space_and_numbered():
    fps = [fp("a1", 0, size=10), fp("a2", 0, size=10), fp("b1", 2**64 - 1, size=500), fp("b2", 2**64 - 1, size=500)]
    groups = find_groups(fps, check_exact=False)
    assert [g.id for g in groups] == [1, 2]
    assert groups[0].keeper.path.startswith("b")


def test_library_groups_match_truth(library):
    root, truth = library
    fps = [fingerprint(p) for p in iter_image_files([str(root)])]
    groups = find_groups(fps)
    got = {frozenset(m.fp.path for m in g.members) for g in groups}
    want = {frozenset(v) for v in truth.values()}
    assert got == want
    for g in groups:
        expected_keeper = next(v[0] for v in truth.values() if g.keeper.path in v)
        assert g.keeper.path == expected_keeper


def test_exact_copies_detected(library):
    root, truth = library
    fps = [fingerprint(p) for p in truth["c"]]
    (g,) = find_groups(fps)
    assert g.kind == "exact" and g.members[1].exact_copy
    fps = [fingerprint(p) for p in truth["a"]]
    (g,) = find_groups(fps)
    assert g.kind == "similar" and not any(m.exact_copy for m in g.members)
