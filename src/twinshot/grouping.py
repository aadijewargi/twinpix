"""Turn pairwise matches into duplicate groups and pick a keeper in each."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Dict, List, Sequence, Tuple

from .hashing import Fingerprint, hamming
from .search import all_pairs

DEFAULT_PHASH_RADIUS = 12
DEFAULT_DHASH_RADIUS = 14


class UnionFind:
    """Disjoint sets with path halving and union by size: ~O(alpha(N)) per op."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def groups(self) -> List[List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def matching_pairs(
    fps: Sequence[Fingerprint],
    phash_radius: int = DEFAULT_PHASH_RADIUS,
    dhash_radius: int = DEFAULT_DHASH_RADIUS,
    method: str = "auto",
) -> List[Tuple[int, int, int]]:
    """Pairs that are close in pHash (via the BK-tree) *and* confirmed by dHash.

    pHash is the candidate generator (multi-index hashing or vectorised brute
    force, see ``search.py``): it is the most robust to resizing and
    re-compression. dHash looks at a different feature (local gradients), so
    requiring both cuts false positives from images that merely share a
    similar low-frequency layout (two different sunsets, say).
    """
    pairs = all_pairs([f.phash for f in fps], phash_radius, method=method)
    return [
        (i, j, d)
        for i, j, d in pairs
        if hamming(fps[i].dhash, fps[j].dhash) <= dhash_radius
    ]


def _complete_linkage(members: List[int], adj: Dict[int, set]) -> List[List[int]]:
    """Split a connected component into clusters where *every* pair matched.

    Single linkage (plain union-find) can chain A~B~C into one group even when
    A and C look nothing alike. This greedy pass grows each cluster from the
    best-connected remaining photo and only admits photos that match all
    current members. Exact maximum-clique partitioning is NP-hard; greedy is
    the standard practical compromise and components are tiny in practice.
    """
    remaining = sorted(members, key=lambda m: (-len(adj[m] & set(members)), m))
    clusters: List[List[int]] = []
    left = set(members)
    for seed in remaining:
        if seed not in left:
            continue
        cluster = [seed]
        left.discard(seed)
        for cand in remaining:
            if cand in left and all(cand in adj[c] for c in cluster):
                cluster.append(cand)
                left.discard(cand)
        clusters.append(sorted(cluster))
    return clusters


def _file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compare_quality(a: Fingerprint, b: Fingerprint) -> Tuple[int, str]:
    """Return (-1 if a is better, 1 if b is better, 0 tie) and the reason."""
    # 1. Resolution: more than 5% more pixels wins outright.
    if a.pixels > b.pixels * 1.05:
        return -1, "highest resolution"
    if b.pixels > a.pixels * 1.05:
        return 1, "highest resolution"
    # 2. Colour beats a black-and-white conversion of the same shot.
    GRAY = 3.0
    if a.colorfulness > GRAY >= b.colorfulness:
        return -1, "in colour"
    if b.colorfulness > GRAY >= a.colorfulness:
        return 1, "in colour"
    # 3. Same size: the sharpest frame wins (bursts, focus misses).
    if a.sharpness > b.sharpness * 1.10:
        return -1, "sharpest"
    if b.sharpness > a.sharpness * 1.10:
        return 1, "sharpest"
    # 4. Larger file at the same size = less aggressive compression.
    if a.bytes > b.bytes * 1.02:
        return -1, "least compressed"
    if b.bytes > a.bytes * 1.02:
        return 1, "least compressed"
    # 5. Oldest file is most likely the original.
    if a.mtime < b.mtime:
        return -1, "oldest (likely original)"
    if b.mtime < a.mtime:
        return 1, "oldest (likely original)"
    # 6. Deterministic tie-break: shortest path, then alphabetical.
    ka, kb = (len(a.path), a.path), (len(b.path), b.path)
    if ka == kb:
        return 0, "tie"
    return (-1 if ka < kb else 1), "shortest path"


def rank_by_quality(fps: Sequence[Fingerprint]) -> Tuple[List[int], str]:
    """Indices of ``fps`` ordered best first, plus why the winner won."""
    order = sorted(
        range(len(fps)), key=cmp_to_key(lambda i, j: _compare_quality(fps[i], fps[j])[0])
    )
    reason = "only photo"
    if len(order) > 1:
        reason = _compare_quality(fps[order[0]], fps[order[1]])[1]
    return order, reason


@dataclass
class Member:
    fp: Fingerprint
    phash_dist: int  # distance to the keeper
    dhash_dist: int
    exact_copy: bool = False  # byte-identical to the keeper


@dataclass
class Group:
    id: int
    members: List[Member]  # keeper first
    reason: str
    kind: str = "similar"  # "exact" when every member is byte-identical

    @property
    def keeper(self) -> Fingerprint:
        return self.members[0].fp

    @property
    def reclaimable_bytes(self) -> int:
        return sum(m.fp.bytes for m in self.members[1:])


@dataclass
class ScanResult:
    groups: List[Group] = field(default_factory=list)
    scanned: int = 0
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def duplicates(self) -> int:
        return sum(len(g.members) - 1 for g in self.groups)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(g.reclaimable_bytes for g in self.groups)


def find_groups(
    fps: Sequence[Fingerprint],
    phash_radius: int = DEFAULT_PHASH_RADIUS,
    dhash_radius: int = DEFAULT_DHASH_RADIUS,
    linkage: str = "complete",
    check_exact: bool = True,
    method: str = "auto",
) -> List[Group]:
    """Cluster fingerprints into duplicate groups (singletons are dropped)."""
    if linkage not in ("single", "complete"):
        raise ValueError("linkage must be 'single' or 'complete'")
    pairs = matching_pairs(fps, phash_radius, dhash_radius, method)
    uf = UnionFind(len(fps))
    adj: Dict[int, set] = {}
    for i, j, _ in pairs:
        uf.union(i, j)
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    clusters: List[List[int]] = []
    for comp in uf.groups():
        if len(comp) < 2:
            continue
        if linkage == "single":
            clusters.append(sorted(comp))
        else:
            clusters.extend(c for c in _complete_linkage(comp, adj) if len(c) > 1)

    groups: List[Group] = []
    for idx in clusters:
        sub = [fps[i] for i in idx]
        order, reason = rank_by_quality(sub)
        keeper = sub[order[0]]
        digests: Dict[str, str] = {}
        if check_exact:
            same_size = [f for f in sub if f.bytes == keeper.bytes]
            if len(same_size) > 1:
                for f in same_size:
                    try:
                        digests[f.path] = _file_digest(f.path)
                    except OSError:
                        pass
        members = []
        for k in order:
            f = sub[k]
            exact = (
                f is not keeper
                and f.path in digests
                and digests.get(f.path) == digests.get(keeper.path)
            )
            members.append(
                Member(
                    fp=f,
                    phash_dist=hamming(f.phash, keeper.phash),
                    dhash_dist=hamming(f.dhash, keeper.dhash),
                    exact_copy=exact,
                )
            )
        kind = "exact" if all(m.exact_copy for m in members[1:]) else "similar"
        groups.append(Group(id=0, members=members, reason=reason, kind=kind))

    # Biggest wins first: most reclaimable space, then path for stability.
    groups.sort(key=lambda g: (-g.reclaimable_bytes, g.keeper.path))
    for n, g in enumerate(groups, start=1):
        g.id = n
    return groups
