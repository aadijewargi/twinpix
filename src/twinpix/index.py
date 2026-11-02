"""Finding all pairs of hashes within a Hamming radius.

The naive approach compares every pair: n(n-1)/2 distance computations, which
is 200 million for a 20 000-photo library. Two strategies are provided:

* :class:`BKTree` - a Burkhard-Keller tree. Hamming distance is a metric, so
  the triangle inequality lets a query skip whole subtrees: if the query is at
  distance ``d`` from a node, only children whose edge label lies in
  ``[d - r, d + r]`` can contain a match. Pairs are found incrementally by
  querying each hash *before* inserting it, so each pair is reported once.
* :func:`brute_force_pairs` - exact, vectorised all-pairs comparison in NumPy.
  It is the correctness reference for the tests and the faster choice for
  small libraries or generous thresholds where the tree cannot prune much.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Pair = tuple[int, int, int]  # (i, j, distance) with i < j


class BKTree:
    """BK-tree over 64-bit integer hashes with Hamming distance.

    Each node stores one hash value, the ids of every item with exactly that
    hash (so exact duplicates cost nothing extra), and a dict of children
    keyed by their distance to the node.
    """

    __slots__ = ("_root", "_size", "distance_calls")

    def __init__(self) -> None:
        self._root: list | None = None  # node = [hash, ids, children]
        self._size = 0
        self.distance_calls = 0

    def __len__(self) -> int:
        return self._size

    def add(self, value: int, item_id: int) -> None:
        self._size += 1
        if self._root is None:
            self._root = [value, [item_id], {}]
            return
        node = self._root
        while True:
            self.distance_calls += 1
            d = (node[0] ^ value).bit_count()
            if d == 0:
                node[1].append(item_id)
                return
            child = node[2].get(d)
            if child is None:
                node[2][d] = [value, [item_id], {}]
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        """Return ``(item_id, distance)`` for every stored item within ``radius``."""
        if self._root is None:
            return []
        found: list[tuple[int, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            self.distance_calls += 1
            d = (node[0] ^ value).bit_count()
            if d <= radius:
                found.extend((i, d) for i in node[1])
            lo, hi = d - radius, d + radius
            for edge, child in node[2].items():
                if lo <= edge <= hi:
                    stack.append(child)
        return found


def bktree_pairs(hashes: Sequence[int], radius: int) -> tuple[list[Pair], int]:
    """All pairs within ``radius`` using a BK-tree.

    Returns the pairs and the number of distance computations performed, so
    the pruning can be compared against the n(n-1)/2 of brute force.
    """
    tree = BKTree()
    pairs: list[Pair] = []
    for j, h in enumerate(hashes):
        for i, d in tree.query(h, radius):
            pairs.append((i, j, d) if i < j else (j, i, d))
        tree.add(h, j)
    pairs.sort()
    return pairs, tree.distance_calls


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # NumPy >= 2.0
        return np.bitwise_count(x)
    return _POPCOUNT8[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1)


def brute_force_pairs(hashes: Sequence[int], radius: int, block: int = 2048) -> list[Pair]:
    """Exact all-pairs search, vectorised in blocks to bound memory."""
    h = np.asarray([int(x) for x in hashes], dtype=np.uint64)
    n = len(h)
    pairs: list[Pair] = []
    for start in range(0, n, block):
        rows = h[start : start + block]
        dist = _popcount64(rows[:, None] ^ h[None, :])
        ii, jj = np.nonzero(dist <= radius)
        ii = ii + start
        keep = jj > ii
        for i, j in zip(ii[keep].tolist(), jj[keep].tolist()):
            pairs.append((i, j, int(dist[i - start, j])))
    pairs.sort()
    return pairs


def find_pairs(hashes: Sequence[int], radius: int, method: str = "auto") -> list[Pair]:
    """Dispatch to the BK-tree or brute force.

    ``auto`` uses brute force for small inputs (where NumPy's vectorisation
    wins) and the BK-tree for large ones (where pruning wins).
    """
    if method == "auto":
        method = "brute" if len(hashes) <= 4000 else "bktree"
    if method == "brute":
        return brute_force_pairs(hashes, radius)
    if method == "bktree":
        return bktree_pairs(hashes, radius)[0]
    raise ValueError(f"unknown method {method!r}")


class UnionFind:
    """Disjoint-set forest with path halving and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def connected_groups(n: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components with more than one member, largest first."""
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(uf.find(i), []).append(i)
    groups = [sorted(c) for c in comps.values() if len(c) > 1]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups

