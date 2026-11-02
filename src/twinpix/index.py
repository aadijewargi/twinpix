"""Finding all pairs of hashes within a Hamming radius.

The naive approach compares every pair: n(n-1)/2 distance computations, which
is 200 million for a 20 000-photo library. Three strategies are provided:

* :func:`mih_pairs` - **multi-index hashing** (Norouzi, Punjani & Fleet,
  2012), the default for large libraries. Split each 64-bit hash into ``m``
  chunks. If two hashes differ in at most ``r`` bits, then by the pigeonhole
  principle at least one chunk differs in at most ``floor(r / m)`` bits. So
  for every chunk we bucket the hashes by that chunk's value and only compare
  hashes whose chunks are within ``floor(r / m)`` bits: a few hundred bucket
  lookups instead of n comparisons per photo. It is exact, not approximate.
* :func:`brute_force_pairs` - exact, vectorised all-pairs comparison in NumPy.
  The correctness reference, and the fastest choice for small libraries.
* :class:`BKTree` - a Burkhard-Keller tree. Hamming distance is a metric, so
  the triangle inequality lets a query skip subtrees whose edge label lies
  outside ``[d - r, d + r]``. Kept for comparison: in 64-bit Hamming space
  almost all distances crowd around 32, so at useful radii it still visits a
  quarter to a third of all pairs, and pure-Python tree walking then loses to
  NumPy brute force (see ``benchmarks/bench_index.py``).
"""

from __future__ import annotations

from itertools import combinations
from math import comb
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


def _chunk_widths(m: int, bits: int = 64) -> list[int]:
    base, extra = divmod(bits, m)
    return [base + (1 if k < extra else 0) for k in range(m)]


def _ball_size(b: int, s: int) -> int:
    """Number of b-bit values within Hamming distance s of a given value."""
    return sum(comb(b, k) for k in range(min(s, b) + 1))


MAX_TABLE_BITS = 22  # bucket-offset table of 2**22 entries = 32 MB
_BATCH = 4_000_000  # candidate pairs expanded at once (~100 MB of temporaries)


def mih_cost(n: int, radius: int, m: int) -> float:
    """Estimated work per query: bucket lookups plus candidate checks."""
    s = radius // m
    cost = 0.0
    for b in _chunk_widths(m):
        if b > MAX_TABLE_BITS:
            return float("inf")
        cost += _ball_size(b, s) * (1.0 + n / 2.0**b)
    return cost


def choose_chunks(n: int, radius: int) -> tuple[int, float]:
    """Pick the number of chunks m that minimises :func:`mih_cost`."""
    best = min(range(3, 17), key=lambda m: mih_cost(n, radius, m))
    return best, mih_cost(n, radius, best)


def _flip_masks(b: int, s: int) -> np.ndarray:
    masks = [0]
    for k in range(1, min(s, b) + 1):
        for bits in combinations(range(b), k):
            masks.append(sum(1 << x for x in bits))
    return np.asarray(masks, dtype=np.int64)


def mih_pairs(hashes: Sequence[int], radius: int, m: int | None = None) -> tuple[list[Pair], int]:
    """Exact all-pairs search with multi-index hashing.

    Returns the pairs and the number of candidate pairs that had their full
    64-bit distance checked (the analogue of the BK-tree's distance calls).
    """
    h = np.asarray([int(x) for x in hashes], dtype=np.uint64)
    n = len(h)
    if n < 2:
        return [], 0
    if m is None:
        m = choose_chunks(n, radius)[0]
    s = radius // m
    found: list[np.ndarray] = []
    checked = 0
    offset = 0
    for b in _chunk_widths(m):
        if b > MAX_TABLE_BITS:
            raise ValueError(f"chunk of {b} bits is too wide; use more chunks")
        keys = ((h >> np.uint64(offset)) & np.uint64((1 << b) - 1)).astype(np.int64)
        offset += b
        order = np.argsort(keys, kind="stable")
        # starts[v] .. starts[v + 1] is the slice of `order` whose chunk equals v.
        starts = np.zeros((1 << b) + 1, dtype=np.int64)
        np.cumsum(np.bincount(keys, minlength=1 << b), out=starts[1:])
        for flip in _flip_masks(b, s):
            q = keys ^ flip
            lo, hi = starts[q], starts[q + 1]
            counts = hi - lo
            # Expand candidates in batches so that a pathological bucket (say,
            # thousands of all-black frames) cannot exhaust memory.
            cum = np.cumsum(counts)
            if cum[-1] == 0:
                continue
            first = 0
            while first < n:
                base = cum[first - 1] if first else 0
                last = int(np.searchsorted(cum, base + _BATCH, side="right"))
                last = max(last, first + 1)
                sel = slice(first, last)
                c = counts[sel]
                total = int(c.sum())
                first = last
                if total == 0:
                    continue
                ii = np.repeat(np.arange(sel.start, sel.stop), c)
                run_start = np.repeat(np.cumsum(c) - c, c)
                jj = order[np.repeat(lo[sel], c) + (np.arange(total) - run_start)]
                keep = jj > ii  # every unordered pair is seen from both ends; keep one
                ii, jj = ii[keep], jj[keep]
                checked += len(ii)
                close = _popcount64(h[ii] ^ h[jj]) <= radius
                if close.any():
                    found.append(ii[close] * n + jj[close])
    if not found:
        return [], checked
    codes = np.unique(np.concatenate(found))  # a pair can match in several chunks
    ii, jj = codes // n, codes % n
    dist = _popcount64(h[ii] ^ h[jj])
    return list(zip(ii.tolist(), jj.tolist(), dist.astype(int).tolist())), checked


def find_pairs(hashes: Sequence[int], radius: int, method: str = "auto") -> list[Pair]:
    """Dispatch to a pair-search strategy.

    ``auto`` estimates the multi-index hashing cost and falls back to NumPy
    brute force when MIH would not save much (small libraries, or radii so
    large that the pigeonhole bound no longer prunes).
    """
    if method == "auto":
        n = len(hashes)
        method = "mih" if n > 1 and choose_chunks(n, radius)[1] < n / 4 else "brute"
    if method == "mih":
        return mih_pairs(hashes, radius)[0]
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

