"""BK-tree: a metric tree for fast "everything within distance r" queries.

Each node stores a hash; its children are keyed by their distance to the node.
For a query ``q`` with radius ``r`` at a node at distance ``d`` from ``q``, the
triangle inequality says any match ``x`` in the child subtree keyed ``k``
satisfies ``|d - k| <= r``. So only children with keys in ``[d - r, d + r]``
need visiting; the rest of the tree is pruned.

This works beautifully for small radii (a few bits). At the radii photo
de-duplication needs (~12 of 64 bits) it degrades badly: perceptual-hash
distances between unrelated photos concentrate around 32 +/- 4 bits, so the
window [d - r, d + r] admits nearly every child and a query visits ~85% of the
tree (measured in ``bench/speed.py``). twinshot therefore uses multi-index
hashing (``search.py``) by default and keeps the BK-tree as an alternative
engine (``--search bktree``) and a benchmark baseline.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .hashing import hamming


class _Node:
    __slots__ = ("key", "items", "children")

    def __init__(self, key: int, item: int):
        self.key = key
        self.items: List[int] = [item]  # several photos may share one hash
        self.children: Dict[int, "_Node"] = {}


class BKTree:
    def __init__(self, distance: Callable[[int, int], int] = hamming):
        self._distance = distance
        self._root: Optional[_Node] = None
        self._size = 0
        self.comparisons = 0  # distance evaluations, for benchmarking

    def __len__(self) -> int:
        return self._size

    def add(self, key: int, item: int) -> None:
        """Insert ``key`` (a hash) carrying a payload ``item`` (e.g. an index)."""
        self._size += 1
        if self._root is None:
            self._root = _Node(key, item)
            return
        node = self._root
        while True:
            d = self._distance(key, node.key)
            if d == 0:
                node.items.append(item)
                return
            child = node.children.get(d)
            if child is None:
                node.children[d] = _Node(key, item)
                return
            node = child

    def extend(self, keys: Iterable[int]) -> None:
        for i, k in enumerate(keys, start=self._size):
            self.add(k, i)

    def query(self, key: int, radius: int) -> List[Tuple[int, int]]:
        """Return ``(distance, item)`` for every stored item within ``radius``."""
        if self._root is None:
            return []
        out: List[Tuple[int, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = self._distance(key, node.key)
            self.comparisons += 1
            if d <= radius:
                out.extend((d, it) for it in node.items)
            lo, hi = d - radius, d + radius
            for k, child in node.children.items():
                if lo <= k <= hi:
                    stack.append(child)
        return out


def all_pairs_within(keys: List[int], radius: int) -> List[Tuple[int, int, int]]:
    """All index pairs ``(i, j, d)`` with ``i < j`` and ``hamming <= radius``.

    Builds the tree incrementally: each key queries the tree of the keys
    before it, then is inserted. Every pair is therefore reported exactly once.
    """
    tree = BKTree()
    pairs: List[Tuple[int, int, int]] = []
    for j, key in enumerate(keys):
        for d, i in tree.query(key, radius):
            pairs.append((i, j, d))
        tree.add(key, j)
    pairs.sort()
    return pairs


def all_pairs_brute_force(keys: List[int], radius: int) -> List[Tuple[int, int, int]]:
    """O(N^2) reference implementation (used by tests and benchmarks)."""
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = hamming(keys[i], keys[j])
            if d <= radius:
                pairs.append((i, j, d))
    return pairs
