"""All-pairs Hamming search: find every pair of hashes within ``radius`` bits.

Three interchangeable engines, all returning the same pairs:

``brute``  Vectorised NumPy over blocks of rows: XOR + popcount on uint64.
           O(N^2) work but each comparison costs about a nanosecond.

``mih``    Multi-index hashing (Norouzi, Punjani & Fleet, CVPR 2012). Split
           the 64-bit hash into ``m`` disjoint chunks. By the pigeonhole
           principle, if two hashes differ in at most ``r`` bits, at least one
           chunk differs in at most ``floor(r / m)`` bits. So for each chunk we
           look up every bucket within that small radius in a sorted table
           (exact-match joins via ``searchsorted``), and only verify the
           resulting candidates. Sub-quadratic: work grows ~N^2 / 2^b instead
           of N^2, where b is the chunk width.

``bktree`` The classic metric tree (``bktree.py``). Kept for comparison: with
           64-bit perceptual hashes distances concentrate around 32 +/- 4 bits,
           so at radius 12 the triangle-inequality window [d-12, d+12] covers
           almost every child and the tree prunes very little. See
           ``bench/speed.py``.

``auto`` uses brute force for small libraries (it has no setup cost) and MIH
above a few thousand photos.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import comb
from typing import List, Sequence, Tuple

import numpy as np

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Per-element popcount of a uint64 array (fast path on NumPy >= 2.0)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    return _POP8[x.view(np.uint8)].reshape(x.shape + (8,)).sum(axis=-1).astype(np.int64)


def as_uint64(keys: Sequence[int]) -> np.ndarray:
    return np.array([k & 0xFFFFFFFFFFFFFFFF for k in keys], dtype=np.uint64)


Pairs = Tuple[np.ndarray, np.ndarray, np.ndarray]  # i, j, distance (i < j)


def _empty() -> Pairs:
    e = np.empty(0, dtype=np.int64)
    return e, e.copy(), e.copy()


def _sorted_unique(i: np.ndarray, j: np.ndarray, d: np.ndarray, n: int) -> Pairs:
    if len(i) == 0:
        return _empty()
    key = i.astype(np.int64) * n + j.astype(np.int64)
    key, first = np.unique(key, return_index=True)
    return key // n, key % n, d[first]


def pairs_brute(h: np.ndarray, radius: int, block_elems: int = 1 << 22) -> Pairs:
    n = len(h)
    out_i, out_j, out_d = [], [], []
    rows = max(1, block_elems // max(n, 1))
    for s in range(0, n, rows):
        e = min(n, s + rows)
        x = h[s:e, None] ^ h[None, :]
        d = popcount64(x)
        ii, jj = np.nonzero(d <= radius)
        ii = ii + s
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        out_i.append(ii)
        out_j.append(jj)
        out_d.append(d[ii - s, jj])
    if not out_i:
        return _empty()
    return _sorted_unique(np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d), n)


@lru_cache(maxsize=64)
def _flip_masks(bits: int, max_flips: int) -> np.ndarray:
    masks = []
    for k in range(max_flips + 1):
        for combo in combinations(range(bits), k):
            m = 0
            for b in combo:
                m |= 1 << b
            masks.append(m)
    return np.array(masks, dtype=np.int64)


def _chunk_layout(m: int, width: int = 64) -> List[Tuple[int, int]]:
    """(offset, bits) for m near-equal chunks covering ``width`` bits."""
    base, extra = divmod(width, m)
    layout, off = [], 0
    for c in range(m):
        b = base + (1 if c < extra else 0)
        layout.append((off, b))
        off += b
    return layout


def mih_cost(n: int, radius: int, m: int) -> float:
    """Rough cost model: probes * n * log(n) + expected candidates."""
    s = radius // m
    total = 0.0
    for _, b in _chunk_layout(m):
        probes = sum(comb(b, k) for k in range(min(s, b) + 1))
        total += probes * (n * np.log2(max(n, 2)) * 0.5 + n * n / 2 ** b)
    return total


def choose_chunks(n: int, radius: int) -> int:
    return min(range(2, 9), key=lambda m: mih_cost(n, radius, m))


def mih_worthwhile(n: int, radius: int) -> bool:
    """MIH only pays off when its probe tables stay small relative to N^2."""
    return mih_cost(n, radius, choose_chunks(n, radius)) < 0.5 * n * n


def pairs_mih(h: np.ndarray, radius: int, m: int = 0, max_candidates: int = 1 << 22) -> Pairs:
    n = len(h)
    if n < 2:
        return _empty()
    if m <= 0:
        if not mih_worthwhile(n, radius):  # e.g. huge radius: probing would explode
            return pairs_brute(h, radius)
        m = choose_chunks(n, radius)
    s = radius // m
    if sum(comb(b, k) for _, b in _chunk_layout(m) for k in range(min(s, b) + 1)) > 4 * n * n + 1024:
        return pairs_brute(h, radius)
    idx = np.arange(n, dtype=np.int64)
    out_i, out_j, out_d = [], [], []
    for off, bits in _chunk_layout(m):
        keys = ((h >> np.uint64(off)) & np.uint64((1 << bits) - 1)).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        for flip in _flip_masks(bits, min(s, bits)):
            targets = keys ^ flip
            lo = np.searchsorted(skeys, targets, "left")
            hi = np.searchsorted(skeys, targets, "right")
            counts = hi - lo
            if int(counts.sum()) == 0:
                continue
            # Expand candidate pairs in slices so memory stays bounded.
            csum = np.cumsum(counts)
            start = 0
            while start < n:
                base = int(csum[start - 1]) if start else 0
                stop = int(np.searchsorted(csum, base + max_candidates, "right"))
                stop = max(stop, start + 1)
                c = counts[start:stop]
                total = int(c.sum())
                if total:
                    ii = np.repeat(idx[start:stop], c)
                    offs = np.repeat(np.cumsum(c) - c, c)
                    jj = order[np.arange(total) - offs + np.repeat(lo[start:stop], c)]
                    keep = ii < jj
                    ii, jj = ii[keep], jj[keep]
                    d = popcount64(h[ii] ^ h[jj])
                    ok = d <= radius
                    out_i.append(ii[ok])
                    out_j.append(jj[ok])
                    out_d.append(d[ok])
                start = stop
    if not out_i:
        return _empty()
    return _sorted_unique(np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d), n)


def pairs_bktree(h: np.ndarray, radius: int) -> Pairs:
    from .bktree import all_pairs_within as bk

    trip = bk([int(x) for x in h], radius)
    if not trip:
        return _empty()
    a = np.array(trip, dtype=np.int64)
    return _sorted_unique(a[:, 0], a[:, 1], a[:, 2], len(h))


AUTO_BRUTE_LIMIT = 1500


def all_pairs(keys: Sequence[int], radius: int, method: str = "auto") -> List[Tuple[int, int, int]]:
    """Every ``(i, j, distance)`` with ``i < j`` and Hamming distance <= radius."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    h = keys if isinstance(keys, np.ndarray) else as_uint64(keys)
    if method == "auto":
        method = "brute" if len(h) <= AUTO_BRUTE_LIMIT else "mih"
    if method == "brute":
        i, j, d = pairs_brute(h, radius)
    elif method == "mih":
        i, j, d = pairs_mih(h, radius)
    elif method == "bktree":
        i, j, d = pairs_bktree(h, radius)
    else:
        raise ValueError(f"unknown search method {method!r}")
    return list(zip(i.tolist(), j.tolist(), d.tolist()))
