"""Choosing the best photo in a group of near-duplicates.

The rule is a lexicographic comparison with tolerances, in this order:

1. **Resolution** - a copy with clearly more pixels (>5 %) wins. Resized
   thumbnails and messenger-app copies lose.
2. **Sharpness** - among same-size shots, a clearly sharper one (>15 %
   higher Laplacian variance) wins. This picks the in-focus frame of a burst.
3. **File size** - otherwise the larger file wins; at the same resolution
   that usually means lighter compression or a lossless original.
4. **Age** - finally the older file, which is more likely the original.

Tolerances stop meaningless differences (a 2 % sharpness wobble from
recompression) from overriding the more trustworthy signals below them.
"""

from __future__ import annotations

from typing import Sequence

from .model import Photo

RES_TOLERANCE = 0.05
SHARP_TOLERANCE = 0.15


def _clearly_greater(a: float, b: float, tol: float) -> bool:
    return a > b * (1.0 + tol)


def is_better(a: Photo, b: Photo) -> bool:
    """True if ``a`` should be kept in preference to ``b``."""
    if _clearly_greater(a.pixels, b.pixels, RES_TOLERANCE):
        return True
    if _clearly_greater(b.pixels, a.pixels, RES_TOLERANCE):
        return False
    if _clearly_greater(a.sharpness, b.sharpness, SHARP_TOLERANCE):
        return True
    if _clearly_greater(b.sharpness, a.sharpness, SHARP_TOLERANCE):
        return False
    if a.size != b.size:
        return a.size > b.size
    if a.mtime_ns != b.mtime_ns:
        return a.mtime_ns < b.mtime_ns
    return a.path < b.path  # deterministic final tie-break


def pick_keeper(photos: Sequence[Photo]) -> Photo:
    if not photos:
        raise ValueError("empty group")
    best = photos[0]
    for p in photos[1:]:
        if is_better(p, best):
            best = p
    return best
