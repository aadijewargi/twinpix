"""twinshot: find near-duplicate photos with perceptual hashing and a BK-tree."""

__version__ = "0.1.0"

from .bktree import BKTree, all_pairs_within  # noqa: E402
from .grouping import Group, ScanResult, find_groups  # noqa: E402
from .hashing import Fingerprint, ahash, dhash, fingerprint, hamming, phash  # noqa: E402

__all__ = [
    "BKTree", "all_pairs_within", "Group", "ScanResult", "find_groups",
    "Fingerprint", "ahash", "dhash", "fingerprint", "hamming", "phash", "__version__",
]
