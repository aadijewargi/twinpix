"""twinpix: find near-duplicate photos with perceptual hashing and a BK-tree."""

from .hashing import dhash, hamming, phash
from .index import BKTree, brute_force_pairs, bktree_pairs
from .model import Group, Photo
from .scanner import ScanResult, group_photos, scan

__version__ = "0.1.0"

__all__ = [
    "BKTree", "Group", "Photo", "ScanResult", "bktree_pairs", "brute_force_pairs",
    "dhash", "group_photos", "hamming", "phash", "scan", "__version__",
]
