"""twinpix: find near-duplicate photos with perceptual hashing and multi-index hashing."""

from .hashing import dhash, hamming, phash
from .index import BKTree, bktree_pairs, brute_force_pairs, find_pairs, mih_pairs
from .model import Group, Photo
from .scanner import ScanResult, group_photos, scan

__version__ = "0.1.0"

__all__ = [
    "BKTree", "Group", "Photo", "ScanResult", "bktree_pairs", "brute_force_pairs",
    "dhash", "find_pairs", "group_photos", "mih_pairs", "hamming", "phash", "scan", "__version__",
]
