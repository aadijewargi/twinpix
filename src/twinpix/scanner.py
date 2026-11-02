"""Walking a folder, hashing photos in parallel and grouping near-duplicates."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from PIL import Image

from .cache import HashCache
from .hashing import dhash, hamming, open_image, phash, sharpness
from .index import connected_groups, find_pairs
from .model import Group, Photo
from .quality import pick_keeper

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff",
}

try:  # optional iPhone photo support
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    IMAGE_EXTENSIONS |= {".heic", ".heif"}
except ImportError:  # pragma: no cover - depends on optional extra
    pass

# Decompression-bomb protection stays on, but allow large panoramas.
Image.MAX_IMAGE_PIXELS = 400_000_000

ProgressFn = Callable[[int, int], None]


def iter_images(root: Path, exclude: Sequence[Path] = ()) -> Iterable[Path]:
    """Yield image files under ``root`` in a stable order, skipping ``exclude``."""
    excluded = {Path(p).resolve() for p in exclude}
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and (here / d).resolve() not in excluded
        )
        for name in sorted(filenames):
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS and not name.startswith("._"):
                yield here / name


def hash_file(path: str) -> Photo:
    """Hash one file. Never raises: failures are recorded in ``Photo.error``."""
    try:
        st = os.stat(path)
    except OSError as exc:
        return Photo(path, 0, 0, error=f"stat failed: {exc}")
    photo = Photo(path, st.st_size, st.st_mtime_ns)
    try:
        img, (w, h) = open_image(path)
        photo.width, photo.height = w, h
        photo.dhash = dhash(img)
        photo.phash = phash(img)
        photo.sharpness = sharpness(img)
    except Exception as exc:  # corrupt, truncated or unsupported file
        photo.error = f"{type(exc).__name__}: {exc}"
    return photo


def hash_files(
    paths: Sequence[Path],
    workers: Optional[int] = None,
    cache: Optional[HashCache] = None,
    progress: Optional[ProgressFn] = None,
) -> list[Photo]:
    """Hash many files, reusing cached results and using a process pool."""
    results: dict[str, Photo] = {}
    todo: list[str] = []
    for p in paths:
        key = str(Path(p).resolve())
        cached = None
        if cache is not None:
            try:
                st = os.stat(key)
                cached = cache.get(key, st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        if cached is not None:
            results[key] = cached
        else:
            todo.append(key)

    total, done = len(paths), len(results)
    if progress:
        progress(done, total)

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if workers <= 1 or len(todo) < 16:
        fresh = []
        for key in todo:
            fresh.append(hash_file(key))
            done += 1
            if progress:
                progress(done, total)
    else:
        fresh = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for photo in pool.map(hash_file, todo, chunksize=8):
                fresh.append(photo)
                done += 1
                if progress:
                    progress(done, total)

    for photo in fresh:
        results[photo.path] = photo
    if cache is not None:
        cache.put_many(fresh)
    return [results[str(Path(p).resolve())] for p in paths]


def file_digest(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fill_digests(photos: Sequence[Photo]) -> None:
    """Content-hash only files that share a byte size with another file.

    Two files can only be byte-identical if their sizes match, so this skips
    reading the vast majority of a library.
    """
    by_size: dict[int, list[Photo]] = {}
    for p in photos:
        by_size.setdefault(p.size, []).append(p)
    for same in by_size.values():
        if len(same) > 1:
            for p in same:
                try:
                    p.digest = file_digest(p.path)
                except OSError:
                    p.digest = None


@dataclass
class ScanResult:
    photos: list[Photo]
    groups: list[Group]
    errors: list[Photo] = field(default_factory=list)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(g.reclaimable_bytes for g in self.groups)

    @property
    def duplicate_count(self) -> int:
        return sum(len(g.duplicates) for g in self.groups)


def group_photos(
    photos: Sequence[Photo], threshold: int = 10, algo: str = "phash", method: str = "auto"
) -> list[Group]:
    """Group photos whose hashes are within ``threshold`` bits of each other.

    Groups are connected components of the "within threshold" graph, so a
    chain A~B~C lands in one group even if A and C are further apart; the
    reported ``max_distance`` makes that visible.
    """
    if algo not in ("phash", "dhash"):
        raise ValueError("algo must be 'phash' or 'dhash'")
    ok = [p for p in photos if p.ok]
    hashes = [getattr(p, algo) for p in ok]
    pairs = find_pairs(hashes, threshold, method=method)
    edges = [(i, j) for i, j, _ in pairs]

    # Byte-identical files always belong together, whatever the threshold.
    by_digest: dict[str, int] = {}
    for idx, p in enumerate(ok):
        if p.digest:
            first = by_digest.setdefault(p.digest, idx)
            if first != idx:
                edges.append((first, idx))

    groups = []
    for comp in connected_groups(len(ok), edges):
        members = [ok[i] for i in comp]
        keeper = pick_keeper(members)
        kh = getattr(keeper, algo)
        distances = {p.path: hamming(kh, getattr(p, algo)) for p in members}
        identical = members[0].digest is not None and all(
            p.digest == members[0].digest for p in members
        )
        groups.append(
            Group(members, keeper, max(distances.values()), identical, distances)
        )
    return groups


def scan(
    root: Path,
    threshold: int = 10,
    algo: str = "phash",
    method: str = "auto",
    workers: Optional[int] = None,
    cache: Optional[HashCache] = None,
    exclude: Sequence[Path] = (),
    progress: Optional[ProgressFn] = None,
) -> ScanResult:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    paths = list(iter_images(root, exclude=exclude))
    photos = hash_files(paths, workers=workers, cache=cache, progress=progress)
    fill_digests([p for p in photos if p.ok])
    groups = group_photos(photos, threshold=threshold, algo=algo, method=method)
    errors = [p for p in photos if not p.ok]
    return ScanResult(photos, groups, errors)
