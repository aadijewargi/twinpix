"""Walk folders, fingerprint photos in parallel, and cache the results."""

from __future__ import annotations

import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

from .hashing import ALGO_VERSION, Fingerprint, fingerprint

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff",
}
HEIF_EXTENSIONS = {".heic", ".heif"}
QUARANTINE_DIRNAME = "twinshot-quarantine"

try:  # optional: iPhone photos
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    IMAGE_EXTENSIONS |= HEIF_EXTENSIONS
except Exception:  # pragma: no cover - optional dependency
    pass


def iter_image_files(roots: Iterable[str], recursive: bool = True) -> Iterator[str]:
    """Yield image paths under ``roots`` in a stable order.

    Skips hidden folders, our own quarantine folder and anything we cannot
    read (permission errors on Windows system folders are common).
    """
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            if os.path.splitext(root)[1].lower() in IMAGE_EXTENSIONS and root not in seen:
                seen.add(root)
                yield root
            continue
        stack = [root]
        while stack:
            folder = stack.pop()
            try:
                with os.scandir(folder) as it:
                    entries = sorted(it, key=lambda e: e.name.lower())
            except OSError:
                continue
            subdirs = []
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if recursive and not e.name.startswith(".") and e.name != QUARANTINE_DIRNAME:
                            subdirs.append(e.path)
                    elif os.path.splitext(e.name)[1].lower() in IMAGE_EXTENSIONS:
                        p = os.path.abspath(e.path)
                        if p not in seen:
                            seen.add(p)
                            yield p
                except OSError:
                    continue
            stack.extend(reversed(subdirs))


def default_cache_path() -> str:
    """Per-user cache location (``%LOCALAPPDATA%\\twinshot`` on Windows)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "twinshot", "fingerprints.sqlite")


class FingerprintCache:
    """SQLite cache keyed on (path, size, mtime) so re-scans are near instant."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS fp_v1 (
                path TEXT PRIMARY KEY, bytes INTEGER, mtime REAL, version INTEGER,
                width INTEGER, height INTEGER, phash TEXT, dhash TEXT, sharpness REAL,
                colorfulness REAL)"""
        )

    def get(self, path: str, size: int, mtime: float) -> Optional[Fingerprint]:
        row = self.db.execute(
            "SELECT width, height, phash, dhash, sharpness, colorfulness FROM fp_v1 "
            "WHERE path=? AND bytes=? AND mtime=? AND version=?",
            (path, size, mtime, ALGO_VERSION),
        ).fetchone()
        if row is None:
            return None
        w, h, ph, dh, sh, col = row
        return Fingerprint(path, size, mtime, w, h, int(ph, 16), int(dh, 16), sh, col)

    def put_many(self, fps: Sequence[Fingerprint]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO fp_v1 VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (f.path, f.bytes, f.mtime, ALGO_VERSION, f.width, f.height,
                 f"{f.phash:016x}", f"{f.dhash:016x}", f.sharpness, f.colorfulness)
                for f in fps
            ],
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def _safe_fingerprint(path: str) -> Tuple[str, Optional[Fingerprint], Optional[str]]:
    try:
        return path, fingerprint(path), None
    except Exception as exc:  # corrupt/truncated/unsupported file
        return path, None, f"{type(exc).__name__}: {exc}"


ProgressFn = Callable[[int, int], None]


def fingerprint_all(
    paths: Sequence[str],
    jobs: Optional[int] = None,
    cache: Optional[FingerprintCache] = None,
    progress: Optional[ProgressFn] = None,
) -> Tuple[List[Fingerprint], List[Tuple[str, str]]]:
    """Fingerprint ``paths``; returns (fingerprints, [(path, error), ...]).

    ``jobs=1`` runs in-process (handy for debugging and tests); otherwise a
    process pool spreads the decode-heavy work across CPU cores.
    """
    results: List[Fingerprint] = []
    skipped: List[Tuple[str, str]] = []
    todo: List[str] = []
    for p in paths:
        hit = None
        if cache is not None:
            try:
                st = os.stat(p)
                hit = cache.get(p, st.st_size, st.st_mtime)
            except OSError as exc:
                skipped.append((p, f"{type(exc).__name__}: {exc}"))
                continue
        if hit is not None:
            results.append(hit)
        else:
            todo.append(p)

    total = len(paths)
    done = total - len(todo)
    if progress:
        progress(done, total)

    fresh: List[Fingerprint] = []
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 1)
    if jobs == 1 or len(todo) < 16:
        iterator: Iterable = map(_safe_fingerprint, todo)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        iterator = pool.map(_safe_fingerprint, todo, chunksize=8)
    try:
        for path, fp, err in iterator:
            done += 1
            if fp is not None:
                fresh.append(fp)
            else:
                skipped.append((path, err or "unknown error"))
            if progress and (done % 25 == 0 or done == total):
                progress(done, total)
    finally:
        if pool is not None:
            pool.shutdown()

    if cache is not None and fresh:
        cache.put_many(fresh)
    results.extend(fresh)
    results.sort(key=lambda f: f.path)
    return results, skipped
