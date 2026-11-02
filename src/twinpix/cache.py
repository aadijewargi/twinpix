"""SQLite cache of per-file hashes so that re-scans only touch new files.

A file is re-hashed only when its absolute path, size or modification time
changes. The cache lives in the user's cache directory, never inside the
photo folder being scanned.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Optional

from .model import Photo

SCHEMA_VERSION = 2


def default_cache_path() -> Path:
    override = os.environ.get("TWINPIX_CACHE")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "twinpix" / "hashes.sqlite"


class HashCache:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(photos)")]
        if cols and "colour" not in cols:  # cache from an older version: start over
            self.db.execute("DROP TABLE photos")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS photos (
                path TEXT PRIMARY KEY, version INTEGER, size INTEGER, mtime_ns INTEGER,
                width INTEGER, height INTEGER, dhash TEXT, phash TEXT, sharpness REAL,
                colour TEXT)"""
        )

    def get(self, path: str, size: int, mtime_ns: int) -> Optional[Photo]:
        row = self.db.execute(
            "SELECT width, height, dhash, phash, sharpness, colour FROM photos "
            "WHERE path=? AND version=? AND size=? AND mtime_ns=?",
            (path, SCHEMA_VERSION, size, mtime_ns),
        ).fetchone()
        if row is None:
            return None
        w, h, dh, ph, sharp, colour = row
        return Photo(path, size, mtime_ns, w, h, int(dh, 16), int(ph, 16), sharp,
                     bytes.fromhex(colour) if colour else None)

    def put_many(self, photos: Iterable[Photo]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO photos VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (p.path, SCHEMA_VERSION, p.size, p.mtime_ns, p.width, p.height,
                 f"{p.dhash:016x}", f"{p.phash:016x}", p.sharpness,
                 p.colour.hex() if p.colour else None)
                for p in photos
                if p.ok
            ],
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "HashCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
