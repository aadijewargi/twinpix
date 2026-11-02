"""Data types shared across twinpix."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Photo:
    path: str
    size: int  # bytes on disk
    mtime_ns: int
    width: int = 0
    height: int = 0
    dhash: int = 0
    phash: int = 0
    sharpness: float = 0.0
    error: Optional[str] = None
    digest: Optional[str] = None  # content hash, filled only for same-size files

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dhash"] = f"{self.dhash:016x}"
        d["phash"] = f"{self.phash:016x}"
        return d


@dataclass
class Group:
    members: list[Photo]
    keeper: Photo
    max_distance: int  # largest hash distance between any member and the keeper
    identical: bool = False  # every member has byte-identical content
    distances: dict[str, int] = field(default_factory=dict)  # path -> distance to keeper

    @property
    def duplicates(self) -> list[Photo]:
        return [p for p in self.members if p is not self.keeper]

    @property
    def reclaimable_bytes(self) -> int:
        return sum(p.size for p in self.duplicates)

    def to_dict(self) -> dict:
        return {
            "keeper": self.keeper.path,
            "identical": self.identical,
            "max_distance": self.max_distance,
            "reclaimable_bytes": self.reclaimable_bytes,
            "members": [
                {**p.to_dict(), "distance_to_keeper": self.distances.get(p.path, 0)}
                for p in self.members
            ],
        }
