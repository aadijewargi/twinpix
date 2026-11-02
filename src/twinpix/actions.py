"""Safe file actions: move duplicates to a review folder, and undo that move.

twinpix never deletes anything. ``move_duplicates`` relocates every
non-keeper into a review folder (mirroring the original sub-folders) and
writes a JSON manifest; ``undo`` reads the manifest and puts every file back.
Once you have checked the review folder you can delete it yourself.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .model import Group

MANIFEST_PREFIX = "twinpix-manifest-"


@dataclass
class MovePlan:
    src: Path
    dst: Path
    size: int


def _unique(dst: Path, taken: set[Path]) -> Path:
    if not dst.exists() and dst not in taken:
        return dst
    stem, suffix = dst.stem, dst.suffix
    n = 1
    while True:
        cand = dst.with_name(f"{stem} ({n}){suffix}")
        if not cand.exists() and cand not in taken:
            return cand
        n += 1


def plan_moves(groups: Sequence[Group], root: Path, review: Path) -> list[MovePlan]:
    root = Path(root).resolve()
    review = Path(review).resolve()
    plans, taken = [], set()
    for g in groups:
        for p in g.duplicates:
            src = Path(p.path).resolve()
            try:
                rel = src.relative_to(root)
            except ValueError:  # outside the root (should not happen): keep the name
                rel = Path(src.name)
            dst = _unique(review / rel, taken)
            taken.add(dst)
            plans.append(MovePlan(src, dst, p.size))
    return plans


def move_duplicates(
    groups: Sequence[Group], root: Path, review: Path, dry_run: bool = False
) -> tuple[list[MovePlan], Path | None]:
    """Move duplicates into ``review``. Returns the moves and the manifest path."""
    plans = plan_moves(groups, root, review)
    if dry_run or not plans:
        return plans, None
    review = Path(review).resolve()
    review.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    manifest = _unique(review / f"{MANIFEST_PREFIX}{stamp}.json", set())
    done: list[dict] = []
    try:
        for plan in plans:
            plan.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(plan.src), str(plan.dst))
            done.append({"from": str(plan.src), "to": str(plan.dst), "size": plan.size})
    finally:
        # Written even if a move fails half-way, so everything moved can be undone.
        manifest.write_text(
            json.dumps({"tool": "twinpix", "created": stamp, "moves": done}, indent=2),
            encoding="utf-8",
        )
    return plans, manifest


def undo(manifest: Path) -> tuple[int, list[str]]:
    """Restore every file listed in a manifest. Returns (restored, problems)."""
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    restored, problems = 0, []
    for move in reversed(data.get("moves", [])):
        src, dst = Path(move["from"]), Path(move["to"])
        if not dst.exists():
            problems.append(f"missing from review folder: {dst}")
            continue
        if src.exists():
            problems.append(f"not restored, original path is occupied: {src}")
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(src))
        restored += 1
        _prune_empty_dirs(dst.parent, stop=manifest.parent)
    if not problems:
        manifest.rename(manifest.with_suffix(".undone.json"))
    return restored, problems


def _prune_empty_dirs(path: Path, stop: Path) -> None:
    path, stop = path.resolve(), stop.resolve()
    while path != stop and stop in path.parents:
        try:
            os.rmdir(path)
        except OSError:
            return
        path = path.parent
