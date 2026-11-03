"""Plans: the reviewable, editable record of what to keep and what to move.

Nothing is ever deleted. ``apply`` moves rejects into a quarantine folder and
writes a JSON-lines manifest as it goes, so ``undo`` can put every file back
even if the run was interrupted halfway.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .grouping import ScanResult
from .scanner import QUARANTINE_DIRNAME

PLAN_VERSION = 1
MANIFEST_NAME = "manifest.jsonl"


class PlanError(ValueError):
    pass


def build_plan(result: ScanResult, roots: Sequence[str], settings: Optional[dict] = None) -> dict:
    groups = []
    for g in result.groups:
        groups.append(
            {
                "id": g.id,
                "kind": g.kind,
                "reason": g.reason,
                "keep": [g.members[0].fp.path],
                "move": [m.fp.path for m in g.members[1:]],
                "files": {
                    m.fp.path: {"bytes": m.fp.bytes, "mtime": m.fp.mtime} for m in g.members
                },
            }
        )
    return {
        "twinshot_plan": PLAN_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "roots": [os.path.abspath(r) for r in roots],
        "settings": settings or {},
        "groups": groups,
    }


def save_plan(plan: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)


def load_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("twinshot_plan") != PLAN_VERSION:
        raise PlanError("not a twinshot plan (or from an incompatible version)")
    if not isinstance(plan.get("groups"), list):
        raise PlanError("plan has no 'groups' list")
    seen_move = set()
    for g in plan["groups"]:
        keep, move = g.get("keep", []), g.get("move", [])
        if not isinstance(keep, list) or not isinstance(move, list):
            raise PlanError(f"group {g.get('id')}: 'keep' and 'move' must be lists")
        overlap = set(keep) & set(move)
        if overlap:
            raise PlanError(f"group {g.get('id')}: file both kept and moved: {sorted(overlap)[0]}")
        if move and not keep:
            raise PlanError(
                f"group {g.get('id')}: every photo is marked to move; keep at least one"
            )
        for p in move:
            if p in seen_move:
                raise PlanError(f"file listed to move twice: {p}")
            seen_move.add(p)


def _relative_destination(path: str, roots: Sequence[str]) -> str:
    """Where inside the quarantine folder ``path`` should go."""
    ap = os.path.abspath(path)
    multi = len(roots) > 1
    for r in roots:
        r = os.path.abspath(r)
        try:
            if os.path.commonpath([ap, r]) == r:
                rel = os.path.relpath(ap, r)
                return os.path.join(os.path.basename(r) or "root", rel) if multi else rel
        except ValueError:  # different drives on Windows
            continue
    drive, tail = os.path.splitdrive(ap)
    drive = drive.replace(":", "").replace("\\", "_").replace("/", "_").strip("_")
    return os.path.join("_outside", drive, tail.lstrip("\\/"))


def _unique(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem} ({n}){ext}"):
        n += 1
    return f"{stem} ({n}){ext}"


@dataclass
class ApplyResult:
    moved: List[Tuple[str, str]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    manifest: Optional[str] = None
    bytes_moved: int = 0


def default_quarantine_dir(plan: dict) -> str:
    roots = plan.get("roots") or [os.getcwd()]
    base = roots[0] if os.path.isdir(roots[0]) else os.path.dirname(roots[0])
    return os.path.join(base, QUARANTINE_DIRNAME, time.strftime("%Y%m%d-%H%M%S"))


def apply_plan(
    plan: dict,
    quarantine_dir: Optional[str] = None,
    dry_run: bool = False,
    log: Callable[[str], None] = lambda s: None,
) -> ApplyResult:
    validate_plan(plan)
    roots = plan.get("roots") or []
    qdir = os.path.abspath(quarantine_dir or default_quarantine_dir(plan))
    res = ApplyResult()
    manifest_fh = None
    try:
        for g in plan["groups"]:
            keep_existing = [p for p in g["keep"] if os.path.isfile(p)]
            if g["move"] and not keep_existing:
                for p in g["move"]:
                    res.skipped.append((p, "no kept photo of this group still exists"))
                log(f"group {g.get('id')}: skipped, none of its kept photos exist")
                continue
            info = g.get("files", {})
            for src in g["move"]:
                if not os.path.isfile(src):
                    res.skipped.append((src, "file no longer exists"))
                    continue
                st = os.stat(src)
                rec = info.get(src)
                if rec and (st.st_size != rec.get("bytes") or abs(st.st_mtime - rec.get("mtime", st.st_mtime)) > 1e-3):
                    res.skipped.append((src, "file changed since the scan"))
                    continue
                dst = _unique(os.path.join(qdir, _relative_destination(src, roots)))
                if dry_run:
                    log(f"would move {src} -> {dst}")
                    res.moved.append((src, dst))
                    res.bytes_moved += st.st_size
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if manifest_fh is None:
                    os.makedirs(qdir, exist_ok=True)
                    res.manifest = os.path.join(qdir, MANIFEST_NAME)
                    manifest_fh = open(res.manifest, "a", encoding="utf-8")
                shutil.move(src, dst)
                manifest_fh.write(json.dumps({"from": src, "to": dst}, ensure_ascii=False) + "\n")
                manifest_fh.flush()
                res.moved.append((src, dst))
                res.bytes_moved += st.st_size
                log(f"moved {src}")
    finally:
        if manifest_fh is not None:
            manifest_fh.close()
    return res


def undo(manifest_path: str, dry_run: bool = False, log: Callable[[str], None] = lambda s: None) -> ApplyResult:
    """Move every file listed in a manifest back to where it came from."""
    if os.path.isdir(manifest_path):
        manifest_path = os.path.join(manifest_path, MANIFEST_NAME)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    res = ApplyResult(manifest=manifest_path)
    remaining: List[Dict[str, str]] = []
    for e in reversed(entries):
        src, dst = e["to"], e["from"]
        if not os.path.isfile(src):
            res.skipped.append((src, "missing from quarantine"))
            continue
        if os.path.exists(dst):
            res.skipped.append((dst, "a file already exists at the original location"))
            remaining.append(e)
            continue
        if dry_run:
            res.moved.append((src, dst))
            continue
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.move(src, dst)
        res.moved.append((src, dst))
        log(f"restored {dst}")
    if not dry_run:
        # Rewrite the manifest with whatever could not be restored.
        with open(manifest_path, "w", encoding="utf-8") as fh:
            for e in reversed(remaining):
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return res
