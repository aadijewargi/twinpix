"""Command-line interface: ``twinshot scan | apply | undo | hash``."""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

from . import __version__
from .grouping import DEFAULT_DHASH_RADIUS, DEFAULT_PHASH_RADIUS, ScanResult, find_groups
from .hashing import fingerprint
from .plan import PlanError, apply_plan, build_plan, load_plan, save_plan, undo
from .report import write_report
from .scanner import FingerprintCache, default_cache_path, fingerprint_all, iter_image_files


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _progress_printer(stream=sys.stderr):
    interactive = stream.isatty()
    start = time.perf_counter()

    def show(done: int, total: int) -> None:
        if not interactive or total == 0:
            return
        rate = done / max(time.perf_counter() - start, 1e-9)
        bar = "#" * int(24 * done / total)
        stream.write(f"\r  hashing [{bar:<24}] {done}/{total}  {rate:5.0f} img/s ")
        if done == total:
            stream.write("\n")
        stream.flush()

    return show


def cmd_scan(args: argparse.Namespace) -> int:
    for p in args.paths:
        if not os.path.exists(p):
            print(f"twinshot: no such file or folder: {p}", file=sys.stderr)
            return 2
    t0 = time.perf_counter()
    paths = list(iter_image_files(args.paths, recursive=not args.no_recursive))
    print(f"Found {len(paths)} images in {', '.join(args.paths)}")
    cache = None if args.no_cache else FingerprintCache(args.cache or default_cache_path())
    try:
        fps, skipped = fingerprint_all(paths, jobs=args.jobs, cache=cache, progress=_progress_printer())
    finally:
        if cache is not None:
            cache.close()
    t1 = time.perf_counter()
    groups = find_groups(fps, args.threshold, args.dhash_threshold, linkage=args.linkage, method=args.search)
    t2 = time.perf_counter()
    result = ScanResult(groups=groups, scanned=len(fps), skipped=skipped)

    settings = {
        "phash_radius": args.threshold,
        "dhash_radius": args.dhash_threshold,
        "linkage": args.linkage,
    }
    plan = build_plan(result, args.paths, settings)
    save_plan(plan, args.plan)
    write_report(result, plan, args.paths, args.report)

    print(f"Hashed {len(fps)} photos in {t1 - t0:.1f}s, grouped in {(t2 - t1) * 1000:.0f} ms")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s), e.g. {_pretty(skipped[0][0])} ({skipped[0][1]})")
    exact = sum(1 for g in groups if g.kind == "exact")
    print(
        f"{len(groups)} duplicate group(s) ({exact} exact, {len(groups) - exact} similar): "
        f"{result.duplicates} photo(s), {human_bytes(result.reclaimable_bytes)} reclaimable"
    )
    for g in groups[: args.top]:
        print(f"  #{g.id:<3} keep {os.path.basename(g.keeper.path)}  ({g.reason})")
        for m in g.members[1:]:
            tag = "identical" if m.exact_copy else f"dp={m.phash_dist:<2} dd={m.dhash_dist}"
            print(f"         move {os.path.basename(m.fp.path)}  [{tag}]")
    if len(groups) > args.top:
        print(f"  ... and {len(groups) - args.top} more (see the report)")
    print(f"\nReport: {os.path.abspath(args.report)}")
    print(f"Plan:   {os.path.abspath(args.plan)}")
    print("Review the report, then run:  twinshot apply " + _quote(args.plan))
    if args.open:
        webbrowser.open(Path(os.path.abspath(args.report)).as_uri())
    return 0


def _pretty(p: str) -> str:
    """Path relative to the current folder when it is inside it."""
    try:
        rel = os.path.relpath(p)
    except ValueError:  # other drive on Windows
        return p
    return p if rel.startswith("..") else rel


def _quote(p: str) -> str:
    return f'"{p}"' if " " in p else p


def cmd_apply(args: argparse.Namespace) -> int:
    try:
        plan = load_plan(args.plan)
    except (OSError, ValueError, PlanError) as exc:
        print(f"twinshot: cannot use plan: {exc}", file=sys.stderr)
        return 2
    n = sum(len(g["move"]) for g in plan["groups"])
    if n == 0:
        print("Plan has nothing to move.")
        return 0
    if not args.dry_run and not args.yes:
        reply = input(f"Move {n} photo(s) into quarantine? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1
    log = print if args.verbose or args.dry_run else (lambda s: None)
    res = apply_plan(plan, quarantine_dir=args.quarantine, dry_run=args.dry_run, log=log)
    verb = "Would move" if args.dry_run else "Moved"
    print(f"{verb} {len(res.moved)} photo(s), {human_bytes(res.bytes_moved)}")
    for path, why in res.skipped:
        print(f"  skipped {path}: {why}")
    if res.manifest:
        print(f"Quarantine manifest: {res.manifest}")
        print("Changed your mind?  twinshot undo " + _quote(res.manifest))
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    try:
        res = undo(args.manifest, dry_run=args.dry_run, log=print if args.verbose else (lambda s: None))
    except OSError as exc:
        print(f"twinshot: cannot read manifest: {exc}", file=sys.stderr)
        return 2
    print(f"{'Would restore' if args.dry_run else 'Restored'} {len(res.moved)} photo(s)")
    for path, why in res.skipped:
        print(f"  skipped {path}: {why}")
    return 0 if not res.skipped else 1


def cmd_hash(args: argparse.Namespace) -> int:
    rc = 0
    for p in args.files:
        try:
            f = fingerprint(p)
            print(f"{f.phash:016x}  {f.dhash:016x}  {f.width}x{f.height}  sharp={f.sharpness:8.1f}  {p}")
        except Exception as exc:
            print(f"error: {p}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="twinshot",
        description="Find near-duplicate photos, review them, and quarantine the rejects safely.",
    )
    p.add_argument("--version", action="version", version=f"twinshot {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="find duplicate groups and write a report + plan")
    s.add_argument("paths", nargs="+", help="folders (or files) to scan")
    s.add_argument("-t", "--threshold", type=int, default=DEFAULT_PHASH_RADIUS,
                   help=f"max pHash Hamming distance (default {DEFAULT_PHASH_RADIUS}; 0 = exact, higher = looser)")
    s.add_argument("--dhash-threshold", type=int, default=DEFAULT_DHASH_RADIUS,
                   help=f"max dHash distance to confirm a match (default {DEFAULT_DHASH_RADIUS})")
    s.add_argument("--linkage", choices=["complete", "single"], default="complete",
                   help="complete: every pair in a group must match (default); single: chain matches")
    s.add_argument("--search", choices=["auto", "mih", "brute", "bktree"], default="auto",
                   help="pair-search engine (default auto: brute force for small sets, multi-index hashing above)")
    s.add_argument("--report", default="twinshot-report.html", help="HTML report path")
    s.add_argument("--plan", default="twinshot-plan.json", help="plan JSON path")
    s.add_argument("-j", "--jobs", type=int, default=None, help="worker processes (default: CPUs - 1)")
    s.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    s.add_argument("--no-cache", action="store_true", help="ignore and do not update the fingerprint cache")
    s.add_argument("--cache", help="cache file (default: per-user cache folder)")
    s.add_argument("--top", type=int, default=10, help="groups to list in the terminal")
    s.add_argument("--open", action="store_true", help="open the report in your browser")
    s.set_defaults(func=cmd_scan)

    a = sub.add_parser("apply", help="move photos marked MOVE into a quarantine folder")
    a.add_argument("plan", help="plan JSON from 'scan' or downloaded from the report")
    a.add_argument("-q", "--quarantine", help="quarantine folder (default: <scan root>/twinshot-quarantine/<timestamp>)")
    a.add_argument("-n", "--dry-run", action="store_true", help="show what would happen")
    a.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=cmd_apply)

    u = sub.add_parser("undo", help="restore photos from a quarantine manifest")
    u.add_argument("manifest", help="manifest.jsonl (or the quarantine folder containing it)")
    u.add_argument("-n", "--dry-run", action="store_true")
    u.add_argument("-v", "--verbose", action="store_true")
    u.set_defaults(func=cmd_undo)

    h = sub.add_parser("hash", help="print the perceptual hashes of files")
    h.add_argument("files", nargs="+")
    h.set_defaults(func=cmd_hash)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    if sys.platform == "win32":  # make unicode filenames print in old consoles
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
            except Exception:
                pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
