"""Command-line interface.

    twinpix scan  <folder>  [--threshold 10] [--algo phash|dhash] [--report out.html] [--json out.json]
    twinpix move  <folder>  --to <review-folder> [--dry-run]
    twinpix undo  <manifest.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .actions import move_duplicates, undo
from .cache import HashCache
from .report import human_bytes, write_report
from .scanner import DEFAULT_MAX_COLOUR, ScanResult, scan


def _progress(done: int, total: int) -> None:
    if not sys.stderr.isatty() or total == 0:
        return
    width = 30
    filled = int(width * done / total)
    sys.stderr.write(f"\r  hashing [{'#' * filled}{'.' * (width - filled)}] {done}/{total}")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _add_scan_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("folder", type=Path, help="folder to scan (recursively)")
    p.add_argument("-t", "--threshold", type=int, default=10,
                   help="max Hamming distance (0-64) to count as a duplicate (default 10)")
    p.add_argument("--algo", choices=["phash", "dhash"], default="phash",
                   help="perceptual hash to compare (default phash)")
    p.add_argument("--index", choices=["auto", "mih", "brute", "bktree"], default="auto",
                   help="pair-search strategy: multi-index hashing, NumPy brute force or BK-tree (default auto)")
    p.add_argument("--no-colour-check", dest="colour_check", action="store_false",
                   help="don't reject hash matches whose colours differ")
    p.add_argument("-j", "--workers", type=int, default=None,
                   help="hashing processes (default: up to 8)")
    p.add_argument("--no-cache", action="store_true", help="don't read or write the hash cache")


def _run_scan(args: argparse.Namespace, exclude: Sequence[Path] = ()) -> tuple[ScanResult, float]:
    if not 0 <= args.threshold <= 64:
        raise SystemExit("error: --threshold must be between 0 and 64")
    t0 = time.perf_counter()
    cache = None if args.no_cache else HashCache()
    try:
        result = scan(args.folder, threshold=args.threshold, algo=args.algo, method=args.index,
                      max_colour=DEFAULT_MAX_COLOUR if args.colour_check else None,
                      workers=args.workers, cache=cache, exclude=exclude, progress=_progress)
    finally:
        if cache:
            cache.close()
    return result, time.perf_counter() - t0


def _summary(result: ScanResult, elapsed: float) -> str:
    lines = [
        f"  scanned      {len(result.photos):>8,} photos in {elapsed:.1f} s",
        f"  groups       {len(result.groups):>8,}",
        f"  duplicates   {result.duplicate_count:>8,}",
        f"  reclaimable  {human_bytes(result.reclaimable_bytes):>8}",
    ]
    if result.errors:
        lines.append(f"  unreadable   {len(result.errors):>8,} (listed in the report)")
    return "\n".join(lines)


def _rel(path: str, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def _print_groups(result: ScanResult, root: Path, limit: int = 10) -> None:
    root = Path(root).resolve()
    for n, g in enumerate(result.groups[:limit], 1):
        tag = "identical" if g.identical else f"d<={g.max_distance}"
        print(f"\n  group {n}  ({len(g.members)} photos, {tag})")
        print(f"    keep  {_rel(g.keeper.path, root)}  [{g.keeper.width}x{g.keeper.height}]")
        for p in g.duplicates:
            print(f"    dupe  {_rel(p.path, root)}  [{p.width}x{p.height}, d={g.distances[p.path]}]")
    if len(result.groups) > limit:
        print(f"\n  ... and {len(result.groups) - limit} more groups (see the report)")


def cmd_scan(args: argparse.Namespace) -> int:
    result, elapsed = _run_scan(args)
    print(_summary(result, elapsed))
    if not args.quiet:
        _print_groups(result, args.folder)
    if args.json:
        payload = {
            "root": str(Path(args.folder).resolve()), "threshold": args.threshold, "algo": args.algo,
            "groups": [g.to_dict() for g in result.groups],
            "errors": [{"path": p.path, "error": p.error} for p in result.errors],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  JSON written to {args.json}")
    if args.report:
        out = write_report(result, args.folder, args.report, args.threshold, args.algo, elapsed)
        print(f"\n  report written to {out.resolve()}")
    return 0


def cmd_move(args: argparse.Namespace) -> int:
    review = args.to.resolve()
    result, elapsed = _run_scan(args, exclude=[review])
    print(_summary(result, elapsed))
    plans, manifest = move_duplicates(result.groups, args.folder, review, dry_run=args.dry_run)
    if not plans:
        print("\n  nothing to move")
        return 0
    verb = "would move" if args.dry_run else "moved"
    for plan in plans[:20]:
        print(f"    {verb}  {plan.src}  ->  {plan.dst}")
    if len(plans) > 20:
        print(f"    ... and {len(plans) - 20} more")
    print(f"\n  {verb} {len(plans)} duplicates ({human_bytes(sum(p.size for p in plans))}) to {review}")
    if manifest:
        print(f"  undo with:  twinpix undo \"{manifest}\"")
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    restored, problems = undo(args.manifest)
    print(f"  restored {restored} files")
    for p in problems:
        print(f"  ! {p}")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twinpix",
        description="Find near-duplicate photos: bursts, resized copies and re-saved files.",
    )
    parser.add_argument("--version", action="version", version=f"twinpix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="find duplicates and write a report (changes nothing)")
    _add_scan_options(p)
    p.add_argument("-r", "--report", type=Path, default=Path("twinpix-report.html"),
                   help="HTML report path (default twinpix-report.html)")
    p.add_argument("--no-report", dest="report", action="store_const", const=None)
    p.add_argument("--json", type=Path, help="also write machine-readable results")
    p.add_argument("-q", "--quiet", action="store_true", help="only print the summary")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("move", help="move duplicates (never the best copy) to a review folder")
    _add_scan_options(p)
    p.add_argument("--to", type=Path, required=True, help="review folder to move duplicates into")
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would move")
    p.set_defaults(func=cmd_move)

    p = sub.add_parser("undo", help="put back everything a previous move relocated")
    p.add_argument("manifest", type=Path, help="twinpix-manifest-*.json from the review folder")
    p.set_defaults(func=cmd_undo)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # don't crash on odd file names in a legacy Windows console
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"error: not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
