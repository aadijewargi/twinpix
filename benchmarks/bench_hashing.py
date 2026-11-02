"""How much does reduced-scale JPEG decoding (``Image.draft``) save?

    python benchmarks/bench_hashing.py demo_photos

Hashes every JPEG in a folder twice, single-process, with and without
draft mode, and checks that the resulting hashes stay within a few bits.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image, ImageOps

from twinpix.hashing import hamming, phash
from twinpix.scanner import iter_images


def full_decode(path: Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path))
    img.load()
    return img


def draft_decode(path: Path) -> Image.Image:
    img = Image.open(path)
    img.draft("RGB", (512, 512))
    img = ImageOps.exif_transpose(img)
    img.load()
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    files = [p for p in iter_images(args.folder) if p.suffix.lower() in (".jpg", ".jpeg")]
    results = {}
    for name, fn in (("full decode", full_decode), ("draft decode", draft_decode)):
        t0 = time.perf_counter()
        results[name] = [phash(fn(p)) for p in files]
        dt = time.perf_counter() - t0
        print(f"{name:>13}: {dt:6.2f} s  ({len(files) / dt:6.1f} photos/s)")
    diffs = [hamming(a, b) for a, b in zip(results["full decode"], results["draft decode"])]
    print(f"hash difference full vs draft: mean {sum(diffs) / len(diffs):.2f} bits, max {max(diffs)}")


if __name__ == "__main__":
    main()
