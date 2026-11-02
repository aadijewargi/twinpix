"""Measure grouping accuracy on the synthetic library against its ground truth.

    python -m twinpix.demo demo_photos --scenes 120
    python benchmarks/accuracy.py demo_photos

Reports pairwise precision/recall (a "pair" is two files that belong to the
same scene) for each hash algorithm over a sweep of thresholds.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from twinpix.scanner import fill_digests, group_photos, hash_files, iter_images


def pair_scores(truth: dict[str, list[str]], root: Path, groups) -> tuple[float, float, int, int]:
    label = {str((root / p).resolve()): k for k, files in truth.items() for p in files}
    pred = {p.path: gi for gi, g in enumerate(groups) for p in g.members}
    tp = fp = fn = 0
    for a, b in itertools.combinations(label, 2):
        same_truth = label[a] == label[b]
        same_pred = a in pred and b in pred and pred[a] == pred[b]
        tp += same_truth and same_pred
        fp += same_pred and not same_truth
        fn += same_truth and not same_pred
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return precision, recall, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--thresholds", default="4,6,8,10,12,14,16")
    args = ap.parse_args()
    root = args.folder.resolve()
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    photos = hash_files(list(iter_images(root)))
    fill_digests(photos)
    print(f"{len(photos)} photos, {len(truth)} scenes\n")
    print("| algo  | threshold | precision | recall | false pairs | missed pairs |")
    print("|-------|----------:|----------:|-------:|------------:|-------------:|")
    for algo in ("phash", "dhash"):
        for t in map(int, args.thresholds.split(",")):
            groups = group_photos(photos, threshold=t, algo=algo)
            p, r, fp, fn = pair_scores(truth, root, groups)
            print(f"| {algo} | {t:>9} | {p:>9.3f} | {r:>6.3f} | {fp:>11} | {fn:>12} |")


if __name__ == "__main__":
    main()
