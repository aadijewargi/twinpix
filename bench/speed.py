"""Speed benchmark: all-pairs search engines (MIH vs brute force vs BK-tree).

Synthetic library: random 64-bit hashes with 32 bits set (like real pHashes,
which threshold at the median), where 30% of photos have 1-3 near copies
(0-10 bit flips). Brute force is NumPy-vectorised so the baseline is fair;
the BK-tree is pure Python, so it is only run on small sizes.

Run:  python bench/speed.py      (writes assets/speed.png, bench/results/speed.json)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from twinshot.bktree import BKTree  # noqa: E402
from twinshot.search import as_uint64, choose_chunks, pairs_brute, pairs_mih  # noqa: E402

RADIUS = 12


def synthetic_hashes(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        bits = np.zeros(64, dtype=bool)
        bits[rng.choice(64, 32, replace=False)] = True
        h = int("".join("1" if b else "0" for b in bits), 2)
        out.append(h)
        if rng.random() < 0.3:
            for _ in range(rng.integers(1, 4)):
                flips = rng.choice(64, rng.integers(0, 11), replace=False)
                c = h
                for f in flips:
                    c ^= 1 << int(f)
                out.append(c)
    return out[:n]


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def main() -> None:
    sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    rows = []
    pairs_mih(as_uint64(synthetic_hashes(500)), RADIUS)  # warm-up: builds the flip-mask tables
    for n in sizes:
        h = as_uint64(synthetic_hashes(n))
        mih, t_mih = timed(pairs_mih, h, RADIUS)
        row = {"n": n, "pairs": int(len(mih[0])), "chunks": choose_chunks(n, RADIUS), "mih_s": t_mih,
               "brute_s": None, "bktree_s": None, "bktree_visit_frac": None}
        if n <= 50000:
            brute, row["brute_s"] = timed(pairs_brute, h, RADIUS)
            assert all((a == b).all() for a, b in zip(mih, brute)), "MIH and brute force disagree"
        if n <= 5000:
            tree = BKTree()
            t0 = time.perf_counter()
            for j, x in enumerate(h.tolist()):
                tree.query(x, RADIUS)
                tree.add(x, j)
            row["bktree_s"] = time.perf_counter() - t0
            row["bktree_visit_frac"] = tree.comparisons / (n * (n - 1) / 2)
        rows.append(row)
        fmt = lambda v: f"{v:7.2f}s" if v is not None else "      -"
        extra = f"  (BK-tree visits {row['bktree_visit_frac']:.0%} of pairs)" if row["bktree_visit_frac"] else ""
        print(f"N={n:7d}  pairs={row['pairs']:7d}  MIH(m={row['chunks']}) {fmt(t_mih)}  "
              f"brute {fmt(row['brute_s'])}  BK-tree {fmt(row['bktree_s'])}{extra}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "speed.json"), "w") as fh:
        json.dump({"radius": RADIUS, "rows": rows}, fh, indent=2)
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1d1d1f", "#6b6b70", "#e4e2dc"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": grid,
                         "xtick.color": muted, "ytick.color": muted, "axes.labelcolor": ink})
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    series = [("bktree_s", "BK-tree (pure Python)", "#9aa0a6"), ("brute_s", "brute force (NumPy, N\u00b2)", "#c2410c"),
              ("mih_s", "multi-index hashing (twinshot)", "#3b5bdb")]
    for key, label, color in series:
        pts = [(r["n"], r[key]) for r in rows if r[key] is not None]
        x, y = zip(*pts)
        ax.plot(x, y, marker="o", lw=2.2 if key == "mih_s" else 1.6, color=color, label=label)
    last = [r for r in rows if r["brute_s"] is not None][-1]
    ax.annotate(f"{last['brute_s'] / last['mih_s']:.1f}x faster\nat {last['n']:,} photos",
                (last["n"], last["mih_s"]), xytext=(10, -34), textcoords="offset points", color="#3b5bdb", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("photos in library")
    ax.set_ylabel("seconds to find all pairs")
    ax.set_title(f"Finding every pair within {RADIUS} bits", loc="left", color=ink, fontsize=11.5)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=grid, lw=.8, which="major")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "..", "assets", "speed.png")
    fig.savefig(out, facecolor="white")
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
