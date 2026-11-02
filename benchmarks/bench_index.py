"""Benchmark all-pairs search: BK-tree vs vectorised brute force vs naive Python.

    python benchmarks/bench_index.py --out docs/img/benchmark.png

Hashes are synthetic but shaped like a real library: most photos are unique
(uniform random 64-bit hashes) and ~30 % belong to small clusters of 2-6
near-copies (a few flipped bits each). Distance computations are counted as
well as wall time, because the count is what the BK-tree actually saves.
"""

from __future__ import annotations

import argparse
import random
import time

from twinpix.index import bktree_pairs, brute_force_pairs


def library_hashes(n: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    out: list[int] = []
    while len(out) < n:
        base = rng.getrandbits(64)
        out.append(base)
        if rng.random() < 0.15:
            for _ in range(rng.randint(1, 5)):
                h = base
                for _ in range(rng.randint(0, 6)):
                    h ^= 1 << rng.randrange(64)
                out.append(h)
    rng.shuffle(out)
    return out[:n]


def naive(hashes: list[int], r: int) -> int:
    count = 0
    for i in range(len(hashes)):
        hi = hashes[i]
        for j in range(i + 1, len(hashes)):
            if (hi ^ hashes[j]).bit_count() <= r:
                count += 1
    return count


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return time.perf_counter() - t0, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,2000,5000,10000,20000,50000,100000")
    ap.add_argument("--radii", default="6,10")
    ap.add_argument("--naive-max", type=int, default=5000)
    ap.add_argument("--out", default=None, help="write a PNG chart here")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    radii = [int(r) for r in args.radii.split(",")]

    rows = []
    print("| n | radius | pairs | BK-tree | distance calls (% of n²/2) | NumPy brute force | naive Python |")
    print("|--:|--:|--:|--:|--:|--:|--:|")
    for r in radii:
        for n in sizes:
            hashes = library_hashes(n)
            t_bk, (pairs, calls) = timed(bktree_pairs, hashes, r)
            t_bf, pairs_bf = timed(brute_force_pairs, hashes, r)
            assert pairs == pairs_bf, "BK-tree and brute force disagree"
            t_nv = timed(naive, hashes, r)[0] if n <= args.naive_max else None
            frac = calls / (n * (n - 1) / 2)
            rows.append((r, n, t_bk, t_bf, t_nv, frac))
            nv = f"{t_nv:.2f} s" if t_nv is not None else "-"
            print(f"| {n:,} | {r} | {len(pairs):,} | {t_bk:.2f} s | {calls:,} ({100 * frac:.1f} %) "
                  f"| {t_bf:.2f} s | {nv} |", flush=True)

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
        colors = {"bk": "#3367d6", "bf": "#e8710a", "nv": "#9aa0a6"}
        for r, style in zip(radii, ["-", "--"]):
            sub = [row for row in rows if row[0] == r]
            ns = [row[1] for row in sub]
            axes[0].plot(ns, [row[2] for row in sub], style, marker="o", color=colors["bk"],
                         label=f"BK-tree (r={r})")
            axes[0].plot(ns, [row[3] for row in sub], style, marker="s", color=colors["bf"],
                         label=f"NumPy brute force (r={r})")
            axes[1].plot(ns, [100 * row[5] for row in sub], style, marker="o", color=colors["bk"],
                         label=f"r = {r}")
        nv = [(row[1], row[4]) for row in rows if row[0] == radii[0] and row[4] is not None]
        if nv:
            axes[0].plot(*zip(*nv), ":", marker="^", color=colors["nv"], label="naive Python loop")
        axes[0].set(xscale="log", yscale="log", xlabel="photos in library",
                    ylabel="seconds to find all pairs", title="All-pairs search time")
        axes[1].set(xscale="log", xlabel="photos in library",
                    ylabel="% of the n(n-1)/2 comparisons", title="BK-tree distance computations")
        axes[1].set_ylim(0, 100)
        for ax in axes:
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.out)
        print(f"\nchart written to {args.out}")


if __name__ == "__main__":
    main()
