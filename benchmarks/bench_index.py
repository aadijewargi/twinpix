"""Benchmark all-pairs search: multi-index hashing vs BK-tree vs brute force.

    python benchmarks/bench_index.py --out docs/img/benchmark.png

Hashes are synthetic but shaped like a real library: most photos are unique
(uniform random 64-bit hashes) and ~30 % belong to small clusters of 2-6
near-copies (a few flipped bits each). Distance computations are counted as
well as wall time, because the count is what an index actually saves.
"""

from __future__ import annotations

import argparse
import random
import time

from twinpix.index import bktree_pairs, brute_force_pairs, mih_pairs


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
    ap.add_argument("--bktree-max", type=int, default=20000)
    ap.add_argument("--out", default=None, help="write a PNG chart here")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    radii = [int(r) for r in args.radii.split(",")]

    rows = []
    print("| n | radius | pairs | multi-index hashing | BK-tree | NumPy brute force | naive Python |")
    print("|--:|--:|--:|--:|--:|--:|--:|")
    for r in radii:
        for n in sizes:
            hashes = library_hashes(n)
            half = n * (n - 1) / 2
            t_mih, (pairs, checked) = timed(mih_pairs, hashes, r)
            t_bf, pairs_bf = timed(brute_force_pairs, hashes, r)
            assert pairs == pairs_bf, "MIH and brute force disagree"
            if n <= args.bktree_max:
                t_bk, (pairs_bk, calls) = timed(bktree_pairs, hashes, r)
                assert pairs_bk == pairs_bf, "BK-tree and brute force disagree"
            else:
                t_bk, calls = None, None
            t_nv = timed(naive, hashes, r)[0] if n <= args.naive_max else None
            rows.append(dict(r=r, n=n, mih=t_mih, bk=t_bk, bf=t_bf, nv=t_nv,
                             mih_frac=checked / half, bk_frac=calls / half if calls else None))
            bk = f"{t_bk:.2f} s ({100 * calls / half:.0f} %)" if t_bk is not None else "-"
            nv = f"{t_nv:.2f} s" if t_nv is not None else "-"
            print(f"| {n:,} | {r} | {len(pairs):,} | {t_mih:.3f} s ({100 * checked / half:.2f} %) "
                  f"| {bk} | {t_bf:.2f} s | {nv} |", flush=True)

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
        series = [("mih", "multi-index hashing", "#1a73e8", "o"),
                  ("bf", "NumPy brute force", "#e8710a", "s"),
                  ("bk", "BK-tree", "#9334e6", "D"),
                  ("nv", "naive Python loop", "#9aa0a6", "^")]
        for r, style in zip(radii, ["-", "--"]):
            sub = [row for row in rows if row["r"] == r]
            for key, label, color, marker in series:
                pts = [(row["n"], row[key]) for row in sub if row[key] is not None]
                if pts and (key != "nv" or r == radii[0]):
                    axes[0].plot(*zip(*pts), style, marker=marker, color=color, ms=4,
                                 label=f"{label} (r={r})" if key != "nv" else label)
            for key, label, color, marker in series[:3:2]:
                frac = key + "_frac"
                pts = [(row["n"], 100 * row[frac]) for row in sub if row.get(frac) is not None]
                axes[1].plot(*zip(*pts), style, marker=marker, color=color, ms=4,
                             label=f"{label} (r={r})")
        axes[1].axhline(100, color="#e8710a", lw=1, alpha=0.6)
        axes[1].text(sizes[0], 70, "brute force = 100 %", color="#e8710a", fontsize=8)
        axes[0].set(xscale="log", yscale="log", xlabel="photos in library",
                    ylabel="seconds to find all pairs", title="All-pairs search time (lower is better)")
        axes[1].set(xscale="log", yscale="log", xlabel="photos in library",
                    ylabel="% of the n(n-1)/2 pairs compared", title="Comparisons actually made")
        fig.tight_layout()
        fig.savefig(args.out)
        print(f"\nchart written to {args.out}")


if __name__ == "__main__":
    main()
