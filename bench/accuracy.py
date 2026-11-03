"""Accuracy benchmark: how well do the hashes separate duplicates from distinct photos?

Run:  python bench/accuracy.py           (writes assets/accuracy.png and prints tables)
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from synth import HARD, POSITIVE, base_images, variants  # noqa: E402
from twinshot.grouping import DEFAULT_DHASH_RADIUS, DEFAULT_PHASH_RADIUS, find_groups  # noqa: E402
from twinshot.hashing import Fingerprint, ahash, dhash, hamming, phash, sharpness  # noqa: E402

HASHES = {"aHash": ahash, "dHash": dhash, "pHash": phash}


def main() -> None:
    bases = base_images()
    names = sorted(bases)
    print(f"{len(names)} base images x {len(POSITIVE) + len(HARD)} variants")

    H = {h: {} for h in HASHES}  # hash -> (base, variant or None) -> int
    fps = []
    truth = {}
    for b in names:
        img = bases[b]
        items = [(None, img)] + variants(img)
        for vname, vimg in items:
            for hname, fn in HASHES.items():
                H[hname][(b, vname)] = fn(vimg)
            # library simulation uses only "positive" variants
            if vname is None or vname in POSITIVE:
                fid = f"{b}|{vname or 'original'}"
                fps.append(Fingerprint(fid, 1, 0.0, *vimg.size,
                                       H["pHash"][(b, vname)], H["dHash"][(b, vname)], sharpness(vimg)))
                truth[fid] = b

    # --- pair distances -----------------------------------------------------
    pos = defaultdict(lambda: defaultdict(list))  # hash -> transform -> [d]
    for hname in HASHES:
        for b in names:
            orig = H[hname][(b, None)]
            for vname in list(POSITIVE) + list(HARD):
                pos[hname][vname].append(hamming(orig, H[hname][(b, vname)]))
    neg = {h: np.array([hamming(H[h][(a, None)], H[h][(b, None)])
                        for a, b in itertools.combinations(names, 2)]) for h in HASHES}
    combo_pos = {v: [] for v in list(POSITIVE) + list(HARD)}
    for b in names:
        for v in combo_pos:
            combo_pos[v].append((hamming(H["pHash"][(b, None)], H["pHash"][(b, v)]),
                                 hamming(H["dHash"][(b, None)], H["dHash"][(b, v)])))
    combo_neg = np.array([(hamming(H["pHash"][(a, None)], H["pHash"][(b, None)]),
                           hamming(H["dHash"][(a, None)], H["dHash"][(b, None)]))
                          for a, b in itertools.combinations(names, 2)])

    # --- threshold sweep: recall on edits vs false-match rate ---------------
    sweep = {}
    for h in HASHES:
        allpos = np.concatenate([pos[h][v] for v in POSITIVE])
        sweep[h] = [{"t": t, "recall": float((allpos <= t).mean()),
                     "false_match_rate": float((neg[h] <= t).mean())} for t in range(0, 25)]
    allc = np.concatenate([combo_pos[v] for v in POSITIVE])
    sweep["pHash+dHash"] = [
        {"t": t, "t_dhash": t + 2,
         "recall": float(((allc[:, 0] <= t) & (allc[:, 1] <= t + 2)).mean()),
         "false_match_rate": float(((combo_neg[:, 0] <= t) & (combo_neg[:, 1] <= t + 2)).mean())}
        for t in range(0, 25)
    ]
    print("\nThreshold sweep (pHash <= t, dHash <= t+2): t, recall, false-match rate")
    for row in sweep["pHash+dHash"][6:21:2]:
        print(f"  t={row['t']:2d}  recall {row['recall']:.1%}  FMR {row['false_match_rate']:.4%}")

    # --- per-transform recall at the defaults ------------------------------
    tp, td = DEFAULT_PHASH_RADIUS, DEFAULT_DHASH_RADIUS
    table = []
    for v in list(POSITIVE) + list(HARD):
        c = np.array(combo_pos[v])
        table.append({
            "variant": v,
            "hard": v in HARD,
            "pHash": float((np.array(pos["pHash"][v]) <= tp).mean()),
            "dHash": float((np.array(pos["dHash"][v]) <= td).mean()),
            "both": float(((c[:, 0] <= tp) & (c[:, 1] <= td)).mean()),
            "median_dp": float(np.median(pos["pHash"][v])),
        })
    fmr_p = float((neg["pHash"] <= tp).mean())
    fmr_both = float(((combo_neg[:, 0] <= tp) & (combo_neg[:, 1] <= td)).mean())

    # --- end-to-end library simulation -------------------------------------
    groups = find_groups(fps, check_exact=False)
    pred_pairs = set()
    for g in groups:
        ids = sorted(m.fp.path for m in g.members)
        pred_pairs.update(itertools.combinations(ids, 2))
    by_base = defaultdict(list)
    for fid, b in truth.items():
        by_base[b].append(fid)
    true_pairs = set()
    for ids in by_base.values():
        true_pairs.update(itertools.combinations(sorted(ids), 2))
    tp_pairs = len(pred_pairs & true_pairs)
    precision = tp_pairs / max(len(pred_pairs), 1)
    recall = tp_pairs / max(len(true_pairs), 1)
    pure = sum(1 for g in groups if len({truth[m.fp.path] for m in g.members}) == 1)

    print(f"\nDefaults: pHash <= {tp}, dHash <= {td}")
    print("| Variant | pHash | dHash | pHash + dHash (default) | median pHash dist |")
    print("|---|---:|---:|---:|---:|")
    for r in table:
        name = r["variant"] + (" *(not supported)*" if r["hard"] else "")
        print(f"| {name} | {r['pHash']:.0%} | {r['dHash']:.0%} | **{r['both']:.0%}** | {r['median_dp']:.0f} |")
    print(f"\nFalse-match rate on {len(neg['pHash'])} distinct pairs: pHash alone {fmr_p:.3%}, "
          f"pHash + dHash {fmr_both:.3%}")
    print(f"Library simulation: {len(fps)} photos, {len(groups)} groups, {pure} pure; "
          f"pairwise precision {precision:.1%}, recall {recall:.1%}")

    out = {
        "n_bases": len(names), "table": table, "sweep": sweep, "fmr_phash": fmr_p, "fmr_both": fmr_both,
        "library": {"photos": len(fps), "groups": len(groups), "pure_groups": pure,
                    "precision": precision, "recall": recall},
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "accuracy.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    plot(table, pos, neg, tp)


def plot(table, pos, neg, tp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1d1d1f", "#6b6b70", "#e4e2dc"
    colors = {"aHash": "#9aa0a6", "dHash": "#e8590c", "pHash": "#3b5bdb", "both": "#1f8a4c"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": grid,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=150)

    # Left: distance distributions for pHash
    allpos = np.concatenate([pos["pHash"][v] for v in POSITIVE])
    bins = np.arange(0, 50, 2) - 1  # pHash distances are always even (32 bits set)
    ax1.hist(neg["pHash"], bins=bins, density=True, color="#c9ccd1", label="different photos")
    ax1.hist(allpos, bins=bins, density=True, color=colors["pHash"], alpha=.85, label="edited copies (13 edits)")
    ax1.axvline(tp + 1, color=ink, lw=1.2, ls="--")
    ax1.text(tp + 2, ax1.get_ylim()[1] * 0.9, f"default\nthreshold = {tp}", color=ink, fontsize=9.5, va="top")
    ax1.set_xlabel("pHash Hamming distance (bits out of 64)")
    ax1.set_ylabel("fraction of pairs")
    ax1.set_title("Duplicates cluster near 0, different photos near 32", loc="left", color=ink, fontsize=11.5)
    ax1.legend(frameon=False)
    ax1.grid(axis="y", color=grid, lw=.8)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # Right: per-variant recall at the default thresholds
    variants_ = [r["variant"] for r in table][::-1]
    vals = [r["both"] for r in table][::-1]
    hard = [r["hard"] for r in table][::-1]
    y = np.arange(len(variants_))
    ax2.barh(y, vals, color=[("#c9ccd1" if h else colors["both"]) for h in hard], height=.68)
    for yi, v in zip(y, vals):
        ax2.text(min(v, 1) + 0.015, yi, f"{v:.0%}", va="center", fontsize=9, color=ink)
    ax2.set_yticks(y, [v + (" (unsupported)" if h else "") for v, h in zip(variants_, hard)])
    ax2.set_xlim(0, 1.12)
    ax2.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax2.set_xlabel("found as duplicate of the original (default thresholds)")
    ax2.set_title(f"Recall per edit, {len(pos['pHash']['resize 50%'])} photos each", loc="left", color=ink, fontsize=11.5)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color=grid, lw=.8)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out = os.path.join(HERE, "..", "assets", "accuracy.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor="white")
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
