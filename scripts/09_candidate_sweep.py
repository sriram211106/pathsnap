"""Step 4a - Choose the candidate radius R and maximum K by measuring the candidate miss rate.

Run (after 05_preprocess.py and 07_build_labels.py):
    python scripts/09_candidate_sweep.py

Candidate miss = the point's true arc (its label) is NOT in its candidate list.
Measured on LABELLED points only. Settings are CHOSEN ON TRAIN ONLY; the
student_test miss rate for the chosen setting is printed as a check.

Writes outputs/candidates/sweep.csv and outputs/candidates/sweep_miss_rate.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.candidates import build_arc_index, nearby_arcs  # noqa: E402
from pathsnap.io import PARTITIONS, list_trips  # noqa: E402

RADII = [15, 20, 25, 30, 40, 50, 60]
KS = [4, 6, 8, 10, 15, 20, 30, 40]
R_MAX = 60.0  # > 50 m label limit, so every labelled point's true arc is within reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="outputs/candidates")
    ap.add_argument("--tolerance-pp", type=float, default=0.5,
                    help="accept settings whose train miss rate is within this many percentage points of the best")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # accumulators per partition: misses / labelled points, candidates / all points
    acc = {p: {"miss": np.zeros((len(RADII), len(KS))), "cands": np.zeros((len(RADII), len(KS))),
               "n_lab": 0, "n_pts": 0} for p in PARTITIONS}

    for part in PARTITIONS:
        for tid in tqdm(list_trips(Path(args.proc) / part), desc=part):
            tdir = Path(args.proc) / part / tid
            pts = pd.read_parquet(tdir / "points.parquet", columns=["x", "y"])
            net = pd.read_parquet(tdir / "arcs.parquet")
            lab = pd.read_parquet(tdir / "labels.parquet", columns=["label_arc_id"])
            df = nearby_arcs(pts["x"].to_numpy(), pts["y"].to_numpy(), net, R_MAX, build_arc_index(net))

            # where does the TRUE arc sit for each labelled point?
            true = lab["label_arc_id"].to_numpy()
            labelled = true >= 0
            df["true_arc"] = true[df["pt"].to_numpy()]
            hit = df[df["arc_id"] == df["true_arc"]].drop_duplicates("pt")
            t_dist = np.full(len(pts), np.inf)
            t_rank = np.full(len(pts), np.iinfo(np.int64).max)
            t_dist[hit["pt"].to_numpy()] = hit["dist_m"].to_numpy()
            t_rank[hit["pt"].to_numpy()] = hit["rank"].to_numpy()

            a = acc[part]
            a["n_lab"] += int(labelled.sum())
            a["n_pts"] += len(pts)
            d, rk = df["dist_m"].to_numpy(), df["rank"].to_numpy()
            for i, R in enumerate(RADII):
                for j, K in enumerate(KS):
                    found = (t_dist <= R) & (t_rank < K)
                    a["miss"][i, j] += int((labelled & ~found).sum())
                    a["cands"][i, j] += int(((d <= R) & (rk < K)).sum())

    rows = []
    for part, a in acc.items():
        for i, R in enumerate(RADII):
            for j, K in enumerate(KS):
                rows.append({"partition": part, "radius_m": R, "max_k": K,
                             "miss_rate": a["miss"][i, j] / max(a["n_lab"], 1),
                             "avg_candidates": a["cands"][i, j] / max(a["n_pts"], 1)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "sweep.csv", index=False)

    tr = sweep[sweep["partition"] == "train"]
    print("\nTRAIN candidate miss rate (%)  rows = radius m, columns = max K")
    print((tr.pivot(index="radius_m", columns="max_k", values="miss_rate") * 100).round(2).to_string())
    print("\nTRAIN average candidates per point")
    print(tr.pivot(index="radius_m", columns="max_k", values="avg_candidates").round(1).to_string())

    best = tr["miss_rate"].min()
    ok = tr[tr["miss_rate"] <= best + args.tolerance_pp / 100]
    pick = ok.sort_values(["avg_candidates", "miss_rate"]).iloc[0]
    st = sweep[(sweep["partition"] == "student_test") & (sweep["radius_m"] == pick["radius_m"])
               & (sweep["max_k"] == pick["max_k"])].iloc[0]
    print(f"\nBest achievable train miss rate: {best:.2%}")
    print(f"SUGGESTED (train only): radius = {pick['radius_m']:.0f} m, max K = {pick['max_k']:.0f}  ->"
          f" train miss {pick['miss_rate']:.2%}, avg {pick['avg_candidates']:.1f} candidates/point")
    print(f"Check on student_test with the same setting: miss {st['miss_rate']:.2%},"
          f" avg {st['avg_candidates']:.1f} candidates/point")

    fig, ax = plt.subplots(figsize=(8, 5))
    for R, g in tr.groupby("radius_m"):
        ax.plot(g["avg_candidates"], g["miss_rate"] * 100, marker="o", ms=3, label=f"R = {R} m")
    ax.scatter([pick["avg_candidates"]], [pick["miss_rate"] * 100], s=120, facecolors="none",
               edgecolors="black", zorder=5, label="chosen")
    ax.set(xlabel="Average candidates per point", ylabel="Candidate miss rate (%)",
           title="Candidate miss rate vs candidate-set size (train; each line = one radius, points = max K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sweep_miss_rate.png", dpi=150)
    print(f"\nSaved {out/'sweep.csv'} and {out/'sweep_miss_rate.png'}")


if __name__ == "__main__":
    main()
