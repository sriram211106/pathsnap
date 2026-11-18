"""Step 1d - Which trips have GPS points far from their reference route, and WHERE in the trip?

Run:  python scripts/04_eda_far_points.py --eda outputs/eda --thr-m 100
Writes outputs/eda/far_points_by_trip.csv and prints the worst trips.

Columns:
  share_far        fraction of the trip's points farther than thr from any route arc
  longest_far_run  longest consecutive run of far points (one block vs scattered noise)
  far_in_first10 / far_in_last10   share of far points in the first / last 10% of the trip
                   (high values -> route probably covers only part of the track)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--thr-m", type=float, default=100.0)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    eda = Path(args.eda)
    pts = pd.read_parquet(eda / "points.parquet")
    inv = pd.read_csv(eda / "trip_inventory.csv", dtype={"trip_id": str})

    rows = []
    for (part, tid), g in pts.groupby(["partition", "trip_id"]):
        g = g.sort_values("point_idx")
        d = g["dist_to_route_m"].to_numpy()
        far = d > args.thr_m
        n = len(d)
        k = max(1, n // 10)
        n_far = int(far.sum())
        rows.append({
            "partition": part, "trip_id": tid, "n_points": n, "n_far": n_far,
            "share_far": n_far / n,
            "longest_far_run": longest_run(far),
            "first_far_idx": int(np.argmax(far)) if n_far else -1,
            "last_far_idx": int(n - 1 - np.argmax(far[::-1])) if n_far else -1,
            "far_in_first10": float(far[:k].sum() / n_far) if n_far else 0.0,
            "far_in_last10": float(far[-k:].sum() / n_far) if n_far else 0.0,
            "dist_p50_m": float(np.nanmedian(d)),
        })
    res = pd.DataFrame(rows).merge(
        inv[["partition", "trip_id", "duration_s", "dt_max", "n_gaps_gt_10s", "speed_max_ms",
             "route_len_arcs", "route_n_continuity_breaks"]],
        on=["partition", "trip_id"], how="left",
    ).sort_values("share_far", ascending=False)
    res.to_csv(eda / "far_points_by_trip.csv", index=False)

    total_far = res["n_far"].sum()
    print(f"Points > {args.thr_m:.0f} m from route: {total_far} of {res['n_points'].sum()} "
          f"({total_far / res['n_points'].sum():.1%})")
    for part, g in res.groupby("partition"):
        print(f"  {part}: trips with any far point = {(g['n_far'] > 0).sum()} / {len(g)}")
    top = res[res["n_far"] > 0].head(args.top)
    top = top.assign(cum_share_of_all_far=top["n_far"].cumsum() / max(total_far, 1))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(top.round(3).to_string(index=False))
    print("saved", eda / "far_points_by_trip.csv")


if __name__ == "__main__":
    main()
