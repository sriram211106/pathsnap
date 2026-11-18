"""Step 1b - EDA figures for the report (reads outputs of 01_eda_inventory.py).

Run:  python scripts/02_eda_plots.py --eda outputs/eda
Saves PNGs to outputs/eda/figures/. Every title states the partition used (report requirement).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"train": "#2a6fdb", "student_test": "#e07b39"}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", default="outputs/eda")
    args = ap.parse_args()
    eda = Path(args.eda)
    figs = eda / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(eda / "trip_inventory.csv", dtype={"trip_id": str})
    pts = pd.read_parquet(eda / "points.parquet")

    # 1. GPS points per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for part, g in inv.groupby("partition"):
        ax.hist(g["n_points"], bins=20, alpha=0.6, label=f"{part} (n={len(g)})", color=COLORS.get(part))
    ax.set(xlabel="GPS observations per trip", ylabel="Number of trips", title="GPS observations per trip (train + student_test)")
    ax.legend()
    save(fig, figs / "01_points_per_trip.png")

    # 2. Sampling interval distribution
    bins = [0, 1.5, 2.5, 5.5, 10.5, 30.5, np.inf]
    labels = ["1 s", "2 s", "3-5 s", "6-10 s", "11-30 s", ">30 s"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    for i, (part, g) in enumerate(pts.dropna(subset=["dt"]).groupby("partition")):
        counts = pd.cut(g["dt"], bins=bins, labels=labels).value_counts().reindex(labels)
        ax.bar(np.arange(len(labels)) + i * width, counts.values, width, label=part, color=COLORS.get(part))
    ax.set_xticks(np.arange(len(labels)) + width / 2, labels)
    ax.set_yscale("log")
    ax.set(xlabel="Time between consecutive GPS points", ylabel="Count (log scale)", title="Sampling intervals (train + student_test)")
    ax.legend()
    save(fig, figs / "02_sampling_intervals.png")

    # 3. GPS-to-reference-route distance (guides candidate radius)
    d = pts.loc[pts["partition"] == "train", "dist_to_route_m"].dropna()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(d.clip(upper=100), bins=100, color=COLORS["train"])
    a1.set(xlabel="Distance to nearest reference-route arc (m, clipped at 100)", ylabel="GPS points", title="GPS noise vs reference route (train)")
    xs = np.sort(d.to_numpy())
    a2.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=COLORS["train"])
    for r in (10, 20, 30, 50):
        share = (d <= r).mean()
        a2.axvline(r, ls="--", color="grey", lw=0.8)
        a2.text(r, 0.05, f"{r} m: {share:.1%}", rotation=90, va="bottom", fontsize=8)
    a2.set(xlim=(0, 100), xlabel="Distance (m)", ylabel="Cumulative share of points", title="CDF of GPS-to-route distance (train)")
    save(fig, figs / "03_gps_to_route_distance.png")

    # 4. Road network size
    fig, ax = plt.subplots(figsize=(7, 4))
    for part, g in inv.groupby("partition"):
        ax.scatter(g["n_nodes"], g["n_arcs"], label=part, alpha=0.7, color=COLORS.get(part))
    ax.set(xscale="log", yscale="log", xlabel="Nodes", ylabel="Arcs (directed)", title="Road-network size per trip (train + student_test)")
    ax.legend()
    save(fig, figs / "04_network_size.png")

    # 5. Network connectivity
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for part, g in inv.groupby("partition"):
        a1.hist(g["n_weak_components"], bins=20, alpha=0.6, label=part, color=COLORS.get(part))
        a2.hist(g["largest_scc_share_of_used_nodes"], bins=20, alpha=0.6, label=part, color=COLORS.get(part))
    a1.set(xlabel="Weakly connected components", ylabel="Trips", title="Network fragmentation (train + student_test)")
    a2.set(xlabel="Share of nodes in largest strongly connected component", ylabel="Trips", title="Directed reachability (train + student_test)")
    a1.legend()
    save(fig, figs / "05_network_connectivity.png")

    # 6. Where trips are (geographic spread)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for part, g in inv.groupby("partition"):
        ax.scatter(g["lon_center"], g["lat_center"], label=part, color=COLORS.get(part), s=25)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Trip locations (centroid of GPS points; train + student_test)")
    ax.legend()
    save(fig, figs / "06_trip_locations.png")

    # 7. Route quality
    if "route_n_continuity_breaks" in inv:
        fig, ax = plt.subplots(figsize=(9, 4))
        g = inv.sort_values(["partition", "trip_id"])
        ax.bar(range(len(g)), g["route_n_continuity_breaks"], color=[COLORS.get(p) for p in g["partition"]])
        ax.set(xlabel="Trip (sorted by partition, id)", ylabel="Breaks", title="Reference-route continuity breaks per trip (train + student_test)")
        save(fig, figs / "07_route_continuity_breaks.png")

    # 8. Implied speed (spots GPS jumps)
    sp = pts.loc[pts["partition"] == "train", "speed_ms"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sp.clip(upper=80), bins=80, color=COLORS["train"])
    ax.set(yscale="log", xlabel="Implied speed between consecutive points (m/s, clipped at 80)", ylabel="Steps (log)", title="Implied speed distribution (train)")
    save(fig, figs / "08_implied_speed.png")

    # missing-value / anomaly table for the report
    cols = ["partition", "trip_id", "n_points", "n_nan_rows", "n_invalid_coords", "n_duplicate_rows",
            "n_non_increasing_t", "n_gaps_gt_10s", "dt_max", "speed_max_ms", "route_n_continuity_breaks",
            "gps_route_dist_p95_m"]
    cols = [c for c in cols if c in inv]
    inv[cols].to_csv(eda / "data_quality_table.csv", index=False)
    print("saved", eda / "data_quality_table.csv")


if __name__ == "__main__":
    main()
