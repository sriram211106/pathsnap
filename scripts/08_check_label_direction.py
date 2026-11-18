"""Step 3 check - Where do the ~10% wrong-direction labels come from?

Run:  python scripts/08_check_label_direction.py

A. wrong-direction share by speed (noise in heading should mostly hit slow points)
B. per-trip spread: is it a few bad trips or everywhere?
C. fast points only (>= 10 m/s, heading is reliable there): for wrong-direction labels,
   does the SAME street exist in the opposite direction, and would that reverse arc
   match the direction of travel? If yes -> the reference route itself uses the
   opposite direction of a two-way street (a data limitation, not our bug).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def angle_diff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


frames = []
for part in ("train", "student_test"):
    for tdir in sorted((PROC / part).iterdir()):
        p = pd.read_parquet(tdir / "points.parquet", columns=["point_idx", "heading_deg", "speed_smooth_ms"])
        lab = pd.read_parquet(tdir / "labels.parquet", columns=["point_idx", "label_arc_id"])
        net = pd.read_parquet(tdir / "arcs.parquet", columns=["u", "v", "bearing_deg"])
        d = p.merge(lab, on="point_idx")
        d = d[(d["label_arc_id"] >= 0) & d["heading_deg"].notna() & (d["speed_smooth_ms"] > 2.0)].copy()
        a = d["label_arc_id"].to_numpy()
        d["diff"] = angle_diff(d["heading_deg"], net["bearing_deg"].to_numpy()[a])
        # reverse arc of the same street (v -> u), if it exists
        n_nodes = int(max(net["u"].max(), net["v"].max())) + 1
        key = pd.Series(np.arange(len(net)), index=net["u"].to_numpy() * n_nodes + net["v"].to_numpy())
        key = key[~key.index.duplicated()]
        rev_key = net["v"].to_numpy()[a] * n_nodes + net["u"].to_numpy()[a]
        rev = key.reindex(rev_key).to_numpy()
        d["has_reverse"] = ~np.isnan(rev)
        rev_bearing = np.where(d["has_reverse"], net["bearing_deg"].to_numpy()[np.nan_to_num(rev).astype(int)], np.nan)
        d["reverse_matches"] = angle_diff(d["heading_deg"], rev_bearing) < 45
        d["partition"], d["trip_id"] = part, tdir.name
        frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["wrong"] = df["diff"] > 90

print("A. Wrong-direction share by speed (all labelled moving points)")
bins = [2, 5, 10, 20, np.inf]
df["speed_bin"] = pd.cut(df["speed_smooth_ms"], bins=bins, labels=["2-5", "5-10", "10-20", ">20"])
print(df.groupby("speed_bin", observed=True)["wrong"].agg(["mean", "size"])
        .rename(columns={"mean": "wrong_share", "size": "n"}).to_string(float_format=lambda x: f"{x:.3f}"))

print("\nB. Per-trip wrong-direction share (all speeds)")
pt = df.groupby(["partition", "trip_id"])["wrong"].mean()
print(f"   median trip: {pt.median():.3f}   trips > 5%: {(pt > 0.05).sum()} / {len(pt)}   trips > 20%: {(pt > 0.2).sum()}")
print("   worst 8:", ", ".join(f"{t}={v:.2f}" for (_, t), v in pt.sort_values(ascending=False).head(8).items()))

print("\nC. Fast points only (>= 10 m/s)")
f = df[df["speed_smooth_ms"] >= 10]
w = f[f["wrong"]]
print(f"   wrong-direction share: {f['wrong'].mean():.3f}  ({len(w)} of {len(f)})")
if len(w):
    print(f"   of those: street exists in opposite direction: {w['has_reverse'].mean():.1%};"
          f"  opposite arc matches travel direction: {w['reverse_matches'].mean():.1%}")
    ft = f.groupby(["partition", "trip_id"])["wrong"].mean().sort_values(ascending=False)
    print("   worst trips at speed:", ", ".join(f"{t}={v:.2f}" for (_, t), v in ft.head(8).items()))
