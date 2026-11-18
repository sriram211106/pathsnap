"""Step 2 check - Why is (almost) nothing flagged stationary?

Run:  python scripts/06_check_stationary.py

Prints:
  A. bug check: how many smoothed speeds are NaN (should be ~1 per segment)
  B. raw speed distribution over ALL train points (do near-zero speeds exist at all?)
  C. for the far-from-route points of trips 33, 54, 2: speed percentiles, how big an area
     they cover, and how far the vehicle actually moves per step.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "train"
EDA = ROOT / "outputs" / "eda" / "points.parquet"
PCTS = [1, 5, 10, 25, 50, 75, 90]

frames = []
for tdir in sorted(PROC.iterdir()):
    p = pd.read_parquet(tdir / "points.parquet")
    p["trip_id"] = tdir.name
    frames.append(p)
pts = pd.concat(frames, ignore_index=True)

print("A. Bug check")
n_seg = pts.groupby("trip_id")["segment_id"].nunique().sum()
print(f"   speed_smooth_ms NaN: {pts['speed_smooth_ms'].isna().sum()}  (segments in total: {n_seg})")
print(f"   speed_ms        NaN: {pts['speed_ms'].isna().sum()}")

print("\nB. Raw speed over ALL train points (m/s)")
s = pts["speed_ms"].dropna()
print("   percentiles " + "  ".join(f"p{q}={np.percentile(s, q):.2f}" for q in PCTS))
for thr in (0.5, 1.0, 2.0, 3.0):
    print(f"   share with speed < {thr} m/s: {(s < thr).mean():.2%}")
print(f"   share with step < 0.5 m (barely moved): {(pts['step_m'].dropna() < 0.5).mean():.2%}")

if not EDA.exists():
    sys.exit("\n(outputs/eda/points.parquet missing - run scripts/01_eda_inventory.py first)")
e = pd.read_parquet(EDA, columns=["partition", "trip_id", "point_idx", "dist_to_route_m"])
e = e[e["partition"] == "train"].drop(columns="partition")
m = pts.merge(e, on=["trip_id", "point_idx"], how="inner")

print("\nC. Far-from-route points (> 100 m) in the three trips")
for tid in ("00000033", "00000054", "00000002"):
    f = m[(m["trip_id"] == tid) & (m["dist_to_route_m"] > 100)]
    if f.empty:
        continue
    sp = f["speed_ms"].dropna()
    w = f["x"].max() - f["x"].min()
    h = f["y"].max() - f["y"].min()
    print(f"\n   trip {tid}: {len(f)} far points")
    print("     speed " + "  ".join(f"p{q}={np.percentile(sp, q):.2f}" for q in PCTS))
    print(f"     area covered by these points: {w:.0f} m x {h:.0f} m")
    print(f"     median step between points: {f['step_m'].median():.2f} m;  "
          f"total distance travelled: {f['step_m'].sum() / 1000:.2f} km")
    print(f"     time gap between points: share dt=1 s = {np.isclose(f['dt'], 1).mean():.1%}")
