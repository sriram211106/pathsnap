"""Step 6 (first, Laptop 1 only) - How many labelled points would the training jitter filter remove?

Run:  python scripts/12_check_jitter_filter.py

Jitter = progress ratio below P (going nowhere) AND apparent speed above S (yet 'moving').
Removed points are left out of TRAINING only; scoring always uses all labelled points.
A good setting removes the known jitter (trips 33, 54) and very little elsewhere.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "features" / "train.parquet",
                     columns=["trip_id", "pt", "y", "progress_ratio", "speed_smooth_ms"])
pts = df[df["y"].notna()].drop_duplicates(["trip_id", "pt"])
known = pts["trip_id"].isin(["00000033", "00000054"])
print(f"Labelled train points: {len(pts):,}  (of which in trips 33/54: {int(known.sum()):,})\n")

rows = []
for P in (0.10, 0.15, 0.20, 0.30):
    for S in (2.0, 3.0, 5.0):
        rm = (pts["progress_ratio"] < P) & (pts["speed_smooth_ms"] > S)
        rows.append({"progress_below": P, "speed_above": S, "removed": int(rm.sum()),
                     "removed_share": rm.mean(),
                     "from_33_54": (rm & known).sum() / max(rm.sum(), 1),
                     "trips_losing_over_5pct": int((rm.groupby(pts["trip_id"]).mean() > 0.05).sum())})
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

P, S = 0.20, 3.0
rm = (pts["progress_ratio"] < P) & (pts["speed_smooth_ms"] > S)
print(f"\nDefault (progress < {P}, speed > {S}) - trips losing the most labelled points:")
by_trip = pd.DataFrame({"removed": rm.groupby(pts["trip_id"]).sum(), "share": rm.groupby(pts["trip_id"]).mean()})
print(by_trip.sort_values("removed", ascending=False).head(10).to_string(float_format=lambda x: f"{x:.3f}"))
