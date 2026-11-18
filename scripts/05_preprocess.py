"""Step 2 - Run preprocessing for every train + student_test trip and save the results.

Run from the repo root:
    python scripts/05_preprocess.py

Writes, per trip:
    data/processed/<partition>/<trip_id>/points.parquet   cleaned GPS points
    data/processed/<partition>/<trip_id>/arcs.parquet     projected road network
    data/processed/<partition>/<trip_id>/meta.json        EPSG code + config used
And:
    outputs/preprocess/preprocess_summary.csv            per-trip counts (segments, jumps, stationary)
    outputs/preprocess/preprocess_config.json            thresholds used (for the report)

Optional sanity check (train only, needs outputs/eda/points.parquet from Step 1):
    are the far-from-route points we saw on the maps mostly flagged stationary?
    The route is used ONLY to print this check - it is never written into the processed data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.io import PARTITIONS, list_trips, load_trip  # noqa: E402
from pathsnap.preprocess import PreprocessConfig, preprocess_track, project_network, trip_projector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out-data", default="data/processed")
    ap.add_argument("--out", default="outputs/preprocess")
    ap.add_argument("--eda", default="outputs/eda")
    args = ap.parse_args()

    cfg = PreprocessConfig()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocess_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    rows = []
    for part in PARTITIONS:
        for tid in tqdm(list_trips(Path(args.data_root) / part), desc=part):
            trip = load_trip(Path(args.data_root) / part / tid)
            proj, epsg = trip_projector(trip["track"])
            pts = preprocess_track(trip["track"], proj, cfg)
            net = project_network(trip["nodes"], trip["arcs"], proj)

            tdir = Path(args.out_data) / part / tid
            tdir.mkdir(parents=True, exist_ok=True)
            pts.to_parquet(tdir / "points.parquet", index=False)
            net.to_parquet(tdir / "arcs.parquet", index=False)
            (tdir / "meta.json").write_text(json.dumps({"trip_id": tid, "partition": part, "epsg": epsg,
                                                        "config": cfg.to_dict()}, indent=2))

            seg_sizes = pts.groupby("segment_id").size()
            rows.append({
                "partition": part, "trip_id": tid, "epsg": epsg,
                "n_points": len(pts),
                "n_segments": int(pts["segment_id"].nunique()),
                "smallest_segment": int(seg_sizes.min()),
                "n_single_point_segments": int((seg_sizes == 1).sum()),
                "n_jumps": int(pts["is_jump"].sum()),
                "share_stationary": float(pts["is_stationary"].mean()),
                "share_heading_missing": float(pts["heading_deg"].isna().mean()),
                "n_arcs": len(net),
                "arc_len_median_m": float(net["length_m"].median()),
                "arc_len_max_m": float(net["length_m"].max()),
            })

    summ = pd.DataFrame(rows)
    summ.to_csv(out / "preprocess_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for part, g in summ.groupby("partition"):
            print(f"\n===== {part}: {len(g)} trips, {g['n_points'].sum()} points =====")
            cols = ["n_segments", "n_single_point_segments", "n_jumps", "share_stationary",
                    "share_heading_missing", "arc_len_median_m", "arc_len_max_m"]
            print(g[cols].describe().T[["min", "50%", "mean", "max"]].round(3).to_string())
        print("\nTrips with the most segments:")
        print(summ.sort_values("n_segments", ascending=False).head(8)[
            ["partition", "trip_id", "n_points", "n_segments", "n_jumps", "share_stationary"]].to_string(index=False))

    # sanity check of the stationary flag against Step-1 findings (train only, printed only)
    eda_pts = Path(args.eda) / "points.parquet"
    if eda_pts.exists():
        e = pd.read_parquet(eda_pts, columns=["partition", "trip_id", "point_idx", "dist_to_route_m"])
        e = e[e["partition"] == "train"]
        frames = []
        for tid in summ.loc[summ["partition"] == "train", "trip_id"]:
            p = pd.read_parquet(Path(args.out_data) / "train" / tid / "points.parquet",
                                columns=["point_idx", "is_stationary"])
            p["trip_id"] = tid
            frames.append(p)
        m = pd.concat(frames).merge(e, on=["trip_id", "point_idx"], how="inner")
        m["far"] = m["dist_to_route_m"] > 100
        print("\nSanity check (train): share of points flagged stationary")
        print(f"  points > 100 m from route : {m.loc[m['far'], 'is_stationary'].mean():.1%}  (n={int(m['far'].sum())})")
        print(f"  points <= 100 m from route: {m.loc[~m['far'], 'is_stationary'].mean():.1%}  (n={int((~m['far']).sum())})")
        for tid in ("00000033", "00000054", "00000002"):
            mt = m[(m["trip_id"] == tid) & m["far"]]
            if len(mt):
                print(f"  trip {tid}: far points flagged stationary = {mt['is_stationary'].mean():.1%} (n={len(mt)})")

    print(f"\nSaved processed trips under {args.out_data}/ and summary under {out}/")


if __name__ == "__main__":
    main()
