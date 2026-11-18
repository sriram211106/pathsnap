"""Step 3 - Build one reference label per GPS point for train + student_test.

Run from the repo root (after scripts/05_preprocess.py):
    python scripts/07_build_labels.py

Writes:
    data/processed/<partition>/<trip_id>/labels.parquet   point_idx + label (kept SEPARATE from features)
    outputs/labels/label_summary.csv                     per-trip label statistics
    outputs/labels/label_config.json

Prints a quality check comparing our order-preserving labels with naive
"nearest route arc" labels:
    wrong-direction share = labelled moving points whose arc points >90 deg away
    from the vehicle's direction of travel (lower is better).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.io import PARTITIONS, list_trips, load_route  # noqa: E402
from pathsnap.labels import LabelConfig, align_points_to_route, angle_diff_deg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="outputs/labels")
    args = ap.parse_args()

    cfg = LabelConfig()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    rows = []
    for part in PARTITIONS:
        for tid in tqdm(list_trips(Path(args.proc) / part), desc=part):
            tdir = Path(args.proc) / part / tid
            pts = pd.read_parquet(tdir / "points.parquet")
            net = pd.read_parquet(tdir / "arcs.parquet")
            route = load_route(Path(args.data_root) / part / tid / f"{tid}.route")

            t0 = time.perf_counter()
            lab = align_points_to_route(pts["x"].to_numpy(), pts["y"].to_numpy(), net, route, cfg)
            secs = time.perf_counter() - t0
            lab.insert(0, "point_idx", pts["point_idx"].to_numpy())
            lab.to_parquet(tdir / "labels.parquet", index=False)

            # quality check vs direction of travel (only points that are moving and have a heading)
            bearing = net["bearing_deg"].to_numpy()
            ok = (lab["label_arc_id"] >= 0).to_numpy() & pts["heading_deg"].notna().to_numpy()
            ok &= (pts["speed_smooth_ms"] > 2.0).to_numpy()
            head = pts["heading_deg"].to_numpy()[ok]
            dp_wrong = angle_diff_deg(head, bearing[lab["label_arc_id"].to_numpy()[ok]]) > 90
            nv_wrong = angle_diff_deg(head, bearing[lab["naive_arc_id"].to_numpy()[ok]]) > 90
            labelled = lab["label_arc_id"] >= 0
            r = np.asarray(route) if route is not None else np.array([])
            rows.append({
                "partition": part, "trip_id": tid, "n_points": len(lab),
                "share_labelled": float(labelled.mean()),
                "n_unlabelled": int((~labelled).sum()),
                "label_dist_p50_m": float(lab.loc[labelled, "label_dist_m"].median()) if labelled.any() else np.nan,
                "label_dist_p95_m": float(lab.loc[labelled, "label_dist_m"].quantile(0.95)) if labelled.any() else np.nan,
                "route_len_arcs": int(len(r)),
                "route_arcs_with_a_point": int(lab.loc[labelled, "route_pos"].nunique()),
                "share_differs_from_naive": float((lab.loc[labelled, "label_arc_id"]
                                                   != lab.loc[labelled, "naive_arc_id"]).mean()) if labelled.any() else np.nan,
                "n_direction_checked": int(ok.sum()),
                "wrong_dir_ours": float(dp_wrong.mean()) if ok.any() else np.nan,
                "wrong_dir_naive": float(nv_wrong.mean()) if ok.any() else np.nan,
                "seconds": round(secs, 2),
            })

    s = pd.DataFrame(rows)
    s.to_csv(out / "label_summary.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        for part, g in s.groupby("partition"):
            w = g["n_direction_checked"]
            print(f"\n===== {part}: {len(g)} trips =====")
            print(f"  points labelled          : {(g['share_labelled'] * g['n_points']).sum() / g['n_points'].sum():.2%}"
                  f"  (unlabelled: {g['n_unlabelled'].sum()})")
            print(f"  label distance (median of trip medians): {g['label_dist_p50_m'].median():.1f} m;"
                  f"  worst trip p95: {g['label_dist_p95_m'].max():.1f} m")
            print(f"  route arcs that received >=1 point: {g['route_arcs_with_a_point'].sum() / g['route_len_arcs'].sum():.1%}")
            print(f"  labels that differ from naive nearest-arc: "
                  f"{(g['share_differs_from_naive'] * g['n_points']).sum() / g['n_points'].sum():.2%}")
            print(f"  WRONG-DIRECTION share  ours: {(g['wrong_dir_ours'] * w).sum() / w.sum():.3%}"
                  f"   naive: {(g['wrong_dir_naive'] * w).sum() / w.sum():.3%}   (n={int(w.sum())})")
            print(f"  slowest trip: {g['seconds'].max():.1f} s")
        print("\nMost unlabelled trips:")
        print(s.sort_values("n_unlabelled", ascending=False).head(6)[
            ["partition", "trip_id", "n_points", "n_unlabelled", "share_labelled"]].to_string(index=False))
        print("\nTrips where naive labels go most often in the wrong direction:")
        print(s.sort_values("wrong_dir_naive", ascending=False).head(6)[
            ["partition", "trip_id", "wrong_dir_ours", "wrong_dir_naive", "n_direction_checked"]].round(4).to_string(index=False))
    print(f"\nSaved labels.parquet per trip and {out/'label_summary.csv'}")


if __name__ == "__main__":
    main()
