"""Step 5 - Build the feature tables for train and student_test.

Run (after 10_build_candidates.py):
    python scripts/11_build_features.py

Writes:
    data/features/train.parquet, data/features/student_test.parquet
        one row per (point, candidate): ids, FEATURES, and the target
        y = 1 if candidate == true arc, 0 if not, NaN if the point is unlabelled
    outputs/features/feature_config.json, outputs/features/feature_sanity.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.features import FEATURES, FeatureConfig, build_features  # noqa: E402
from pathsnap.io import PARTITIONS, list_trips  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out-data", default="data/features")
    ap.add_argument("--out", default="outputs/features")
    args = ap.parse_args()
    cfg = FeatureConfig()
    Path(args.out_data).mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_config.json").write_text(json.dumps({**cfg.to_dict(), "features": FEATURES}, indent=2))

    for part in PARTITIONS:
        frames = []
        for tid in tqdm(list_trips(Path(args.proc) / part), desc=part):
            tdir = Path(args.proc) / part / tid
            pts = pd.read_parquet(tdir / "points.parquet")
            net = pd.read_parquet(tdir / "arcs.parquet")
            cand = pd.read_parquet(tdir / "candidates.parquet")
            f = build_features(pts, net, cand, cfg)

            # target: attached AFTER features are built, from the separate labels file
            lab = pd.read_parquet(tdir / "labels.parquet", columns=["label_arc_id"])["label_arc_id"].to_numpy()
            true = lab[f["pt"].to_numpy()]
            f["y"] = np.where(true >= 0, (f["arc_id"].to_numpy() == true).astype(float), np.nan)
            f.insert(0, "segment_id", pts["segment_id"].to_numpy()[f["pt"].to_numpy()])
            f.insert(0, "point_idx", pts["point_idx"].to_numpy()[f["pt"].to_numpy()])
            f.insert(0, "trip_id", tid)
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        float_cols = df.select_dtypes("float64").columns.drop("y")
        df[float_cols] = df[float_cols].astype("float32")
        df.to_parquet(Path(args.out_data) / f"{part}.parquet", index=False)

        lab_rows = df[df["y"].notna()]
        n_pts = df.groupby(["trip_id", "pt"]).ngroups
        n_lab_pts = lab_rows.groupby(["trip_id", "pt"]).ngroups
        print(f"\n===== {part} =====")
        print(f"  rows: {len(df):,}   points: {n_pts:,}   labelled points: {n_lab_pts:,}")
        print(f"  share of labelled rows that are the true arc (y=1): {lab_rows['y'].mean():.2%}")
        nan_cols = {c: int(df[c].isna().sum()) for c in FEATURES if df[c].isna().any()}
        print(f"  features with NaN: {nan_cols if nan_cols else 'none'}")
        top1 = lab_rows[lab_rows["rank"] == 0].groupby(["trip_id", "pt"])["y"].max()
        print(f"  baseline - true arc is among the nearest (rank 0) candidates: {top1.mean():.2%}")

        if part == "train":
            sanity = lab_rows.groupby("y")[FEATURES].mean().T
            sanity.columns = ["mean_when_wrong (y=0)", "mean_when_true (y=1)"]
            sanity.to_csv(out / "feature_sanity.csv")
            with pd.option_context("display.width", 160):
                print("\n  Feature means, wrong vs true candidates (train, labelled points):")
                print(sanity.round(3).to_string())
            pr = df.drop_duplicates(["trip_id", "pt"])
            jit = pr[pr["trip_id"].isin(["00000033", "00000054"]) & pr["y"].isna()]["progress_ratio"]
            lab = pr[pr["y"].notna()]["progress_ratio"]
            q = [5, 25, 50, 75]
            print("\n  Progress ratio (1 = moving straight, ~0 = going nowhere):")
            print("    labelled points       : " + "  ".join(f"p{k}={np.percentile(lab, k):.2f}" for k in q))
            if len(jit):
                print("    unlabelled pts 33/54  : " + "  ".join(f"p{k}={np.percentile(jit, k):.2f}" for k in q))

    print(f"\nSaved feature tables under {args.out_data}/")


if __name__ == "__main__":
    main()
