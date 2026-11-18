"""Step 4b - Build the final candidate set for every train + student_test trip.

Run with the setting chosen from 09_candidate_sweep.py, e.g.:
    python scripts/10_build_candidates.py --radius 30 --k 10

Writes:
    data/processed/<partition>/<trip_id>/candidates.parquet
        point_idx, pt (row position), arc_id, dist_m, rank, is_fallback
        (no labels inside - labels are joined only when the training table is built)
    outputs/candidates/candidate_summary.csv
    outputs/candidates/candidate_config.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.candidates import CandidateConfig, build_arc_index, generate_candidates  # noqa: E402
from pathsnap.io import PARTITIONS, list_trips  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="outputs/candidates")
    args = ap.parse_args()

    cfg = CandidateConfig(radius_m=args.radius, max_k=args.k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    rows = []
    for part in PARTITIONS:
        for tid in tqdm(list_trips(Path(args.proc) / part), desc=part):
            tdir = Path(args.proc) / part / tid
            pts = pd.read_parquet(tdir / "points.parquet", columns=["point_idx", "x", "y"])
            net = pd.read_parquet(tdir / "arcs.parquet")
            cand = generate_candidates(pts["x"].to_numpy(), pts["y"].to_numpy(), net, cfg, build_arc_index(net))
            cand.insert(0, "point_idx", pts["point_idx"].to_numpy()[cand["pt"].to_numpy()])
            cand.to_parquet(tdir / "candidates.parquet", index=False)

            # miss-rate check uses labels (evaluation only; labels are NOT written into candidates)
            lab = pd.read_parquet(tdir / "labels.parquet", columns=["label_arc_id"])["label_arc_id"].to_numpy()
            has_true = np.zeros(len(pts), dtype=bool)
            m = cand["arc_id"].to_numpy() == lab[cand["pt"].to_numpy()]
            has_true[cand["pt"].to_numpy()[m]] = True
            labelled = lab >= 0
            per_pt = cand.groupby("pt").size()
            rows.append({
                "partition": part, "trip_id": tid, "n_points": len(pts), "n_labelled": int(labelled.sum()),
                "n_candidates": len(cand),
                "avg_candidates": float(per_pt.mean()),
                "max_candidates": int(per_pt.max()),
                "n_fallback_points": int(cand.loc[cand["is_fallback"], "pt"].nunique()),
                "n_missed": int((labelled & ~has_true).sum()),
            })

    s = pd.DataFrame(rows)
    s.to_csv(out / "candidate_summary.csv", index=False)
    for part, g in s.groupby("partition"):
        print(f"\n===== {part} (radius {cfg.radius_m:.0f} m, max K {cfg.max_k}) =====")
        print(f"  candidate miss rate      : {g['n_missed'].sum() / g['n_labelled'].sum():.2%}"
              f"  ({g['n_missed'].sum()} of {g['n_labelled'].sum()} labelled points)")
        print(f"  avg candidates per point : {g['n_candidates'].sum() / g['n_points'].sum():.1f}"
              f"   (max in any point: {g['max_candidates'].max()})")
        print(f"  points using fallback    : {g['n_fallback_points'].sum()}")
        print(f"  total candidate rows     : {g['n_candidates'].sum():,}")
    print("\nTrips with the most missed points:")
    print(s.assign(miss_rate=s["n_missed"] / s["n_labelled"].clip(lower=1))
            .sort_values("n_missed", ascending=False).head(6)
            [["partition", "trip_id", "n_labelled", "n_missed", "miss_rate"]].round(4).to_string(index=False))
    print(f"\nSaved candidates.parquet per trip and {out/'candidate_summary.csv'}")


if __name__ == "__main__":
    main()
