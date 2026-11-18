"""Step 6 - Fair rule-based baselines (exactly ONE arc per point, same metric as the models).

Run:  python scripts/13b_baselines.py

  nearest_arbitrary : smallest distance; ties (two directions of a street) broken by arc_id
  nearest_heading   : smallest distance; ties broken by smallest heading difference
Top-1 is measured on labelled points, exactly like scripts/13_train_model.py.
Writes outputs/models/baselines.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.training import make_folds  # noqa: E402

cols = ["trip_id", "pt", "arc_id", "y", "dist_m", "heading_diff"]
res = {}
for part in ("train", "student_test"):
    df = pd.read_parquet(ROOT / "data" / "features" / f"{part}.parquet", columns=cols)
    df = df[df["y"].notna()]
    out = {}
    for name, keys in (("nearest_arbitrary", ["dist_m", "arc_id"]),
                       ("nearest_heading", ["dist_m", "heading_diff", "arc_id"])):
        # round distance to 1 mm so the two directions of a street tie exactly
        d = df.assign(dist_m=df["dist_m"].round(3)).sort_values(["trip_id", "pt"] + keys, kind="stable")
        top = d.drop_duplicates(["trip_id", "pt"])
        out[name] = {"top1_accuracy": float(top["y"].mean())}
        if part == "train":   # per CV fold, for a like-for-like comparison with CV results
            fold_of = make_folds(df)
            per = top.groupby(top["trip_id"].map(fold_of))["y"].mean()
            out[name]["cv_fold_top1"] = [round(float(v), 4) for v in per.to_numpy()]
            out[name]["cv_mean"] = float(per.mean())
    res[part] = out
    print(f"\n{part}:")
    for k, v in out.items():
        extra = f"   per fold {v['cv_fold_top1']}  mean {v['cv_mean']:.4f}" if "cv_fold_top1" in v else ""
        print(f"  {k:<18} top1 {v['top1_accuracy']:.4f}{extra}")

(ROOT / "outputs" / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "models" / "baselines.json").write_text(json.dumps(res, indent=2))
print("\nSaved outputs/models/baselines.json")
