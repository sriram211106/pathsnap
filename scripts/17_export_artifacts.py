"""Step 10 - Export everything the API needs into artifacts/ (commit this folder to git).

Run on Laptop 1:
    python scripts/17_export_artifacts.py --model lgbm_tuned

Copies the trained model and collects every configuration used in training, so the API
runs EXACTLY the pipeline that was evaluated:
    artifacts/model.joblib
    artifacts/manifest.json   model name, metrics summary, and all configs:
                              preprocess, candidates, features (+ order), HMM, library versions
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm_tuned")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    mdir = ROOT / "outputs" / "models" / args.model
    hdir = ROOT / "outputs" / "hmm" / args.model
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((mdir / "metrics.json").read_text())
    feat_cfg = json.loads((ROOT / "outputs" / "features" / "feature_config.json").read_text())
    if feat_cfg.get("features") != FEATURES or metrics["features"] != FEATURES:
        sys.exit("Feature list mismatch between code, feature build and model - rebuild before exporting.")
    if not (hdir / "hmm_config.json").exists():
        sys.exit(f"Missing {hdir/'hmm_config.json'} - run: python scripts/15_hmm_tune_evaluate.py --model {args.model}")

    shutil.copy2(mdir / "model.joblib", out / "model.joblib")
    import lightgbm, numpy, pandas, sklearn, shapely, pyproj  # noqa: E401
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "model_type": metrics.get("model_type", args.model),
        "params": metrics["params"],
        "features": FEATURES,
        "preprocess_config": json.loads((ROOT / "outputs" / "preprocess" / "preprocess_config.json").read_text()),
        "candidate_config": json.loads((ROOT / "outputs" / "candidates" / "candidate_config.json").read_text()),
        "feature_config": {k: v for k, v in feat_cfg.items() if k != "features"},
        "hmm_config": json.loads((hdir / "hmm_config.json").read_text()),
        "metrics_summary": {
            "cv_top1": metrics["cv"]["top1_accuracy"], "cv_macro_f1": metrics["cv"]["macro_f1"],
            "student_test_top1_classifier": metrics["student_test"]["top1_accuracy"],
            "hmm_summary": json.loads((hdir / "summary.json").read_text()) if (hdir / "summary.json").exists() else None,
        },
        "library_versions": {m.__name__: m.__version__ for m in (lightgbm, numpy, pandas, sklearn, shapely, pyproj)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {args.model} -> {out}/model.joblib + manifest.json")
    print(f"HMM: beta={manifest['hmm_config']['beta_m']} m, alpha={manifest['hmm_config']['emission_weight']}")
    print("Commit the artifacts/ folder so the API laptop runs exactly this.")


if __name__ == "__main__":
    main()
