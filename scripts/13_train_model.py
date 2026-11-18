"""Step 6 - Train ONE model: 5-fold grouped CV on train, refit on all train, evaluate on student_test.

Run (each laptop runs its own models):
    python scripts/13_train_model.py --model logreg
    models: logreg, gnb, knn, rf, lgbm, mlp, svm_rbf

Needs only data/features/train.parquet and data/features/student_test.parquet.
Writes outputs/models/<model>/:
    metrics.json          config, CV per fold + mean/std, student_test metrics, times
    model.joblib          final model (refit on all train)
    oof_train.parquet     out-of-fold scores for every train candidate (for comparison + HMM tuning)
    test_scores.parquet   scores for every student_test candidate
    history.png/.json     train/validation loss per epoch or iteration (mlp, lgbm only)
    confusion_matrix.png  student_test, candidate level
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.training import (MODEL_SPECS, N_FOLDS, SEED, TrainFilterConfig, evaluate, fit_model,  # noqa: E402
                               jitter_rows, load_features, make_folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="outputs/models")
    ap.add_argument("--params", default=None, help="JSON file with hyperparameters overriding the defaults")
    ap.add_argument("--name", default=None, help="output folder name, e.g. lgbm_tuned")
    args = ap.parse_args()
    name, spec = args.model, MODEL_SPECS[args.model]
    params = dict(spec["params"])
    if args.params:
        params.update(json.loads(Path(args.params).read_text()))
    run_name = args.name or name
    out = Path(args.out) / run_name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    fcfg = TrainFilterConfig()

    train = load_features(Path(args.features) / "train.parquet")
    test = load_features(Path(args.features) / "student_test.parquet")
    X = train[FEATURES].to_numpy(np.float32)
    y = train["y"].to_numpy()
    labelled = ~np.isnan(y)
    jitter = jitter_rows(train, fcfg)
    fold_of = make_folds(train[labelled])
    fold = train["trip_id"].map(fold_of).to_numpy()
    print(f"[{name}] train rows {len(train):,} | labelled {labelled.sum():,} | "
          f"left out of training as jitter {int((labelled & jitter).sum()):,}")

    def sample(idx):
        cap = spec["max_train_rows"]
        return np.sort(rng.choice(idx, size=cap, replace=False)) if cap and len(idx) > cap else idx

    oof = np.full(len(train), np.nan)
    folds, history, best_epochs = [], None, []
    for k in range(N_FOLDS):
        tr = sample(np.flatnonzero((fold != k) & labelled & ~jitter))
        va = np.flatnonzero(fold == k)
        va_lab = np.flatnonzero((fold == k) & labelled)
        want_val = name == "mlp" or (name == "lgbm" and k == 0)
        t0 = time.perf_counter()
        m, hist = fit_model(name, params, X[tr], y[tr].astype(int),
                            X[va_lab] if want_val else None, y[va_lab].astype(int) if want_val else None)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred_s = time.perf_counter() - t0
        if k == 0 and hist:
            history = hist
        if name == "mlp":
            best_epochs.append(m.best_epoch)
        r = evaluate(train.iloc[va], oof[va])
        r.update({"fold": k, "n_train_rows": int(len(tr)), "fit_seconds": round(fit_s, 1),
                  "predict_seconds": round(pred_s, 1),
                  "val_trips": sorted(t for t, f in fold_of.items() if f == k)})
        folds.append(r)
        print(f"  fold {k}: top1 {r['top1_accuracy']:.4f}  macroF1 {r['macro_f1']:.4f}  "
              f"logloss {r['log_loss']:.4f}  ({fit_s:.0f}s fit)")

    cv = {m_: {"mean": float(np.mean([f[m_] for f in folds])), "std": float(np.std([f[m_] for f in folds]))}
          for m_ in ("top1_accuracy", "macro_f1", "f1", "precision", "recall", "log_loss")}
    oof_all = evaluate(train, oof)

    # refit on ALL train trips, then score student_test
    tr = sample(np.flatnonzero(labelled & ~jitter))
    fixed = int(np.median(best_epochs)) if best_epochs else None
    t0 = time.perf_counter()
    final, _ = fit_model(name, params, X[tr], y[tr].astype(int), fixed_epochs=fixed)
    train_s = time.perf_counter() - t0
    Xt = test[FEATURES].to_numpy(np.float32)
    t0 = time.perf_counter()
    ts = final.predict_proba(Xt)[:, 1]
    inf_s = time.perf_counter() - t0
    n_test_points = test.groupby(["trip_id", "pt"]).ngroups
    test_m = evaluate(test, ts)

    metrics = {
        "model": run_name, "model_type": name, "family": spec["family"], "params": params,
        "params_source": args.params or "defaults in training.py", "seed": SEED, "n_folds": N_FOLDS,
        "max_train_rows": spec["max_train_rows"], "final_epochs": fixed, "features": FEATURES,
        "train_filter": fcfg.to_dict(), "machine": platform.node(),
        "cv_folds": folds, "cv": cv, "cv_oof_overall": oof_all, "student_test": test_m,
        "final_train_rows": int(len(tr)), "final_train_seconds": round(train_s, 1),
        "inference_seconds_student_test": round(inf_s, 3),
        "inference_ms_per_1000_points": round(1000 * inf_s / n_test_points * 1000, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(final, out / "model.joblib")
    train[["trip_id", "pt", "arc_id"]].assign(score=oof.astype(np.float32)).to_parquet(out / "oof_train.parquet", index=False)
    test[["trip_id", "pt", "arc_id"]].assign(score=ts.astype(np.float32)).to_parquet(out / "test_scores.parquet", index=False)

    if history:
        (out / "history.json").write_text(json.dumps(history))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(history["train"]) + 1), history["train"], label="train")
        if history.get("valid"):
            ax.plot(range(1, len(history["valid"]) + 1), history["valid"], label="validation (CV fold 0)")
        unit = "epoch" if name == "mlp" else "boosting iteration"
        ax.set(xlabel=unit, ylabel="log loss", title=f"{name}: training vs validation loss (train partition, CV fold 0)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "history.png", dpi=150)
        plt.close(fig)

    cm = np.array(test_m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not chosen", "chosen"], yticklabels=["wrong arc", "true arc"],
           xlabel="prediction (top-1 per point)", ylabel="truth", title=f"{name}: student_test candidates")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\n[{name}] CV top1 {cv['top1_accuracy']['mean']:.4f} ± {cv['top1_accuracy']['std']:.4f} | "
          f"CV macroF1 {cv['macro_f1']['mean']:.4f} ± {cv['macro_f1']['std']:.4f} | "
          f"CV logloss {cv['log_loss']['mean']:.4f}")
    print(f"[{name}] student_test top1 {test_m['top1_accuracy']:.4f} | macroF1 {test_m['macro_f1']:.4f} | "
          f"logloss {test_m['log_loss']:.4f} | train {train_s:.0f}s | "
          f"inference {metrics['inference_ms_per_1000_points']} ms / 1000 points")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
