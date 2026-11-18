"""Step 8 - Bayesian hyperparameter optimisation of LightGBM (Optuna TPE), train partition only.

Run on the strongest laptop:
    python scripts/16_bayes_tune_lgbm.py --trials 30
    (optional) --timeout-min 120    stop after this many minutes even if trials remain

Objective : maximise mean top-1 accuracy over 3-fold grouped-by-trip CV on the 80 train trips.
Trial 0   : the UNTUNED settings (enqueued), so the search starts from our baseline.
Pruning   : MedianPruner stops clearly worse trials after their first fold(s).
The study is stored in SQLite, so an interrupted run can be resumed by running the command again.

Writes outputs/tuning/lgbm/:
    study.db, trials.csv, best_params.json, search_space.json,
    optimisation_history.png, param_importances.png
Then refit with:  python scripts/13_train_model.py --model lgbm --params outputs/tuning/lgbm/best_params.json --name lgbm_tuned
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.training import MODEL_SPECS, SEED, evaluate, load_features, make_folds  # noqa: E402

SEARCH_SPACE = {
    "n_estimators": "int 100-1000 step 50",
    "learning_rate": "float 0.01-0.2 log",
    "num_leaves": "int 15-255 log",
    "min_child_samples": "int 10-500 log",
    "subsample": "float 0.5-1.0",
    "colsample_bytree": "float 0.5-1.0",
    "reg_lambda": "float 0.001-10 log",
    "reg_alpha": "float 0.001-10 log",
}


def suggest(trial, max_trees=1000):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, max_trees, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout-min", type=float, default=None)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="fraction of labelled TRAINING-FOLD points used to fit each trial (validation always full)")
    ap.add_argument("--max-trees", type=int, default=1000)
    args = ap.parse_args()
    out = ROOT / "outputs" / "tuning" / "lgbm"
    out.mkdir(parents=True, exist_ok=True)
    space = dict(SEARCH_SPACE, n_estimators=f"int 100-{args.max_trees} step 50")
    (out / "search_space.json").write_text(json.dumps(
        {"space": space, "objective": f"maximise mean top-1, {args.folds}-fold grouped CV (train only)",
         "fit_sample": f"{args.sample_frac:.0%} of training-fold points per trial (whole candidate groups); "
                       "validation folds scored in full",
         "sampler": "TPE (seed 42, 8 random start-up trials)", "pruner": "MedianPruner"}, indent=2))

    train = load_features(ROOT / "data" / "features" / "train.parquet")
    lab = train[train["y"].notna()].reset_index(drop=True)
    X = lab[FEATURES].to_numpy(np.float32)
    y = lab["y"].to_numpy().astype(int)
    fold_of = make_folds(lab, n_folds=args.folds)
    fold = lab["trip_id"].map(fold_of).to_numpy()
    # fixed point sample (same for every trial -> trials are comparable)
    rng = np.random.default_rng(SEED)
    pkey = lab["trip_id"].astype(str) + "_" + lab["pt"].astype(str)
    upts = pkey.unique()
    keep_pts = set(rng.choice(upts, size=int(len(upts) * args.sample_frac), replace=False)) \
        if args.sample_frac < 1 else None
    in_sample = pkey.isin(keep_pts).to_numpy() if keep_pts is not None else np.ones(len(lab), dtype=bool)
    print(f"Tuning on {len(lab):,} labelled train rows ({in_sample.mean():.0%} used for fitting), "
          f"{args.folds}-fold grouped CV, trees <= {args.max_trees}")

    def objective(trial):
        params = suggest(trial, args.max_trees)
        scores = []
        t0 = time.perf_counter()
        for k in range(args.folds):
            tr, va = np.flatnonzero((fold != k) & in_sample), np.flatnonzero(fold == k)
            m = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, subsample_freq=1, **params)
            m.fit(X[tr], y[tr])
            scores.append(evaluate(lab.iloc[va], m.predict_proba(X[va])[:, 1])["top1_accuracy"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", [round(s, 5) for s in scores])
        trial.set_user_attr("seconds", round(time.perf_counter() - t0, 1))
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name="lgbm_top1", direction="maximize", storage=f"sqlite:///{(out / 'study.db').as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    if len(study.trials) == 0:
        base = dict(MODEL_SPECS["lgbm"]["params"])
        base.pop("subsample_freq", None)
        base.update({"reg_lambda": 1e-3, "reg_alpha": 1e-3})  # ~ no regularisation (untuned used 0)
        base["n_estimators"] = min(base["n_estimators"], args.max_trees)
        study.enqueue_trial(base, user_attrs={"note": "untuned baseline settings"})

    remaining = max(args.trials - len(study.trials), 0)
    t0 = time.perf_counter()
    if remaining:
        study.optimize(objective, n_trials=remaining,
                       timeout=args.timeout_min * 60 if args.timeout_min else None, show_progress_bar=False)
    total_s = time.perf_counter() - t0

    df = study.trials_dataframe()
    df.to_csv(out / "trials.csv", index=False)
    done = df[df["state"] == "COMPLETE"]
    best = study.best_trial
    best_params = {**best.params, "subsample_freq": 1}
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2))

    first = study.trials[0]
    print(f"\nTrials: {len(df)} total | completed {len(done)} | pruned {int((df['state'] == 'PRUNED').sum())}")
    if first.value is not None:
        print(f"Trial 0 (untuned settings): mean top-1 {first.value:.4f}")
    print(f"Best trial #{best.number}: mean top-1 {best.value:.4f}  folds {best.user_attrs.get('fold_top1')}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Search time this run: {total_s / 60:.1f} min")

    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        ax = plot_optimization_history(study)
        ax.set_title("Bayesian optimisation history (LightGBM, train 3-fold grouped CV)")
        ax.figure.set_size_inches(8, 4.5); ax.figure.tight_layout()
        ax.figure.savefig(out / "optimisation_history.png", dpi=150); plt.close(ax.figure)
        if len(done) >= 3:
            ax = plot_param_importances(study)
            ax.set_title("Hyperparameter importance (Optuna fANOVA)")
            ax.figure.set_size_inches(7, 4.5); ax.figure.tight_layout()
            ax.figure.savefig(out / "param_importances.png", dpi=150); plt.close(ax.figure)
    except Exception as e:  # plotting is optional; the CSV + DB are the real record
        print(f"[warn] plotting skipped: {e}")
    print(f"\nSaved study and results under {out}/")
    print("Next: python scripts/13_train_model.py --model lgbm --params outputs/tuning/lgbm/best_params.json --name lgbm_tuned")


if __name__ == "__main__":
    main()
