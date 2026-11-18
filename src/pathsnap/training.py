"""Step 6 - Shared training utilities: data loading, folds, jitter filter, models, metrics.

Every laptop imports this file, so all models use the SAME folds, filter, seed and metrics.
"""
import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .features import FEATURES

SEED = 42
N_FOLDS = 5


@dataclass(frozen=True)
class TrainFilterConfig:
    """Optional training-only jitter filter.

    DISABLED after scripts/12_check_jitter_filter.py: the rule removed 14% of labelled
    train points, only ~8% of them from the known jitter trips (33, 54) and large shares
    of normal-but-noisy trips (2, 10, 82, 84). Not selective -> all labelled points are used.
    """
    enabled: bool = False
    jitter_progress_max: float = 0.2
    jitter_speed_min: float = 3.0

    def to_dict(self):
        return asdict(self)


def load_features(path) -> pd.DataFrame:
    return pd.read_parquet(path).reset_index(drop=True)


def jitter_rows(df: pd.DataFrame, cfg: TrainFilterConfig = TrainFilterConfig()) -> np.ndarray:
    """Point-level flag (identical for every candidate row of a point). All False when disabled."""
    if not cfg.enabled:
        return np.zeros(len(df), dtype=bool)
    return ((df["progress_ratio"] < cfg.jitter_progress_max)
            & (df["speed_smooth_ms"] > cfg.jitter_speed_min)).to_numpy()


def make_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> dict:
    """trip_id -> fold number. Grouped by trip, deterministic (same on every laptop)."""
    df = df.sort_values(["trip_id", "pt"], kind="stable")
    trips = df["trip_id"].to_numpy()
    fold_of = {}
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(np.zeros(len(df)), groups=trips)):
        for t in np.unique(trips[va]):
            fold_of[t] = k
    return fold_of


# ----------------------------------------------------------------------------- models
MODEL_SPECS = {
    "logreg": {"family": "Linear / probabilistic", "max_train_rows": None,
               "params": {"C": 1.0, "max_iter": 1000}},
    "gnb": {"family": "Probabilistic (Naive Bayes)", "max_train_rows": None,
            "params": {"var_smoothing": 1e-9}},
    "knn": {"family": "Instance-based", "max_train_rows": 100_000,
            "params": {"n_neighbors": 25, "weights": "distance"}},
    "rf": {"family": "Ensemble (bagging)", "max_train_rows": None,
           "params": {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "max_samples": 0.3}},
    "lgbm": {"family": "Boosting", "max_train_rows": None,
             "params": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
                        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}},
    "mlp": {"family": "Neural network", "max_train_rows": None,
            "params": {"hidden_layer_sizes": [64, 32], "alpha": 1e-4, "batch_size": 2048,
                       "learning_rate_init": 1e-3, "max_epochs": 30, "patience": 3}},
    "svm_rbf": {"family": "Kernel (RBF via Nystroem approximation)", "max_train_rows": 200_000,
                "params": {"gamma": 0.05, "n_components": 300, "C": 1.0}},
}


class EpochMLP:
    """MLP trained one epoch at a time so we can record train/validation log loss per epoch
    and stop early when validation loss stops improving."""

    def __init__(self, hidden_layer_sizes, alpha, batch_size, learning_rate_init, max_epochs, patience,
                 fixed_epochs=None):
        self.cfg = dict(hidden_layer_sizes=tuple(hidden_layer_sizes), alpha=alpha, batch_size=batch_size,
                        learning_rate_init=learning_rate_init)
        self.max_epochs, self.patience, self.fixed_epochs = max_epochs, patience, fixed_epochs

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(SEED)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        Xv = self.scaler.transform(X_val) if X_val is not None else None
        self.clf = MLPClassifier(random_state=SEED, **self.cfg)
        sub = rng.choice(len(Xs), size=min(200_000, len(Xs)), replace=False)  # train-loss estimate
        self.history_ = {"train": [], "valid": []}
        best, best_clf, bad, self.best_epoch = np.inf, None, 0, None
        n_epochs = self.fixed_epochs or self.max_epochs
        for e in range(n_epochs):
            perm = rng.permutation(len(Xs))
            self.clf.partial_fit(Xs[perm], y[perm], classes=np.array([0, 1]))
            self.history_["train"].append(float(log_loss(y[sub], self.clf.predict_proba(Xs[sub]), labels=[0, 1])))
            if Xv is not None and self.fixed_epochs is None:
                vl = float(log_loss(y_val, self.clf.predict_proba(Xv), labels=[0, 1]))
                self.history_["valid"].append(vl)
                if vl < best - 1e-4:
                    best, best_clf, bad, self.best_epoch = vl, copy.deepcopy(self.clf), 0, e + 1
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_clf is not None:
            self.clf = best_clf
        if self.best_epoch is None:
            self.best_epoch = len(self.history_["train"])
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))


def build_model(name: str, params: dict):
    p = dict(params)
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(random_state=SEED, **p))])
    if name == "gnb":
        return GaussianNB(**p)
    if name == "knn":
        return Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=-1, **p))])
    if name == "rf":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **p)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **p)
    if name == "svm_rbf":
        return Pipeline([
            ("scale", StandardScaler()),
            ("kernel", Nystroem(gamma=p["gamma"], n_components=p["n_components"], random_state=SEED)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=p["C"], random_state=SEED), cv=3, method="sigmoid")),
        ])
    raise ValueError(f"unknown model {name}")


def fit_model(name, params, X, y, X_val=None, y_val=None, fixed_epochs=None):
    """Returns (fitted model, history dict or None). History only for epoch/iteration-based models."""
    if name == "mlp":
        m = EpochMLP(**params, fixed_epochs=fixed_epochs).fit(X, y, X_val, y_val)
        return m, (m.history_ if m.history_["train"] else None)
    m = build_model(name, params)
    if name == "lgbm" and X_val is not None:
        m.fit(X, y, eval_set=[(X, y), (X_val, y_val)], eval_names=["train", "valid"], eval_metric="binary_logloss")
        h = m.evals_result_
        return m, {"train": list(map(float, h["train"]["binary_logloss"])),
                   "valid": list(map(float, h["valid"]["binary_logloss"]))}
    m.fit(X, y)
    return m, None


# ----------------------------------------------------------------------------- metrics
def evaluate(df: pd.DataFrame, score: np.ndarray) -> dict:
    """Metrics on LABELLED rows. df needs trip_id, pt, y. score = P(candidate is the true arc).

    Top-1: per point, the highest-scoring candidate is the prediction.
    Candidate-level metrics treat 'is this point's top choice' as the predicted class.
    """
    d = df[["trip_id", "pt", "y"]].copy()
    d["score"] = score
    d = d[d["y"].notna()]
    top = d.groupby(["trip_id", "pt"])["score"].idxmax()
    pred = np.zeros(len(d), dtype=int)
    pred[d.index.get_indexer(top.to_numpy())] = 1
    y = d["y"].to_numpy().astype(int)
    p = np.clip(d["score"].to_numpy(), 1e-7, 1 - 1e-7)
    return {
        "top1_accuracy": float(d.loc[top.to_numpy(), "y"].mean()),
        "n_points": int(len(top)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
