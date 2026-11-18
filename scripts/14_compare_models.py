"""Step 7 - Compare all trained models (+ rule baselines) and support model selection.

Run on Laptop 1 (needs data/processed for the road network) - rerun whenever a model finishes:
    python scripts/14_compare_models.py

Reads outputs/models/<model>/{metrics.json, oof_train.parquet, test_scores.parquet} for every
model folder present, and outputs/models/baselines.json.

Adds metrics that need the road network or sequences:
  route continuity : consecutive points (same segment) whose predicted arcs are the same arc or
                     directly connected (arc i ends where arc i+1 starts). Also computed for the
                     reference labels themselves, as a realistic ceiling.
  per-trip top-1   : mean over trips and the worst trip (sequence-level view).

Writes outputs/comparison/: model_comparison.csv, model_comparison.md and figures.
Selection rule: rank by CV top-1 (primary, train only); student_test only confirms.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "outputs" / "models"
OUT = ROOT / "outputs" / "comparison"
ORDER = ["logreg", "gnb", "knn", "rf", "lgbm", "mlp", "svm_rbf", "lgbm_tuned"]


def load_truth(part):
    f = pd.read_parquet(ROOT / "data" / "features" / f"{part}.parquet",
                        columns=["trip_id", "pt", "arc_id", "segment_id", "y"])
    return f


def network_uv(part, trips):
    uv = {}
    for t in trips:
        a = pd.read_parquet(ROOT / "data" / "processed" / part / t / "arcs.parquet", columns=["u", "v"])
        uv[t] = (a["u"].to_numpy(), a["v"].to_numpy())
    return uv


def continuity(seq: pd.DataFrame, uv) -> float:
    """seq: one row per point with trip_id, pt, segment_id, arc (sorted by trip, pt)."""
    ok_total, n_total = 0, 0
    for t, g in seq.groupby("trip_id", sort=False):
        u, v = uv[t]
        a = g["arc"].to_numpy()
        s = g["segment_id"].to_numpy()
        same_seg = s[1:] == s[:-1]
        a0, a1 = a[:-1][same_seg], a[1:][same_seg]
        ok = (a0 == a1) | (v[a0] == u[a1])
        ok_total += int(ok.sum())
        n_total += len(ok)
    return ok_total / max(n_total, 1)


def top1_table(truth: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    d = truth.assign(score=score)
    top = d.loc[d.groupby(["trip_id", "pt"])["score"].idxmax()]
    return top.rename(columns={"arc_id": "arc"}).sort_values(["trip_id", "pt"])


def seq_metrics(truth, score, uv):
    top = top1_table(truth, score)
    lab = top[top["y"].notna()]
    per_trip = lab.groupby("trip_id")["y"].mean()
    return {"route_continuity": continuity(top, uv),
            "per_trip_top1_mean": float(per_trip.mean()),
            "worst_trip_top1": float(per_trip.min()),
            "worst_trip": str(per_trip.idxmin())}


def reference_continuity(truth, uv):
    """Continuity of the reference labels themselves (labelled points only, same definition)."""
    lab = truth[truth["y"] == 1][["trip_id", "pt", "segment_id", "arc_id"]].rename(columns={"arc_id": "arc"})
    return continuity(lab.sort_values(["trip_id", "pt"]), uv)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    present = [m for m in ORDER if (MODELS_DIR / m / "metrics.json").exists()]
    print("Models found:", present)
    tr_truth, te_truth = load_truth("train"), load_truth("student_test")
    uv_tr = network_uv("train", tr_truth["trip_id"].unique())
    uv_te = network_uv("student_test", te_truth["trip_id"].unique())
    ref_cont_tr, ref_cont_te = reference_continuity(tr_truth, uv_tr), reference_continuity(te_truth, uv_te)

    rows, folds = [], {}
    for m in present:
        mt = json.loads((MODELS_DIR / m / "metrics.json").read_text())
        oof = pd.read_parquet(MODELS_DIR / m / "oof_train.parquet")
        ts = pd.read_parquet(MODELS_DIR / m / "test_scores.parquet")
        assert (oof["arc_id"].to_numpy() == tr_truth["arc_id"].to_numpy()).all(), f"{m}: train row order mismatch"
        assert (ts["arc_id"].to_numpy() == te_truth["arc_id"].to_numpy()).all(), f"{m}: test row order mismatch"
        sq_tr = seq_metrics(tr_truth, oof["score"].to_numpy(), uv_tr)
        sq_te = seq_metrics(te_truth, ts["score"].to_numpy(), uv_te)
        cv, te = mt["cv"], mt["student_test"]
        folds[m] = [f["top1_accuracy"] for f in mt["cv_folds"]]
        key_cfg = ", ".join(f"{k}={v}" for k, v in mt["params"].items())
        if mt.get("max_train_rows"):
            key_cfg += f", sample={mt['max_train_rows']:,} rows"
        rows.append({
            "model": m, "family": mt["family"], "key_configuration": key_cfg,
            "cv_macro_f1_mean": cv["macro_f1"]["mean"], "cv_macro_f1_std": cv["macro_f1"]["std"],
            "cv_top1_mean": cv["top1_accuracy"]["mean"], "cv_top1_std": cv["top1_accuracy"]["std"],
            "cv_log_loss": cv["log_loss"]["mean"],
            "cv_route_continuity": sq_tr["route_continuity"],
            "test_top1": te["top1_accuracy"], "test_macro_f1": te["macro_f1"],
            "test_precision": te["precision"], "test_recall": te["recall"], "test_log_loss": te["log_loss"],
            "test_route_continuity": sq_te["route_continuity"],
            "test_per_trip_top1_mean": sq_te["per_trip_top1_mean"],
            "test_worst_trip_top1": sq_te["worst_trip_top1"], "test_worst_trip": sq_te["worst_trip"],
            "train_seconds": mt["final_train_seconds"], "inference_ms_per_1000_points": mt["inference_ms_per_1000_points"],
        })
    res = pd.DataFrame(rows)

    base = json.loads((MODELS_DIR / "baselines.json").read_text()) if (MODELS_DIR / "baselines.json").exists() else {}
    base_rows = []
    for b in ("nearest_arbitrary", "nearest_heading"):
        if b in base.get("train", {}):
            base_rows.append({"model": f"[baseline] {b}", "family": "Rule (no learning)",
                              "cv_top1_mean": base["train"][b]["cv_mean"],
                              "test_top1": base["student_test"][b]["top1_accuracy"]})
    table = pd.concat([res, pd.DataFrame(base_rows)], ignore_index=True)
    table.to_csv(OUT / "model_comparison.csv", index=False)

    # report-style markdown table (docs' column set)
    def fmt(x, pct=True, d=4):
        return "N/A" if pd.isna(x) else (f"{x:.{d}f}")
    md = ["| Model | Algorithm family | Key configuration | CV Macro F1 | CV Top-1 | Student-test Top-1 | "
          "Route continuity (test) | Student-test Macro F1 | Log loss (test) | Inference (ms/1000 pts) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        is_base = str(r["model"]).startswith("[baseline]")
        md.append("| " + " | ".join([
            str(r["model"]), str(r["family"]),
            "N/A (rule)" if is_base else str(r["key_configuration"]),
            "N/A - no probabilities/classifier" if is_base else f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f}",
            fmt(r["cv_top1_mean"]), fmt(r["test_top1"]),
            "N/A" if is_base else fmt(r["test_route_continuity"]),
            "N/A" if is_base else fmt(r["test_macro_f1"]),
            "N/A - rule gives no probabilities" if is_base else fmt(r["test_log_loss"]),
            "N/A" if is_base else f"{r['inference_ms_per_1000_points']:.2f}",
        ]) + " |")
    md.append(f"\nReference labels' own route continuity: train {ref_cont_tr:.4f}, student_test {ref_cont_te:.4f} "
              f"(same definition; realistic ceiling at 1 Hz).")
    (OUT / "model_comparison.md").write_text("\n".join(md))

    # --- print summary + selection
    show = ["model", "cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "cv_log_loss", "test_top1",
            "test_macro_f1", "test_route_continuity", "test_worst_trip_top1", "inference_ms_per_1000_points"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res[show].sort_values("cv_top1_mean", ascending=False).round(4).to_string(index=False))
        if base_rows:
            print("\nBaselines:", pd.DataFrame(base_rows)[["model", "cv_top1_mean", "test_top1"]].round(4).to_string(index=False))
    print(f"\nReference labels' route continuity: train {ref_cont_tr:.4f} | student_test {ref_cont_te:.4f}")
    rank = res.sort_values("cv_top1_mean", ascending=False)
    print(f"\nRanking by CV top-1 (selection is made on train only): {', '.join(rank['model'])}")

    # --- figures
    res = res.set_index("model").loc[[m for m in ORDER if m in present]]
    x = np.arange(len(res))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, res["cv_macro_f1_mean"], yerr=res["cv_macro_f1_std"], capsize=4, color="#2a6fdb")
    ax.set(xticks=x, xticklabels=res.index, ylabel="Macro F1 (mean ± std over 5 folds)",
           title="Cross-validation Macro F1 by model (train, grouped by trip)")
    ax.set_ylim(max(0, res["cv_macro_f1_mean"].min() - 0.1), 1)
    fig.tight_layout(); fig.savefig(OUT / "01_cv_macro_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    ax.bar(x - w / 2, res["test_top1"], w, label="Top-1 accuracy", color="#2a6fdb")
    ax.bar(x + w / 2, res["test_macro_f1"], w, label="Macro F1", color="#e07b39")
    for b in base_rows:
        ax.axhline(b["test_top1"], ls="--", lw=1, color="grey")
        ax.text(len(res) - 0.5, b["test_top1"], b["model"].replace("[baseline] ", "baseline: "), fontsize=7, va="bottom", ha="right")
    ax.set(xticks=x, xticklabels=res.index, title="Student-test Top-1 accuracy and Macro F1 (dashed = rule baselines, top-1)")
    ax.set_ylim(0.4, 1)
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "02_student_test_top1_macro_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, res["cv_log_loss"], w, label="CV (train)", color="#2a6fdb")
    ax.bar(x + w / 2, res["test_log_loss"], w, label="Student-test", color="#e07b39")
    ax.set(xticks=x, xticklabels=res.index, ylabel="Log loss (lower is better)", title="Log loss by model")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "03_log_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for m in res.index:
        ax.plot(range(len(folds[m])), folds[m], marker="o", label=m)
    if "train" in base and "nearest_heading" in base["train"]:
        ax.plot(range(5), base["train"]["nearest_heading"]["cv_fold_top1"], ls="--", color="grey", label="baseline: nearest+heading")
    ax.set(xticks=range(5), xlabel="CV fold", ylabel="Top-1 accuracy", title="Per-fold Top-1 (stability across folds; train)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "04_cv_fold_stability.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, res["test_route_continuity"], color="#2a6fdb")
    ax.axhline(ref_cont_te, ls="--", color="black", lw=1)
    ax.text(len(res) - 0.5, ref_cont_te, "reference labels", fontsize=8, va="bottom", ha="right")
    ax.set(xticks=x, xticklabels=res.index, ylabel="Share of consecutive points connected",
           title="Route continuity of classifier-only predictions (student-test)")
    ax.set_ylim(max(0, res["test_route_continuity"].min() - 0.05), 1)
    fig.tight_layout(); fig.savefig(OUT / "05_route_continuity.png", dpi=150); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(x, res["train_seconds"], color="#2a6fdb")
    a1.set(xticks=x, xticklabels=res.index, ylabel="seconds", title="Final training time (all train trips)")
    a2.bar(x, res["inference_ms_per_1000_points"], color="#e07b39")
    a2.set(xticks=x, xticklabels=res.index, ylabel="ms per 1000 GPS points (log scale)", yscale="log",
           title="Inference time (student-test)")
    fig.tight_layout(); fig.savefig(OUT / "06_train_inference_time.png", dpi=150); plt.close(fig)

    print(f"\nSaved table + figures under {OUT}/")


if __name__ == "__main__":
    main()
