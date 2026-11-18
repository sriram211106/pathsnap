"""Step 9 - Tune the HMM on train (out-of-fold scores), confirm on student_test,
and compare classifier-only vs classifier + HMM.

Run on Laptop 1 (needs data/processed + the model's score files):
    python scripts/15_hmm_tune_evaluate.py --model lgbm

Uses outputs/models/<model>/oof_train.parquet (train) and test_scores.parquet (student_test).
Out-of-fold scores are honest: each train trip was scored by a model that never saw it.

Writes outputs/hmm/<model>/:
    grid_train.csv            top-1 + continuity for every (beta, alpha) on train
    hmm_config.json           chosen parameters (picked on TRAIN top-1)
    per_trip.csv              classifier-only vs HMM per trip, both partitions
    student_test_predictions.parquet   point_idx, classifier arc, HMM arc (for maps / report)
    summary.json, grid_heatmap.png, per_trip_change.png
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
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.hmm import HMMConfig, RoadGraph, classifier_only, decode, prepare_trip, route_continuity  # noqa: E402

BETAS = [2, 5, 10, 20, 50, 100]
ALPHAS = [0.5, 1.0, 2.0, 4.0]


def load_partition(part: str, scores: pd.DataFrame, cfg: HMMConfig):
    trips = {}
    t0 = time.perf_counter()
    for tid, sc in tqdm(scores.groupby("trip_id", sort=True), desc=f"{part}: shortest paths"):
        tdir = ROOT / "data" / "processed" / part / tid
        pts = pd.read_parquet(tdir / "points.parquet", columns=["point_idx", "x", "y", "segment_id"])
        net = pd.read_parquet(tdir / "arcs.parquet")
        lab = pd.read_parquet(tdir / "labels.parquet", columns=["label_arc_id"])["label_arc_id"].to_numpy()
        prep = prepare_trip(pts, net, sc, cfg, RoadGraph(net))
        trips[tid] = {"prep": prep, "label": lab, "net": net, "point_idx": pts["point_idx"].to_numpy()}
    return trips, time.perf_counter() - t0


def score_preds(trips: dict, preds: dict) -> dict:
    hit = lab = ok = tot = 0
    per_trip = {}
    for tid, t in trips.items():
        p, y = preds[tid], t["label"]
        m = y >= 0
        h = int((p[m] == y[m]).sum())
        c_ok, c_tot = route_continuity(p, t["prep"]["segment_id"], t["net"])
        hit, lab, ok, tot = hit + h, lab + int(m.sum()), ok + c_ok, tot + c_tot
        per_trip[tid] = {"top1": h / max(m.sum(), 1), "continuity": c_ok / max(c_tot, 1), "n_labelled": int(m.sum())}
    return {"top1": hit / max(lab, 1), "continuity": ok / max(tot, 1), "per_trip": per_trip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    mdir = ROOT / "outputs" / "models" / args.model
    out = ROOT / "outputs" / "hmm" / args.model
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = HMMConfig()

    train, prep_s_tr = load_partition("train", pd.read_parquet(mdir / "oof_train.parquet"), base_cfg)
    test, prep_s_te = load_partition("student_test", pd.read_parquet(mdir / "test_scores.parquet"), base_cfg)

    # ---- classifier-only reference
    clf_tr = score_preds(train, {t: classifier_only(v["prep"]) for t, v in train.items()})
    clf_te = score_preds(test, {t: classifier_only(v["prep"]) for t, v in test.items()})
    print(f"\nClassifier only   train top1 {clf_tr['top1']:.4f} cont {clf_tr['continuity']:.4f} | "
          f"student_test top1 {clf_te['top1']:.4f} cont {clf_te['continuity']:.4f}")

    # ---- grid search on TRAIN only
    rows = []
    for b in BETAS:
        for a in ALPHAS:
            r = score_preds(train, {t: decode(v["prep"], b, a, base_cfg.disconnected_margin)[0] for t, v in train.items()})
            rows.append({"beta_m": b, "emission_weight": a, "train_top1": r["top1"], "train_continuity": r["continuity"]})
            print(f"  beta {b:>4}  alpha {a:>3}:  train top1 {r['top1']:.4f}  continuity {r['continuity']:.4f}")
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "grid_train.csv", index=False)
    best = grid.sort_values(["train_top1", "train_continuity"], ascending=False).iloc[0]
    cfg = HMMConfig(beta_m=float(best["beta_m"]), emission_weight=float(best["emission_weight"]))
    (out / "hmm_config.json").write_text(json.dumps({**cfg.to_dict(), "chosen_on": "train out-of-fold top-1",
                                                     "scores_from_model": args.model}, indent=2))

    # ---- final: chosen params on both partitions
    t0 = time.perf_counter()
    dec_te = {t: decode(v["prep"], cfg.beta_m, cfg.emission_weight, cfg.disconnected_margin) for t, v in test.items()}
    dec_s = time.perf_counter() - t0
    hmm_te = score_preds(test, {t: d[0] for t, d in dec_te.items()})
    hmm_tr = score_preds(train, {t: decode(v["prep"], cfg.beta_m, cfg.emission_weight, cfg.disconnected_margin)[0]
                                 for t, v in train.items()})
    n_te_pts = sum(v["prep"]["n"] for v in test.values())
    disc_share = sum(d[1].sum() for d in dec_te.values()) / max(n_te_pts, 1)

    per = []
    for part, clf, hmm in (("train", clf_tr, hmm_tr), ("student_test", clf_te, hmm_te)):
        for tid in clf["per_trip"]:
            c, h = clf["per_trip"][tid], hmm["per_trip"][tid]
            per.append({"partition": part, "trip_id": tid, "n_labelled": c["n_labelled"],
                        "clf_top1": c["top1"], "hmm_top1": h["top1"], "change": h["top1"] - c["top1"],
                        "clf_continuity": c["continuity"], "hmm_continuity": h["continuity"]})
    per = pd.DataFrame(per)
    per.to_csv(out / "per_trip.csv", index=False)

    rows = []
    for tid, v in test.items():
        rows.append(pd.DataFrame({"trip_id": tid, "point_idx": v["point_idx"],
                                  "label_arc_id": v["label"], "clf_arc_id": classifier_only(v["prep"]),
                                  "hmm_arc_id": dec_te[tid][0], "hmm_disconnected_step": dec_te[tid][1]}))
    pd.concat(rows, ignore_index=True).to_parquet(out / "student_test_predictions.parquet", index=False)

    summary = {
        "model": args.model, "chosen": cfg.to_dict(),
        "train": {"clf_top1": clf_tr["top1"], "hmm_top1": hmm_tr["top1"],
                  "clf_continuity": clf_tr["continuity"], "hmm_continuity": hmm_tr["continuity"]},
        "student_test": {"clf_top1": clf_te["top1"], "hmm_top1": hmm_te["top1"],
                         "clf_continuity": clf_te["continuity"], "hmm_continuity": hmm_te["continuity"],
                         "share_points_after_disconnected_step": float(disc_share)},
        "seconds": {"prepare_train": round(prep_s_tr, 1), "prepare_student_test": round(prep_s_te, 1),
                    "decode_student_test": round(dec_s, 2)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nChosen on TRAIN: beta = {cfg.beta_m} m, alpha = {cfg.emission_weight}")
    print(f"{'':16}{'top-1 clf':>11}{'top-1 HMM':>11}{'cont clf':>10}{'cont HMM':>10}")
    print(f"{'train (OOF)':16}{clf_tr['top1']:>11.4f}{hmm_tr['top1']:>11.4f}{clf_tr['continuity']:>10.4f}{hmm_tr['continuity']:>10.4f}")
    print(f"{'student_test':16}{clf_te['top1']:>11.4f}{hmm_te['top1']:>11.4f}{clf_te['continuity']:>10.4f}{hmm_te['continuity']:>10.4f}")
    for part in ("train", "student_test"):
        g = per[per["partition"] == part]
        print(f"  {part}: trips improved {int((g['change'] > 0.001).sum())}, worse {int((g['change'] < -0.001).sum())},"
              f" unchanged {int((g['change'].abs() <= 0.001).sum())}")
    print("  student_test per trip:")
    print(per[per["partition"] == "student_test"][["trip_id", "clf_top1", "hmm_top1", "change", "hmm_continuity"]]
          .round(4).to_string(index=False))
    print(f"  student_test points decoded right after a disconnected step: {disc_share:.2%}")
    print(f"  timing: shortest paths {prep_s_te:.1f}s + Viterbi {dec_s:.2f}s for all student_test trips")

    piv = grid.pivot(index="beta_m", columns="emission_weight", values="train_top1")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
    ax.set(xticks=range(len(piv.columns)), xticklabels=piv.columns, yticks=range(len(piv.index)), yticklabels=piv.index,
           xlabel="emission weight (alpha)", ylabel="beta (m)", title=f"HMM grid: train top-1 ({args.model} scores)")
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im)
    fig.tight_layout(); fig.savefig(out / "grid_heatmap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for part, col in (("train", "#2a6fdb"), ("student_test", "#e07b39")):
        g = per[per["partition"] == part]
        ax.scatter(g["clf_top1"], g["hmm_top1"], label=part, color=col, s=20)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set(xlabel="classifier-only top-1 (per trip)", ylabel="classifier + HMM top-1 (per trip)",
           title="Per-trip effect of HMM (above line = improved)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "per_trip_change.png", dpi=150); plt.close(fig)
    print(f"\nSaved results under {out}/")


if __name__ == "__main__":
    main()
