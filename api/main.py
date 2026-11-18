"""Step 11 - PathSnap FastAPI backend (runs on LAPTOP 1 only).

Start from the repo root:
    set PATHSNAP_EVAL_DIR=C:\\path\\to\\evaluator_test          (Windows; folder with the 10 trip folders)
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Rehearsal with student_test instead:
    set PATHSNAP_EVAL_DIR=data\\raw\\student_test

Endpoints
    GET  /health            status + which model is loaded
    GET  /trips             trip IDs available for evaluation
    POST /predict           {"trip_id": "00000015"} -> predicted route + server-side metrics

Integrity rules implemented here:
  * predict_route() reads ONLY .track, .nodes, .arcs - never .route.
  * The reference route is read afterwards in evaluate_server_side(), used to compute
    metrics, and NEVER returned to the client (only numbers go back).
  * Every request is logged to this terminal (required at evaluation).
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.candidates import CandidateConfig, build_arc_index, generate_candidates  # noqa: E402
from pathsnap.features import FEATURES, FeatureConfig, build_features  # noqa: E402
from pathsnap.hmm import HMMConfig, RoadGraph, classifier_only, decode, prepare_trip, route_continuity  # noqa: E402
from pathsnap.io import list_trips, load_arcs, load_nodes, load_route, load_track, trip_files  # noqa: E402
from pathsnap.labels import LabelConfig, align_points_to_route  # noqa: E402
from pathsnap.preprocess import PreprocessConfig, preprocess_track, project_network, trip_projector  # noqa: E402

log = logging.getLogger("uvicorn.error")  # shows in the uvicorn terminal

ART = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "evaluator_test"))


def _fields(cls, d):
    return {k: v for k, v in d.items() if k in cls.__dataclass_fields__}


MANIFEST = json.loads((ART / "manifest.json").read_text())
if MANIFEST["features"] != FEATURES:
    raise RuntimeError("Feature list in artifacts does not match the code - re-export artifacts.")
MODEL = joblib.load(ART / "model.joblib")
PCFG = PreprocessConfig(**_fields(PreprocessConfig, MANIFEST["preprocess_config"]))
CCFG = CandidateConfig(**_fields(CandidateConfig, MANIFEST["candidate_config"]))
FCFG = FeatureConfig(**_fields(FeatureConfig, MANIFEST["feature_config"]))
HCFG = HMMConfig(**_fields(HMMConfig, MANIFEST["hmm_config"]))

app = FastAPI(title="PathSnap API", version="1.0")


class PredictRequest(BaseModel):
    trip_id: str


@app.on_event("startup")
def _startup():
    log.info(f"PathSnap API | model={MANIFEST['model']} | artifacts={ART} | eval dir={EVAL_DIR}")
    log.info(f"Trips available: {list_trips(EVAL_DIR) if EVAL_DIR.exists() else 'EVAL DIR NOT FOUND'}")


@app.get("/health")
def health():
    return {"status": "ok", "model": MANIFEST["model"], "eval_dir_exists": EVAL_DIR.exists()}


@app.get("/trips")
def trips():
    if not EVAL_DIR.exists():
        raise HTTPException(500, f"evaluation folder not found: {EVAL_DIR}")
    return {"trips": list_trips(EVAL_DIR)}


def predict_route(trip_dir: Path) -> dict:
    """The full prediction pipeline. Uses ONLY GPS + road network (no reference route)."""
    t = {}
    t0 = time.perf_counter()
    f = trip_files(trip_dir)
    track, nodes, arcs = load_track(f["track"]), load_nodes(f["nodes"]), load_arcs(f["arcs"])
    proj, _ = trip_projector(track)
    pts = preprocess_track(track, proj, PCFG)
    net = project_network(nodes, arcs, proj)
    t["preprocess_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cand = generate_candidates(pts["x"].to_numpy(), pts["y"].to_numpy(), net, CCFG, build_arc_index(net))
    feats = build_features(pts, net, cand, FCFG)
    t["candidates_features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    score = MODEL.predict_proba(feats[FEATURES].to_numpy(np.float32))[:, 1]
    t["model_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prep = prepare_trip(pts, net, feats[["pt", "arc_id"]].assign(score=score), HCFG, RoadGraph(net))
    hmm_arcs, disc = decode(prep, HCFG.beta_m, HCFG.emission_weight, HCFG.disconnected_margin)
    t["hmm_s"] = time.perf_counter() - t0
    return {"pts": pts, "net": net, "nodes": nodes, "hmm": hmm_arcs, "clf": classifier_only(prep),
            "n_candidates": len(cand), "disconnected": disc, "timing": t}


def evaluate_server_side(trip_dir: Path, pts, net, hmm_arcs, clf_arcs) -> dict:
    """Reference route used HERE ONLY, after prediction. Only numbers leave this function."""
    tid = trip_dir.name
    route = load_route(trip_dir / f"{tid}.route")
    if route is None:
        return {"available": False}
    lab = align_points_to_route(pts["x"].to_numpy(), pts["y"].to_numpy(), net, route, LabelConfig())
    y = lab["label_arc_id"].to_numpy()
    m = y >= 0
    seg = pts["segment_id"].to_numpy()
    c_ok, c_n = route_continuity(hmm_arcs, seg, net)
    k_ok, k_n = route_continuity(clf_arcs, seg, net)
    return {
        "available": True,
        "n_points": int(len(y)), "n_scored_points": int(m.sum()),
        "accuracy_hmm": float((hmm_arcs[m] == y[m]).mean()) if m.any() else None,
        "accuracy_classifier_only": float((clf_arcs[m] == y[m]).mean()) if m.any() else None,
        "route_continuity_hmm": c_ok / max(c_n, 1),
        "route_continuity_classifier_only": k_ok / max(k_n, 1),
        "note": "Accuracy = share of GPS points whose predicted road segment equals the reference segment, "
                "on points that have a reference segment within 50 m. Computed on the server.",
    }


def route_geometry(arc_seq: np.ndarray, net, nodes) -> list:
    """Predicted route as a list of [lat, lon] lines (one per arc, consecutive duplicates removed)."""
    keep = np.r_[True, arc_seq[1:] != arc_seq[:-1]]
    arcs = arc_seq[keep]
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    u, v = net["u"].to_numpy()[arcs], net["v"].to_numpy()[arcs]
    return [[[float(lat[a]), float(lon[a])], [float(lat[b]), float(lon[b])]] for a, b in zip(u, v)]


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    trip_dir = EVAL_DIR / tid
    if not trip_dir.is_dir() or tid not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {tid}")
    t_start = time.perf_counter()
    log.info(f"[request] POST /predict trip={tid}")

    r = predict_route(trip_dir)
    metrics = evaluate_server_side(trip_dir, r["pts"], r["net"], r["hmm"], r["clf"])
    total = time.perf_counter() - t_start

    pts = r["pts"]

    def num(k):
        v = metrics.get(k)
        return float("nan") if v is None else v

    log.info(f"[result ] trip={tid} points={len(pts)} candidates={r['n_candidates']} | "
             f"accuracy HMM={num('accuracy_hmm'):.2%} "
             f"(classifier only {num('accuracy_classifier_only'):.2%}) | "
             f"continuity {num('route_continuity_hmm'):.3f} | {total:.2f}s "
             f"[{', '.join(f'{k} {v:.2f}' for k, v in r['timing'].items())}]")

    return {
        "trip_id": tid,
        "model": MANIFEST["model"],
        "gps_points": pts[["lat", "lon"]].round(6).to_numpy().tolist(),
        "predicted_route": route_geometry(r["hmm"], r["net"], r["nodes"]),
        "metrics": metrics,
        "timing_s": {**{k: round(v, 3) for k, v in r["timing"].items()}, "total": round(total, 3)},
    }
