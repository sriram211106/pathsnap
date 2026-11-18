# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR campus).

PathSnap matches noisy GPS observations to a road network and reconstructs the route a vehicle actually drove.
For every GPS point it generates nearby candidate road segments (arcs), scores each candidate with a trained
classifier (LightGBM, tuned with Bayesian optimisation), and decodes the most likely **connected** route with an
HMM + Viterbi. The system is served by a FastAPI backend (Laptop 1) and a Streamlit evaluator UI (Laptop 2).

```
raw GPS + road network ─► preprocess ─► candidate roads ─► features ─► LightGBM ─► HMM + Viterbi ─► route
                                                                                          (API, Laptop 1)
                                                         evaluator UI (Laptop 2) ◄────────┘
```

---

## Repository layout

```
pathsnap/
├── src/pathsnap/            shared pipeline code (used by training scripts AND the API)
│   ├── io.py                loaders for .track / .nodes / .arcs / .route
│   ├── geo.py               per-trip UTM projection, distances
│   ├── preprocess.py        cleaning, speed/heading, segmentation at gaps
│   ├── labels.py            per-point reference labels (order-preserving alignment to the route)
│   ├── candidates.py        candidate road segments per GPS point
│   ├── features.py          features per (point, candidate) pair
│   ├── training.py          models, grouped CV folds, metrics
│   └── hmm.py               HMM + Viterbi route reconstruction
├── scripts/                 numbered pipeline steps (run in order, see below)
├── api/main.py              FastAPI backend (Laptop 1)
├── ui/app.py                Streamlit evaluator UI (Laptop 2)
├── artifacts/               final model + all configs used by the API (model.joblib, manifest.json)
├── outputs/                 results: EDA, labels, candidates, features, models, comparison, tuning, HMM
├── data/                    NOT in git (raw data, processed trips, feature tables)
└── requirements.txt
```

---

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Place the provided data (not included in the repo):

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/...
```

The evaluator-testing data is **never** placed in `data/` and is used only by the API on Laptop 1 at evaluation time.

---

## Reproducing the pipeline (run from the repo root, in order)

| Step | Command | Output |
|---|---|---|
| 1. Exploration | `python scripts/01_eda_inventory.py`<br>`python scripts/02_eda_plots.py`<br>`python scripts/03_eda_folium_map.py --trip-dir data/raw/train/00000001`<br>`python scripts/04_eda_far_points.py` | `outputs/eda/` (inventory, quality table, figures, Folium maps) |
| 2. Preprocessing | `python scripts/05_preprocess.py`<br>`python scripts/06_check_stationary.py` | `data/processed/<partition>/<trip>/points.parquet, arcs.parquet` |
| 3. Labels | `python scripts/07_build_labels.py`<br>`python scripts/08_check_label_direction.py` | `labels.parquet` per trip, `outputs/labels/` |
| 4. Candidates | `python scripts/09_candidate_sweep.py`<br>`python scripts/10_build_candidates.py --radius 40 --k 30` | `candidates.parquet` per trip, `outputs/candidates/` |
| 5. Features | `python scripts/11_build_features.py` | `data/features/{train,student_test}.parquet` |
| 6. Models | `python scripts/12_check_jitter_filter.py`<br>`python scripts/13_train_model.py --model <logreg\|gnb\|knn\|rf\|lgbm\|mlp>`<br>`python scripts/13b_baselines.py`<br>`python scripts/13c_feature_importance.py --model <name>` | `outputs/models/<model>/` |
| 7. Comparison | `python scripts/14_compare_models.py` | `outputs/comparison/` (table + figures) |
| 8. Bayesian tuning | `python scripts/16_bayes_tune_lgbm.py --trials 25 --timeout-min 3 --sample-frac 0.2 --max-trees 300`<br>`python scripts/13_train_model.py --model lgbm --params outputs/tuning/lgbm/best_params.json --name lgbm_tuned` | `outputs/tuning/lgbm/` (Optuna study, trials, best params) |
| 9. HMM + Viterbi | `python scripts/15_hmm_tune_evaluate.py --model lgbm_tuned` | `outputs/hmm/lgbm_tuned/` |
| 10. Export | `python scripts/17_export_artifacts.py --model lgbm_tuned` | `artifacts/` |

Seeds are fixed (42) and cross-validation folds are grouped by trip and deterministic.

**Model weights:** only the final model (`artifacts/model.joblib`) is committed. The per-model `model.joblib`
files in `outputs/models/` are excluded because of GitHub's file-size limit (the Random Forest file is ~200 MB);
every one can be regenerated with `scripts/13_train_model.py`. All their metrics, scores and plots are committed.

---

## Running the system (two laptops)

Both laptops must be on the same network (a phone hotspot is the most reliable).

### Laptop 1: API

```powershell
cd <repo>
.\.venv\Scripts\Activate.ps1
$env:PATHSNAP_EVAL_DIR="C:\full\path\to\folder\containing\the\trip\folders"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

* For rehearsal, point `PATHSNAP_EVAL_DIR` at `data\raw\student_test`.
* Find Laptop 1's address with `ipconfig` (IPv4 Address). Allow Python through the firewall on private networks.
* Endpoints: `GET /health`, `GET /trips`, `POST /predict` with body `{"trip_id": "<id>"}`. Interactive docs: `http://<ip>:8000/docs`.
* Every request is logged in this terminal (`[request]` / `[result]` lines with accuracy and timing).

### Laptop 2: UI

```powershell
python -m venv .venv-ui
.\.venv-ui\Scripts\Activate.ps1
pip install -r ui\requirements.txt
streamlit run ui\app.py
```

Enter `http://<Laptop-1-IP>:8000` in the sidebar, choose a trip and click **Reconstruct route**.
Laptop 2 holds no model, no data and no ground truth: it only sends the trip ID and displays what the API returns.

---

## Evaluation integrity

* The prediction pipeline (`predict_route` in `api/main.py`) reads only `.track`, `.nodes` and `.arcs`.
  The reference `.route` is read afterwards, server-side, only to compute metrics, and is never sent to the UI.
* No predictions are hardcoded; every result is computed live by the API and logged on Laptop 1.
* Evaluator-testing data was not used for any development decision (features, models, thresholds, hyperparameters).
* Learned preprocessing (feature scaling) is fitted inside each training fold only.

---

## Key design decisions (details in the report)

* **Per-trip UTM projection:** trips lie in different countries; all distances are in metres.
* **Order-preserving label alignment:** the route lists arcs in driving order but not per GPS point; points are aligned to
  route positions that never go backwards (dynamic programming). This resolves the two directions of two-way streets.
  Points > 50 m from their aligned arc are unlabelled and excluded from training and scoring.
* **Candidates:** arcs within 40 m, at most 30 per point (chosen on train from the candidate miss rate).
* **Features:** geometry, direction of travel (vs. arc direction), consistency with neighbouring points, point reliability.
* **Models compared:** Logistic Regression, Gaussian Naive Bayes, KNN, Random Forest, LightGBM, MLP
  (+ rule baselines). LightGBM selected, then tuned with Optuna (TPE).
* **HMM:** emissions = classifier probabilities; transitions = |road-network distance − straight-line distance| / β;
  Viterbi decoding per segment.

Results tables and figures: `outputs/comparison/`, `outputs/tuning/lgbm/`, `outputs/hmm/lgbm_tuned/`.

---

## AI assistance

*(Team: state here how AI tools were used, as required by the event policy.)*
