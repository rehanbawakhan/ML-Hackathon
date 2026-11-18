# ML Hackathon — PathSnap Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University (18 Nov 2026).

Noisy GPS points are matched to road segments: candidate arcs are generated per GPS point, a classifier scores each candidate, and HMM/Viterbi decoding reconstructs a connected route. Served via a FastAPI backend (Laptop 1) and an evaluator UI (Laptop 2).

## Team
| Member | SRN | Ownership |
|---|---|---|
| Rehan S. Bawakhan | PES2UG25CS820 | |
| Narendra Kathare| PES2UG25CS814 | |
| Eshwar B H | PES2UG25CS808 | |

## Repository layout
```
data/raw/            raw trips (not committed) -> train/, student_test/, eval/
data/processed/      derived labels, candidates, features
src/pathsnap/        library code: io, geo, labels, candidates, features, hmm, metrics
notebooks/           EDA and experiment notebooks
scripts/             runnable pipeline steps (build features, train, tune, evaluate)
models/artifacts/    exported model + preprocessing + HMM config
experiments/logs/    per-model results CSVs, seeds, configs
experiments/optuna/  Bayesian optimisation study + trial log
results/figures/     plots used in the report
results/maps/        Folium HTML maps
results/tables/      comparison tables
api/                 FastAPI backend (Laptop 1)
ui/                  evaluator interface (Laptop 2)
report/              technical report (PDF)
tests/               sanity tests
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset so the folders look like:
```
data/raw/train/00000001/00000001.{track,nodes,arcs,route}
data/raw/student_test/...
data/raw/eval/...            # evaluator trips, Laptop 1 only
```

## Running
From the repo root, in order:
```bash
python scripts/build_dataset.py     # labels + candidates + features -> data/processed/ (~1 min)
python scripts/eda.py               # EDA tables, plots, Folium maps -> results/
python scripts/train_models.py      # 6 models, GroupKFold CV + student_test -> experiments/logs/ (~5 min)
```
```bash
python scripts/tune_optuna.py       # Bayesian HPO of LightGBM -> experiments/optuna/, models/artifacts/final_model.joblib (~10 min)
python scripts/evaluate_hmm.py      # classifier-only vs classifier + HMM/Viterbi -> experiments/logs/, models/artifacts/hmm_config.yaml
```
`train_models.py --models lgbm rf` re-runs only some models; results merge into `experiments/logs/model_comparison.csv`.

### Pipeline summary
1. **Projection:** each trip is projected to its own UTM zone (pyproj), so all distances are in metres.
2. **Network cleaning:** duplicate (from,to) arcs are mapped to one canonical id; self-loops and zero-length arcs are dropped from candidates.
3. **Gap handling:** the track is split into segments where the time gap exceeds `gaps.break_seconds`.
4. **Per-point labels:** `.route` is only the ordered arc list, so each GPS point is assigned a route position by a monotone dynamic-programming alignment (positions never move backwards) minimising total point-to-arc distance. See `src/pathsnap/labels.py`.
5. **Candidates:** arcs within `candidates.radius_m` of each point, nearest `max_k` kept; if none, nearest arcs are taken and flagged (`cand_expanded`).
6. **Features:** distance and distance rank, GPS-heading vs arc-direction difference, speed, projection position, arc length, node degrees, two-way flag, distance of the same arc to neighbouring points, connectivity to neighbouring points' nearest arcs, time gaps.
7. **Models:** binary "is this candidate the reference arc?" classifiers; Top-1 = highest probability candidate per point.

## Two-laptop deployment
**Laptop 1 — API** (has the repo, trained artifacts and `data/raw/eval/`):
```bash
ipconfig                                         # note the IPv4 address, e.g. 192.168.1.20
netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000   # admin terminal, once
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Every request and its result is logged in this terminal and in `api/logs/requests.log`.

**Laptop 2 — UI** (repo cloned, `pip install -r requirements.txt`, no data needed):
```bash
streamlit run ui/app.py
```
Enter `http://<laptop-1-IP>:8000`, click **Connect**, pick a trip, click **Run map-matching**.

Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "00000015"}`.
Ground truth is read only inside the API's scoring step after prediction and is never sent to the UI.

## Data partitions & leakage rules
- Train (80 trips): fitting, cross-validation (GroupKFold by trip), Bayesian optimisation.
- Student test (10 trips): development-stage comparison only.
- Evaluator test (10 trips): final evaluation only; ground truth is read only inside the API's scoring step and never reaches the model or the UI.

## AI assistance disclosure
_Describe AI tool use here as required by the event policy._
