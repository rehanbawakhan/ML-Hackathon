# ML Hackathon — PathSnap Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University (18 Nov 2026).

Noisy GPS points are matched to road segments: candidate arcs are generated per GPS point, a classifier scores each candidate, and HMM/Viterbi decoding reconstructs a connected route. Served via a FastAPI backend (Laptop 1) and an evaluator UI (Laptop 2).

## Team
| Member | SRN | 
|---|---|
| Rehan S. Bawakhan | PES2UG25CS820 |
| Narendra Kathare | PES2UG25CS814 |
| Eshwar B H | PES2UG25CS808 |

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
_To be filled in as pipeline steps are added._

## Data partitions & leakage rules
- Train (80 trips): fitting, cross-validation (GroupKFold by trip), Bayesian optimisation.
- Student test (10 trips): development-stage comparison only.
- Evaluator test (10 trips): final evaluation only; ground truth is read only inside the API's scoring step and never reaches the model or the UI.

## AI assistance disclosure
_Describe AI tool use here as required by the event policy._
