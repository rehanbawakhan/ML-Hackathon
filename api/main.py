"""PathSnap FastAPI backend (Laptop 1).

Holds the trained model, HMM configuration, road networks and the evaluator trips.
Ground truth (.route) is read ONLY inside `score_against_reference`, after the
prediction is complete, and is never returned to the client.

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables (optional):
    PATHSNAP_TRIPS_DIR   folder with the evaluation trips (default data/raw/eval)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.geo import project, utm_transformer  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.labels import reference_point_arcs  # noqa: E402
from pathsnap.metrics import continuity  # noqa: E402
from pathsnap.pipeline import predict_trip  # noqa: E402

# ---------- logging: console + file, so the evaluator can see every request ----------
LOG_DIR = ROOT / "api" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "requests.log")])
log = logging.getLogger("pathsnap")

# ---------- load artifacts once at startup ----------
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
ARTS = ROOT / CFG["paths"]["artifacts"]
hmm_file = ARTS / "hmm_config.yaml"
if hmm_file.exists():
    CFG["hmm"].update(yaml.safe_load(open(hmm_file)))
MODEL_PATH = ARTS / "final_model.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = ARTS / "models" / "lgbm.joblib"
MODEL = joblib.load(MODEL_PATH)
TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / CFG["paths"]["raw"] / "eval"))
log.info("model=%s | hmm=%s | trips_dir=%s", MODEL_PATH.name, CFG["hmm"], TRIPS_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("%s %s from %s -> %s (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, time.time() - t0)
    return response


def available_trips() -> list[str]:
    if not TRIPS_DIR.exists():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir() if p.is_dir())


def score_against_reference(trip, clf_pred, hmm_pred) -> dict:
    """Server-side scoring. The reference route is used here only, after prediction."""
    if trip.route is None or len(trip.route) == 0:
        return {}
    ref, _, net = reference_point_arcs(trip)
    rs = set(map(int, net.canon[trip.route]))
    ps = set(map(int, hmm_pred))
    tp = len(ps & rs)
    prec, rec = tp / max(len(ps), 1), tp / max(len(rs), 1)
    return {
        "point_accuracy_hmm": float((hmm_pred == ref).mean()),
        "point_accuracy_classifier_only": float((clf_pred == ref).mean()),
        "route_precision": prec, "route_recall": rec,
        "route_f1": 2 * prec * rec / max(prec + rec, 1e-9),
    }


def arc_polylines(trip, arc_ids):
    """Collapse consecutive repeats and return each arc as [[lat, lon], [lat, lon]]."""
    seq = [int(a) for k, a in enumerate(arc_ids) if k == 0 or a != arc_ids[k - 1]]
    n = trip.nodes
    return [[[n[trip.arcs[a, 0], 1], n[trip.arcs[a, 0], 0]], [n[trip.arcs[a, 1], 1], n[trip.arcs[a, 1], 0]]] for a in seq]


def nearby_network(trip, pad_m=150.0, cap=15000):
    tf = utm_transformer(trip.track[:, 0].mean(), trip.track[:, 1].mean())
    P, N = project(tf, trip.track[:, :2]), project(tf, trip.nodes)
    lo, hi = P.min(0) - pad_m, P.max(0) + pad_m
    inside = np.all((N >= lo) & (N <= hi), axis=1)
    sel = np.flatnonzero(inside[trip.arcs[:, 0]] & inside[trip.arcs[:, 1]])[:cap]
    n = trip.nodes
    return [[[n[a, 1], n[a, 0]], [n[b, 1], n[b, 0]]] for a, b in trip.arcs[sel]]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    return {"trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in available_trips():
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    trip = load_trip(TRIPS_DIR / req.trip_id)  # route stays inside this function
    log.info("predict trip=%s points=%d arcs=%d", req.trip_id, len(trip.track), len(trip.arcs))

    r = predict_trip(trip, MODEL, CFG, use_hmm=True)  # prediction never sees the route
    clf_pred, hmm_pred = r["clf_pred"], r["hmm_pred"]
    metrics = score_against_reference(trip, clf_pred, hmm_pred)
    metrics.update({
        "route_continuity_hmm": continuity(hmm_pred, trip.arcs),
        "route_continuity_classifier_only": continuity(clf_pred, trip.arcs),
        "hmm_breaks": int(r["n_breaks"]),
        "inference_seconds": round(r["seconds"], 3),
    })
    log.info("result trip=%s %s", req.trip_id, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    step = max(1, len(trip.track) // 2000)
    return {
        "trip_id": req.trip_id,
        "n_points": int(len(trip.track)),
        "metrics": metrics,
        "gps": trip.track[::step, [1, 0]].tolist(),
        "predicted_route": arc_polylines(trip, hmm_pred),
        "predicted_arcs": [int(a) for a in hmm_pred],
        "network": nearby_network(trip),
    }
