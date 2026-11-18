"""End-to-end prediction for one trip folder: features -> classifier -> HMM/Viterbi.

Used by both scripts/evaluate_hmm.py and the API, so the demo runs exactly the
pipeline that was evaluated. Ground truth is never read here.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import FEATURES, build_trip_frame
from .geo import project, utm_transformer
from .hmm import viterbi_trip
from .io import Trip


def predict_trip(trip: Trip, model, cfg: dict, use_hmm: bool = True) -> dict:
    t0 = time.time()
    trip_nolabel = Trip(trip.trip_id, trip.track, trip.nodes, trip.arcs, route=None)
    cands, points, net = build_trip_frame(trip_nolabel, cfg, with_labels=False)
    cands["p"] = model.predict_proba(cands[FEATURES])[:, 1]
    t_feat = time.time() - t0

    top = cands.loc[cands.groupby("point_idx")["p"].idxmax(), ["point_idx", "arc_id"]]
    clf_pred = top.set_index("point_idx")["arc_id"].reindex(points["point_idx"]).to_numpy()

    hmm_pred, n_breaks = clf_pred, 0
    if use_hmm:
        tf = utm_transformer(trip.track[:, 0].mean(), trip.track[:, 1].mean())
        P = project(tf, trip.track[:, :2])
        h = cfg.get("hmm", {})
        dec, n_breaks = viterbi_trip(cands, P, net, beta=h.get("beta_m", 20.0),
                                     emission_weight=h.get("emission_weight", 1.0))
        hmm_pred = dec.set_index("point_idx")["arc_id"].reindex(points["point_idx"]).to_numpy()
    return {"cands": cands, "P": None if not use_hmm else P, "points": points, "net": net, "clf_pred": clf_pred, "hmm_pred": hmm_pred,
            "n_breaks": n_breaks, "seconds": time.time() - t0, "feature_seconds": t_feat}
