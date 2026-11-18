"""Classifier-only vs classifier + HMM/Viterbi on the student_test trips.

1. Scores candidates with the saved classifier (default: final model, else lgbm).
2. Grid-searches the decoder parameters (beta, emission_weight) on student_test, which
   the guidelines allow for decoder parameters (never on evaluator trips).
3. Reports point accuracy, route continuity, route-level precision/recall/F1 and breaks.

Usage:
    python scripts/evaluate_hmm.py
    python scripts/evaluate_hmm.py --model models/artifacts/models/lgbm.joblib
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.geo import project, utm_transformer  # noqa: E402
from pathsnap.hmm import viterbi_trip  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.labels import reference_point_arcs  # noqa: E402
from pathsnap.metrics import continuity  # noqa: E402
from pathsnap.pipeline import predict_trip  # noqa: E402


def route_prf(pred, ref_route, canon):
    """Route-level overlap between the set of predicted arcs and the reference route arcs."""
    ps, rs = set(map(int, pred)), set(map(int, canon[ref_route]))
    tp = len(ps & rs)
    prec, rec = tp / max(len(ps), 1), tp / max(len(rs), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def score_trip(name, pred, ref, trip, net):
    prec, rec, f1 = route_prf(pred, trip.route, net.canon)
    return {"method": name, "point_acc": float((pred == ref).mean()),
            "continuity": continuity(pred, trip.arcs),
            "route_precision": prec, "route_recall": rec, "route_f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    arts = ROOT / cfg["paths"]["artifacts"]
    model_path = Path(args.model) if args.model else (
        arts / "final_model.joblib" if (arts / "final_model.joblib").exists() else arts / "models" / "lgbm.joblib")
    model = joblib.load(model_path)
    print("model:", model_path)

    trips = []
    for d in list_trips(ROOT / cfg["paths"]["raw"] / "student_test"):
        trip = load_trip(d)
        r = predict_trip(trip, model, cfg, use_hmm=False)
        ref, _, net = reference_point_arcs(trip)
        tf = utm_transformer(trip.track[:, 0].mean(), trip.track[:, 1].mean())
        trips.append((trip, r, ref, net, project(tf, trip.track[:, :2])))

    # decoder parameter grid
    grid = []
    for beta, w in itertools.product([1, 2, 5, 10, 20, 40], [0.1, 0.25, 0.5, 1.0]):
        accs = []
        for trip, r, ref, net, P in trips:
            dec, _ = viterbi_trip(r["cands"], P, net, beta=beta, emission_weight=w)
            pred = dec.set_index("point_idx")["arc_id"].reindex(r["points"]["point_idx"]).to_numpy()
            accs.append((pred == ref).mean())
        grid.append({"beta_m": beta, "emission_weight": w, "mean_point_acc": float(np.mean(accs))})
        print(grid[-1])
    grid = pd.DataFrame(grid).sort_values("mean_point_acc", ascending=False)
    best = grid.iloc[0]
    print("best decoder params:", best.to_dict())

    rows = []
    for trip, r, ref, net, P in trips:
        dec, breaks = viterbi_trip(r["cands"], P, net, beta=best.beta_m, emission_weight=best.emission_weight)
        hmm = dec.set_index("point_idx")["arc_id"].reindex(r["points"]["point_idx"]).to_numpy()
        for name, pred in (("classifier_only", r["clf_pred"]), ("classifier_hmm", hmm)):
            row = score_trip(name, pred, ref, trip, net)
            row.update(trip_id=trip.trip_id, n_points=len(ref), hmm_breaks=breaks if name == "classifier_hmm" else np.nan)
            rows.append(row)
    res = pd.DataFrame(rows)

    logs = ROOT / "experiments" / "logs"
    figs = ROOT / "results" / "figures"
    grid.to_csv(logs / "hmm_param_grid.csv", index=False)
    res.to_csv(logs / "hmm_vs_classifier_per_trip.csv", index=False)
    summary = res.groupby("method")[["point_acc", "continuity", "route_precision", "route_recall", "route_f1"]].mean()
    summary.to_csv(logs / "hmm_vs_classifier_summary.csv")
    print(summary.round(4))
    with open(arts / "hmm_config.yaml", "w") as f:
        yaml.safe_dump({"beta_m": float(best.beta_m), "emission_weight": float(best.emission_weight)}, f)

    piv = res.pivot(index="trip_id", columns="method", values="point_acc")
    ax = piv[["classifier_only", "classifier_hmm"]].plot.bar(figsize=(10, 4))
    ax.set_ylabel("point accuracy"); ax.set_title("Classifier-only vs classifier + HMM (student_test)")
    plt.tight_layout(); plt.savefig(figs / "hmm_vs_classifier.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
