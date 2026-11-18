"""Train and compare candidate classifiers.

Each row is a (GPS point, candidate arc) pair; label = 1 if the candidate is the
reference arc. Top-1 picks the highest-probability candidate per point.

- Cross-validation: GroupKFold by trip on the 80 training trips.
- Development comparison: the 10 student_test trips.
- Training rows are a fixed random sample of GPS points (all candidates of a
  sampled point are kept) so every model sees exactly the same data.

Usage:
    python scripts/train_models.py                    # all models
    python scripts/train_models.py --models lgbm rf   # subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.metrics import candidate_metrics, continuity, top1  # noqa: E402

SEED = 42
# fraction of the (already sampled) training points used per model, when not all
SUBSAMPLE = {"knn": 0.25}


def make_models() -> dict:
    return {
        "logreg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))),
        "gnb": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB())),
        "knn": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="kd_tree", n_jobs=-1))),
        "rf": ("Ensemble (bagging)", RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "lgbm": ("Boosting", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED, verbose=-1)),
        "mlp": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=30, random_state=SEED))),
    }


def sample_points(df: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    keys = df[["trip_id", "point_idx"]].drop_duplicates().sample(frac=frac, random_state=seed)
    return df.merge(keys, on=["trip_id", "point_idx"])


def route_continuity(pred: pd.DataFrame, raw: Path) -> float:
    vals = []
    for tid, g in pred.sort_values(["trip_id", "point_idx"]).groupby("trip_id"):
        arcs = load_trip(raw / tid, with_route=False).arcs
        vals.append(continuity(g["arc_id"].to_numpy(), arcs))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--frac", type=float, default=0.15, help="fraction of training GPS points used")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed"]
    raw_test = ROOT / cfg["paths"]["raw"] / "student_test"
    logs = ROOT / "experiments" / "logs"
    figs = ROOT / "results" / "figures"
    arts = ROOT / cfg["paths"]["artifacts"] / "models"
    for p in (logs, figs, arts):
        p.mkdir(parents=True, exist_ok=True)

    train = sample_points(pd.read_parquet(proc / "candidates_train.parquet"), args.frac)
    test = pd.read_parquet(proc / "candidates_student_test.parquet")
    print(f"train rows {len(train)} ({train.trip_id.nunique()} trips) | test rows {len(test)}")

    models = make_models()
    names = args.models or list(models)
    rows, fold_rows = [], []
    for name in names:
        family, model = models[name]
        data = train
        if name in SUBSAMPLE:  # instance-based model: query cost grows with training size
            data = sample_points(train, SUBSAMPLE[name])
        print(f"\n== {name} (train rows {len(data)})")
        # group-aware CV on training trips
        gkf = GroupKFold(n_splits=args.folds)
        cv = []
        for f, (tri, vai) in enumerate(gkf.split(data, groups=data["trip_id"])):
            tr, va = data.iloc[tri], data.iloc[vai].copy()
            m = _clone(model)
            m.fit(tr[FEATURES], tr["label"])
            va["p"] = m.predict_proba(va[FEATURES])[:, 1]
            r = candidate_metrics(va)
            r.update(model=name, fold=f)
            fold_rows.append(r)
            cv.append(r)
            print(f"  fold {f}: top1 {r['top1_acc']:.4f} macroF1 {r['macro_f1']:.4f}")
        # refit on all sampled training points, evaluate on student_test
        m = _clone(model)
        t0 = time.time()
        m.fit(data[FEATURES], data["label"])
        fit_s = time.time() - t0
        t0 = time.time()
        test["p"] = m.predict_proba(test[FEATURES])[:, 1]
        inf_s = time.time() - t0
        te = candidate_metrics(test)
        cont = route_continuity(top1(test), raw_test)
        row = {"model": name, "family": family, "config": json.dumps(_params(m)),
               "cv_macro_f1_mean": np.mean([c["macro_f1"] for c in cv]),
               "cv_macro_f1_std": np.std([c["macro_f1"] for c in cv]),
               "cv_top1_mean": np.mean([c["top1_acc"] for c in cv]),
               "test_top1": te["top1_acc"], "test_macro_f1": te["macro_f1"],
               "test_precision": te["precision"], "test_recall": te["recall"],
               "test_log_loss": te["log_loss"], "test_route_continuity": cont,
               "train_time_s": fit_s, "inference_time_s": inf_s,
               "inference_ms_per_point": 1000 * inf_s / test[["trip_id", "point_idx"]].drop_duplicates().shape[0]}
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "config"})
        joblib.dump(m, arts / f"{name}.joblib")
        test[["trip_id", "point_idx", "arc_id", "label", "p"]].to_parquet(logs / f"test_scores_{name}.parquet", index=False)
        if name == "mlp":
            mlp = m[-1]
            plt.figure(figsize=(6, 4))
            plt.plot(mlp.loss_curve_, label="training loss")
            plt.plot(1 - np.array(mlp.validation_scores_), label="1 - validation accuracy")
            plt.xlabel("epoch"); plt.legend(); plt.title("MLP training history")
            plt.tight_layout(); plt.savefig(figs / "mlp_training_curve.png", dpi=150); plt.close()

    res = pd.DataFrame(rows)
    out = logs / "model_comparison.csv"
    if out.exists() and args.models:
        old = pd.read_csv(out)
        res = pd.concat([old[~old.model.isin(res.model)], res])
    res.to_csv(out, index=False)
    pd.DataFrame(fold_rows).to_csv(logs / f"cv_folds_{'_'.join(names)}.csv", index=False)
    plot_comparison(res, figs)
    print("\n", res[["model", "cv_macro_f1_mean", "test_top1", "test_macro_f1", "test_log_loss", "test_route_continuity", "inference_time_s"]].round(4).to_string(index=False))


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def _params(m):
    p = m.get_params()
    return {k: v for k, v in p.items() if isinstance(v, (int, float, str, bool, type(None), tuple))}


def plot_comparison(res: pd.DataFrame, figs: Path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(res.model, res.cv_macro_f1_mean, yerr=res.cv_macro_f1_std, capsize=4)
    ax[0].set_title("CV macro F1 (GroupKFold by trip)")
    x = np.arange(len(res))
    ax[1].bar(x - 0.2, res.test_top1, 0.4, label="Top-1 accuracy")
    ax[1].bar(x + 0.2, res.test_macro_f1, 0.4, label="Macro F1")
    ax[1].set_xticks(x, res.model); ax[1].legend(); ax[1].set_title("Student-test")
    ax[2].bar(res.model, res.test_log_loss); ax[2].set_title("Student-test log loss")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(figs / "model_comparison.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
