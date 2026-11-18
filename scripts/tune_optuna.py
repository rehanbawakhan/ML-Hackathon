"""Bayesian hyperparameter optimisation of LightGBM with Optuna (TPE sampler).

- Search runs only on the 80 training trips with GroupKFold by trip.
- Objective: minimise mean CV log loss. Log loss measures probability quality, which is
  what the HMM uses as emission scores; CV macro F1 and Top-1 are recorded per trial too.
- After the search, the untuned (baseline) and tuned models are refitted on the same
  training sample and compared on the same student_test trips.
- The tuned model is exported as the final model used by the HMM and the API.

Usage:
    python scripts/tune_optuna.py              # 25 trials
    python scripts/tune_optuna.py --trials 50
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
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics  # noqa: E402

SEED = 42
BASELINE = dict(n_estimators=300, learning_rate=0.05, num_leaves=63)


def sample_points(df, frac, seed=SEED):
    keys = df[["trip_id", "point_idx"]].drop_duplicates().sample(frac=frac, random_state=seed)
    return df.merge(keys, on=["trip_id", "point_idx"])


def cv_eval(params, data, folds):
    res = []
    for tri, vai in GroupKFold(n_splits=folds).split(data, groups=data["trip_id"]):
        tr, va = data.iloc[tri], data.iloc[vai].copy()
        m = LGBMClassifier(**params, random_state=SEED, verbose=-1).fit(tr[FEATURES], tr["label"])
        va["p"] = m.predict_proba(va[FEATURES])[:, 1]
        res.append(candidate_metrics(va))
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


def fit_and_test(params, train, test):
    t0 = time.time()
    m = LGBMClassifier(**params, random_state=SEED, verbose=-1).fit(train[FEATURES], train["label"])
    fit_s = time.time() - t0
    t0 = time.time()
    test = test.copy()
    test["p"] = m.predict_proba(test[FEATURES])[:, 1]
    inf_s = time.time() - t0
    return m, candidate_metrics(test), fit_s, inf_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed"]
    out = ROOT / "experiments" / "optuna"
    figs = ROOT / "results" / "figures"
    arts = ROOT / cfg["paths"]["artifacts"]
    for p in (out, figs, arts):
        p.mkdir(parents=True, exist_ok=True)

    train = sample_points(pd.read_parquet(proc / "candidates_train.parquet"), args.frac)
    test = pd.read_parquet(proc / "candidates_student_test.parquet")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        r = cv_eval(params, train, args.folds)
        for k, v in r.items():
            trial.set_user_attr(f"cv_{k}", v)
        print(f"trial {trial.number}: cv_log_loss {r['log_loss']:.4f} cv_macro_f1 {r['macro_f1']:.4f}", flush=True)
        return r["log_loss"]

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="lgbm_pathsnap")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials)
    search_s = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(out / "optuna_trials.csv", index=False)
    best = {**study.best_params, "subsample_freq": 1}
    json.dump({"best_params": best, "best_cv_log_loss": study.best_value, "n_trials": args.trials,
               "search_seconds": search_s, "sampler": "TPE", "seed": SEED,
               "objective": "minimise GroupKFold CV log loss"}, open(out / "best_params.json", "w"), indent=2)

    # untuned vs tuned on identical data
    rows = []
    for name, params in (("lgbm_untuned", BASELINE), ("lgbm_tuned", best)):
        cv = cv_eval(params, train, args.folds)
        m, te, fit_s, inf_s = fit_and_test(params, train, test)
        rows.append({"model": name, "params": json.dumps(params),
                     "cv_macro_f1": cv["macro_f1"], "cv_log_loss": cv["log_loss"], "cv_top1": cv["top1_acc"],
                     "test_top1": te["top1_acc"], "test_macro_f1": te["macro_f1"], "test_log_loss": te["log_loss"],
                     "train_time_s": fit_s, "inference_time_s": inf_s})
        if name == "lgbm_tuned":
            joblib.dump(m, arts / "final_model.joblib")
            json.dump({"model": "LightGBM (Optuna-tuned)", "features": FEATURES, "params": params},
                      open(arts / "final_model_meta.json", "w"), indent=2)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "tuned_vs_untuned.csv", index=False)
    print(comp.drop(columns="params").round(4).to_string(index=False))
    print(f"search time {search_s:.0f}s for {args.trials} trials")

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    vals = trials["value"].to_numpy()
    ax[0].plot(vals, "o", alpha=0.5, label="trial")
    ax[0].plot(np.minimum.accumulate(vals), "-", label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV log loss"); ax[0].legend(); ax[0].set_title("Optuna optimisation history")
    metrics = ["cv_macro_f1", "test_top1", "test_macro_f1", "test_log_loss"]
    x = np.arange(len(metrics))
    for k, r in comp.iterrows():
        ax[1].bar(x + (k - 0.5) * 0.4, [r[m] for m in metrics], 0.4, label=r["model"])
    ax[1].set_xticks(x, metrics, rotation=20); ax[1].legend(); ax[1].set_title("Untuned vs tuned LightGBM")
    plt.tight_layout(); plt.savefig(figs / "optuna_results.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
