"""Evaluation metrics for candidate scoring and route reconstruction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def top1(df: pd.DataFrame, score_col: str = "p") -> pd.DataFrame:
    """Highest-scoring candidate per GPS point."""
    idx = df.groupby(["trip_id", "point_idx"])[score_col].idxmax()
    return df.loc[idx]


def continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive predictions that are the same arc or physically connected
    (end node of one = start node of the next)."""
    if len(pred_arcs) < 2:
        return 1.0
    a, b = pred_arcs[:-1], pred_arcs[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def candidate_metrics(df: pd.DataFrame, score_col: str = "p", thr: float = 0.5) -> dict:
    y, p = df["label"].to_numpy(), df[score_col].to_numpy()
    yhat = (p >= thr).astype(int)
    t1 = top1(df, score_col)
    return {
        "top1_acc": float(t1["label"].mean()),
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }
