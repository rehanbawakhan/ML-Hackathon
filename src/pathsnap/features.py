"""Candidate generation and feature engineering for one trip.

One row per (GPS point, candidate arc). The same function is used for training
(with labels) and inside the API (without labels), so preprocessing is identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff, point_segment, project, utm_transformer
from .io import Trip
from .labels import align_route
from .network import build_network

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "head_diff", "head_diff_undir", "speed", "step_len",
    "proj_t", "arc_len", "from_in_deg", "to_out_deg", "has_reverse",
    "nb_dist_mean", "nb_dist_max", "conn_prev", "conn_next",
    "dt_prev", "dt_next", "cand_expanded",
]


def segment_ids(t: np.ndarray, break_s: float) -> np.ndarray:
    """Split the track into continuous pieces wherever the time gap exceeds break_s."""
    return np.r_[0, np.cumsum(np.diff(t) > break_s)]


def gps_heading(P: np.ndarray, seg: np.ndarray, k: int = 2):
    """Heading at each point from points k steps before/after (same segment only)."""
    N = len(P)
    i0 = np.clip(np.arange(N) - k, 0, N - 1)
    i1 = np.clip(np.arange(N) + k, 0, N - 1)
    # stay inside the point's own segment
    i0 = np.where(seg[i0] == seg, i0, np.arange(N))
    i1 = np.where(seg[i1] == seg, i1, np.arange(N))
    d = P[i1] - P[i0]
    return np.arctan2(d[:, 0], d[:, 1]), np.hypot(d[:, 0], d[:, 1])


def candidates(P: np.ndarray, net, radius: float, k: int):
    """Arcs within `radius` metres of each point, nearest `k` kept.
    If none are inside the radius, the nearest arcs are taken and flagged."""
    rows_i, rows_a, expanded = [], [], []
    hits = net.sample_tree.query_ball_point(P, r=radius + net.spacing / 2)
    for i, h in enumerate(hits):
        flag = 0
        arcs = np.unique(net.sample_arc[h]) if h else np.empty(0, int)
        if len(arcs):
            d, _ = point_segment(np.repeat(P[i:i + 1], len(arcs), 0), net.a[arcs], net.b[arcs])
            arcs = arcs[d <= radius]
        if len(arcs) == 0:
            _, nn = net.sample_tree.query(P[i], k=min(64, len(net.sample_arc)))
            arcs = np.unique(net.sample_arc[np.atleast_1d(nn)])
            flag = 1
        d, _ = point_segment(np.repeat(P[i:i + 1], len(arcs), 0), net.a[arcs], net.b[arcs])
        keep = arcs[np.argsort(d)[:k]]
        rows_i.append(np.full(len(keep), i))
        rows_a.append(keep)
        expanded.append(np.full(len(keep), flag))
    return np.concatenate(rows_i), np.concatenate(rows_a), np.concatenate(expanded)


def build_trip_frame(trip: Trip, cfg: dict, with_labels: bool = True):
    """Returns (candidate_df, point_df)."""
    radius = cfg["candidates"]["radius_m"]
    k = cfg["candidates"]["max_k"]
    break_s = cfg["gaps"]["break_seconds"]

    lon0, lat0 = trip.track[:, 0].mean(), trip.track[:, 1].mean()
    tf = utm_transformer(lon0, lat0)
    P = project(tf, trip.track[:, :2])
    t = trip.track[:, 2]
    net = build_network(project(tf, trip.nodes), trip.arcs)
    N = len(P)

    seg = segment_ids(t, break_s)
    head, step = gps_heading(P, seg)
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    span_t = np.maximum(t[np.clip(np.arange(N) + 2, 0, N - 1)] - t[np.clip(np.arange(N) - 2, 0, N - 1)], 1)
    speed = step / span_t

    ci, ca, cexp = candidates(P, net, radius, k)
    dist, proj_t = point_segment(P[ci], net.a[ca], net.b[ca])

    df = pd.DataFrame({"trip_id": trip.trip_id, "point_idx": ci, "arc_id": ca,
                       "segment": seg[ci], "t": t[ci], "dist": dist, "proj_t": proj_t,
                       "cand_expanded": cexp})
    g = df.groupby("point_idx")["dist"]
    df["dist_rank"] = g.rank(method="first") - 1
    dmin = g.transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio"] = df["dist"] / (dmin + 1.0)
    df["n_cand"] = g.transform("size")

    hd = angle_diff(head[ci], net.bearing[ca])
    df["head_diff"] = hd                                  # directional (0 = same way)
    df["head_diff_undir"] = np.minimum(hd, np.pi - hd)    # ignores direction
    df["speed"] = speed[ci]
    df["step_len"] = step[ci]
    df["arc_len"] = net.length[ca]
    df["from_in_deg"] = net.in_deg[net.arcs[ca, 0]]
    df["to_out_deg"] = net.out_deg[net.arcs[ca, 1]]
    df["has_reverse"] = net.has_reverse[ca].astype(int)

    # neighbourhood consistency: how close is this arc to the surrounding points?
    nb = []
    for off in (-2, -1, 1, 2):
        j = np.clip(ci + off, 0, N - 1)
        same = seg[j] == seg[ci]
        d, _ = point_segment(P[j], net.a[ca], net.b[ca])
        nb.append(np.where(same, d, np.nan))
    nb = np.column_stack(nb + [dist])
    df["nb_dist_mean"] = np.nanmean(nb, axis=1)
    df["nb_dist_max"] = np.nanmax(nb, axis=1)

    # connectivity to the nearest candidate of the previous / next point (no labels used)
    nearest = df.loc[df["dist_rank"] == 0].set_index("point_idx")["arc_id"].reindex(range(N)).to_numpy()
    def connected(other):
        ok = ~np.isnan(other)
        o = np.where(ok, other, 0).astype(int)
        u, v = net.arcs[ca, 0], net.arcs[ca, 1]
        ou, ov = net.arcs[o, 0], net.arcs[o, 1]
        c = (o == ca) | (ov == u) | (v == ou) | (ou == u) | (ov == v)
        return (c & ok).astype(int)
    df["conn_prev"] = connected(nearest[np.clip(ci - 1, 0, N - 1)])
    df["conn_next"] = connected(nearest[np.clip(ci + 1, 0, N - 1)])
    df["dt_prev"] = np.nan_to_num(dt_prev[ci], nan=0.0)
    df["dt_next"] = np.nan_to_num(dt_next[ci], nan=0.0)

    df["arc_id"] = net.canon[ca]  # report canonical id
    points = pd.DataFrame({"trip_id": trip.trip_id, "point_idx": np.arange(N), "t": t,
                           "segment": seg, "lon": trip.track[:, 0], "lat": trip.track[:, 1]})

    if with_labels and trip.route is not None and len(trip.route):
        _, true_arc, true_d = align_route(P, net, trip.route)
        points["true_arc"] = true_arc
        points["true_dist"] = true_d
        df["label"] = (df["arc_id"].to_numpy() == true_arc[ci]).astype(int)
        hit = df.groupby("point_idx")["label"].max().reindex(range(N), fill_value=0)
        points["cand_hit"] = hit.to_numpy()
    return df, points, net
