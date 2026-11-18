"""HMM + Viterbi route reconstruction.

Hidden states   : the candidate arcs of each GPS point (state k at time i = "the vehicle
                  is on candidate arc k when point i was recorded").
Emissions       : log of the classifier probability that the candidate is correct,
                  normalised over the point's candidates, times `emission_weight`.
Transitions     : Newson & Krumm (2009): a correct move between consecutive points has a
                  road distance close to the straight-line distance between the points:
                      log P(b | a) = -| d_route(a -> b) - d_straight | / beta
                  d_route = shortest path on the directed road graph from the projection
                  on arc a to the projection on arc b (bounded Dijkstra). Unreachable -> -inf.
Gaps            : a time gap > gaps.break_seconds starts a new segment; Viterbi restarts
                  from the emissions there (no transition is assumed across the gap).
Disconnections  : if no candidate at point i is reachable from any candidate at i-1, the
                  chain is cut and restarted at i (counted in `n_breaks`).
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
import pandas as pd


class RoadGraph:
    """Directed graph over valid (canonical) arcs with cached bounded Dijkstra."""

    def __init__(self, net):
        self.net = net
        self.adj = defaultdict(list)
        for e in np.flatnonzero(net.valid):
            u, v = net.arcs[e]
            self.adj[int(u)].append((int(v), float(net.length[e])))
        self._cache: dict[int, tuple[float, dict]] = {}

    def dist_from(self, src: int, limit: float) -> dict:
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for v, w in self.adj.get(u, ()):
                nd = d + w
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (limit, dist)
        return dist


def route_distance(g: RoadGraph, a: int, ta: float, b: int, tb: float, limit: float) -> float:
    """Road distance from position ta on arc a to position tb on arc b (metres)."""
    net = g.net
    if a == b:
        return abs(tb - ta) * net.length[a]
    d = g.dist_from(int(net.arcs[a, 1]), limit).get(int(net.arcs[b, 0]))
    if d is None:
        return np.inf
    return (1 - ta) * net.length[a] + d + tb * net.length[b]


def viterbi_trip(cands: pd.DataFrame, P: np.ndarray, net, beta: float = 20.0,
                 emission_weight: float = 1.0, score_col: str = "p",
                 max_route_factor: float = 3.0, min_limit: float = 200.0):
    """Decode one trip.

    cands: one row per (point, candidate) with point_idx, arc_id, proj_t, segment,
           dist_rank and `score_col` (classifier probability).
    P:     projected GPS coordinates (metres), indexed by point_idx.
    Returns (DataFrame[point_idx, arc_id, restart], n_breaks).
    """
    g = RoadGraph(net)
    cands = cands.sort_values(["point_idx", "dist_rank"])
    pts, arcs_l, ts_l, segs = [], [], [], []
    for i, grp in cands.groupby("point_idx", sort=True):
        pts.append(int(i))
        arcs_l.append(grp["arc_id"].to_numpy())
        ts_l.append(grp["proj_t"].to_numpy())
        segs.append(int(grp["segment"].iloc[0]))
        p = np.clip(grp[score_col].to_numpy(), 1e-6, 1.0)
        grp_em = emission_weight * np.log(p / p.sum())
        arcs_l[-1] = (arcs_l[-1], grp_em)

    n = len(pts)
    scores, backs, restarts = [], [], []
    n_breaks = 0
    for k in range(n):
        arcs, em = arcs_l[k]
        ts = ts_l[k]
        restart = k == 0 or segs[k] != segs[k - 1]
        if not restart:
            pa, _ = arcs_l[k - 1]
            pt = ts_l[k - 1]
            d_straight = float(np.hypot(*(P[pts[k]] - P[pts[k - 1]])))
            limit = max(min_limit, max_route_factor * d_straight + 50.0)
            trans = np.full((len(pa), len(arcs)), -np.inf)
            for x in range(len(pa)):
                for y in range(len(arcs)):
                    d_route = route_distance(g, pa[x], pt[x], arcs[y], ts[y], limit)
                    if np.isfinite(d_route):
                        trans[x, y] = -abs(d_route - d_straight) / beta
            total = scores[k - 1][:, None] + trans
            back = np.argmax(total, axis=0)
            sc = total[back, np.arange(len(arcs))] + em
            if not np.isfinite(sc).any():
                restart = True
                n_breaks += 1
        if restart:
            back = np.full(len(arcs), -1)
            sc = em.copy()
        scores.append(sc)
        backs.append(back)
        restarts.append(restart)

    # backtrack; a restart at k means step k-1 ended its own chain
    choice = np.zeros(n, dtype=int)
    choice[n - 1] = int(np.argmax(scores[n - 1]))
    for k in range(n - 1, 0, -1):
        if restarts[k]:
            choice[k - 1] = int(np.argmax(scores[k - 1]))
        else:
            choice[k - 1] = int(backs[k][choice[k]])
    pred = [arcs_l[k][0][choice[k]] for k in range(n)]
    return pd.DataFrame({"point_idx": pts, "arc_id": pred, "restart": restarts}), n_breaks
