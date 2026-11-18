"""Derive a per-GPS-point reference arc from the ordered .route file.

The .route file is only the ordered list of arcs driven; it does not say which
GPS point belongs to which arc. We assign each point i a route position j_i so
that j_i never decreases (the vehicle moves forward along the route) and the
total point-to-arc distance is minimal. This is a monotone alignment solved by
dynamic programming in O(N * R):

    C[i, j] = d(i, j) + min_{j' <= j} C[i-1, j']

Monotonicity stops points from being assigned to a later/earlier pass over the
same road when the route loops or overlaps itself.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment
from .network import Network


def align_route(P: np.ndarray, net: Network, route: np.ndarray, chunk: int = 256):
    N, R = len(P), len(route)
    A, B = net.a[route], net.b[route]
    back_dtype = np.int16 if R < 32767 else np.int32
    back = np.empty((N, R), dtype=back_dtype)
    prev = None
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        n = e - s
        pp = np.repeat(P[s:e], R, axis=0)
        dist, _ = point_segment(pp, np.tile(A, (n, 1)), np.tile(B, (n, 1)))
        D = dist.reshape(n, R)
        for k in range(n):
            i = s + k
            if prev is None:
                cur = D[k].copy()
                back[i] = np.arange(R)
            else:
                # prefix minimum of previous row and where it came from
                idx = np.arange(R)
                is_new_min = np.r_[True, prev[1:] < np.minimum.accumulate(prev)[:-1]]
                arg = np.maximum.accumulate(np.where(is_new_min, idx, 0))
                cur = D[k] + prev[arg]
                back[i] = arg
            prev = cur
    # backtrack
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    arc = route[j]
    d_final, _ = point_segment(P, net.a[arc], net.b[arc])
    return j, net.canon[arc], d_final
