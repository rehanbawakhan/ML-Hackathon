"""Road-network preprocessing for one trip."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Network:
    xy: np.ndarray          # (V,2) projected node coords, metres
    arcs: np.ndarray        # (E,2) from,to (original ids kept)
    canon: np.ndarray       # (E,) canonical arc id for duplicate (from,to) pairs
    valid: np.ndarray       # (E,) bool: canonical, non-self-loop, non-zero length
    a: np.ndarray           # (E,2) segment start
    b: np.ndarray           # (E,2) segment end
    length: np.ndarray      # (E,)
    bearing: np.ndarray     # (E,) radians
    in_deg: np.ndarray      # (V,)
    out_deg: np.ndarray     # (V,)
    has_reverse: np.ndarray # (E,) bool
    sample_tree: cKDTree    # KD-tree over points sampled along valid arcs
    sample_arc: np.ndarray  # arc id per sample
    spacing: float


def build_network(nodes_xy: np.ndarray, arcs: np.ndarray, spacing: float = 15.0) -> Network:
    E = len(arcs)
    # canonical id: first arc with the same (from,to) pair
    key = arcs[:, 0] * (len(nodes_xy) + 1) + arcs[:, 1]
    _, first_idx, inv = np.unique(key, return_index=True, return_inverse=True)
    canon = first_idx[inv]

    a = nodes_xy[arcs[:, 0]]
    b = nodes_xy[arcs[:, 1]]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    bearing = np.arctan2(d[:, 0], d[:, 1])
    valid = (canon == np.arange(E)) & (arcs[:, 0] != arcs[:, 1]) & (length > 0.01)

    V = len(nodes_xy)
    out_deg = np.bincount(arcs[valid, 0], minlength=V)
    in_deg = np.bincount(arcs[valid, 1], minlength=V)

    rev_key = arcs[:, 1] * (V + 1) + arcs[:, 0]
    has_reverse = np.isin(rev_key, key)

    # sample points along each valid arc for candidate search
    ids = np.flatnonzero(valid)
    n_s = np.maximum(1, np.ceil(length[ids] / spacing).astype(int)) + 1
    arc_rep = np.repeat(ids, n_s)
    # fraction positions 0..1 within each arc
    offs = np.concatenate([np.linspace(0, 1, k) for k in n_s]) if len(ids) else np.empty(0)
    pts = a[arc_rep] + d[arc_rep] * offs[:, None]
    tree = cKDTree(pts) if len(pts) else None

    return Network(nodes_xy, arcs, canon, valid, a, b, length, bearing,
                   in_deg, out_deg, has_reverse, tree, arc_rep, spacing)
