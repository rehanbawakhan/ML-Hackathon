"""Raw trip loading.

Each trip folder <id>/ holds:
  <id>.track  lon lat t          (GPS observations)
  <id>.nodes  lon lat            (row index = node_id)
  <id>.arcs   from_node to_node  (row index = arc_id)
  <id>.route  arc_id             (reference route, may be absent at inference)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (V, 2) lon, lat
    arcs: np.ndarray             # (E, 2) from, to
    route: np.ndarray | None = None   # (R,) arc ids
    meta: dict = field(default_factory=dict)


def _load(path: Path, dtype=float, ncols: int | None = None) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols) if ncols else (0,), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2 if ncols else 1)
    return arr


def load_trip(trip_dir: str | os.PathLike, with_route: bool = True) -> Trip:
    d = Path(trip_dir)
    tid = d.name
    track = _load(d / f"{tid}.track", float, 3)
    nodes = _load(d / f"{tid}.nodes", float, 2)
    arcs = _load(d / f"{tid}.arcs", np.int64, 2)
    route = None
    if with_route and (d / f"{tid}.route").exists():
        route = _load(d / f"{tid}.route", np.int64).ravel()
    return Trip(tid, track, nodes, arcs, route)


def list_trips(partition_dir: str | os.PathLike) -> list[Path]:
    p = Path(partition_dir)
    return sorted(x for x in p.iterdir() if x.is_dir())
