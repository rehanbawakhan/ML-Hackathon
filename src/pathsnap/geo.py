"""Geometry helpers.

Trips are spread across the world, so each trip is projected to its own UTM
zone (metres). Euclidean distance in degrees is never used.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


def utm_transformer(lon: float, lat: float) -> Transformer:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)


def project(tf: Transformer, lonlat: np.ndarray) -> np.ndarray:
    if len(lonlat) == 0:
        return np.empty((0, 2))
    x, y = tf.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points p to segments a->b (all (n,2), broadcast row-wise).

    Returns (distance, t) where t in [0,1] is the projection position on the segment.
    """
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.linalg.norm(p - proj, axis=1), t


def bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass-style angle in radians of a displacement in projected metres."""
    return np.arctan2(dx, dy)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute smallest difference between two angles, in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)
