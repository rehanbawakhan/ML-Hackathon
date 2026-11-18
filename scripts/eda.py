"""Exploratory data analysis on the train and student_test partitions.

Outputs:
  results/tables/eda_inventory.csv      per-trip counts, gaps, extent, connectivity
  results/figures/eda_*.png             distributions
  results/maps/trip_<id>.html           Folium map: GPS, nearby road network, reference route

Usage:
    python scripts/eda.py
    python scripts/eda.py --map-trips 00000010 00000080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.geo import project, utm_transformer  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def trip_stats(trip, part):
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    dt = np.diff(tr[:, 2])
    step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    V = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(V, V))
    n_weak, lab = connected_components(g, directed=True, connection="weak")
    deg = np.bincount(arcs.ravel(), minlength=V)
    key = set(map(tuple, arcs))
    cont = (arcs[route[:-1], 1] == arcs[route[1:], 0]).mean() if route is not None and len(route) > 1 else np.nan
    return {
        "partition": part, "trip_id": trip.trip_id, "n_points": len(tr),
        "n_nodes": V, "n_arcs": len(arcs), "n_route_arcs": 0 if route is None else len(route),
        "nan_values": int(np.isnan(tr).sum()), "non_increasing_t": int((dt <= 0).sum()),
        "duration_s": tr[-1, 2] - tr[0, 2], "share_dt_1s": float((dt == 1).mean()),
        "gaps_over_10s": int((dt > 10).sum()), "max_gap_s": float(dt.max()),
        "path_km": step.sum() / 1000, "max_step_m": float(step.max()),
        "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(), "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
        "duplicate_arcs": len(arcs) - len(key), "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "isolated_nodes": int((deg == 0).sum()), "weak_components": int(n_weak),
        "share_two_way_arcs": float(np.mean([(b, a) in key for a, b in arcs])),
        "route_continuity": cont,
    }


def folium_map(trip, path: Path, pad_m: float = 300):
    tr = trip.track
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    # road network near the track only (full networks can have 300k arcs)
    tf = utm_transformer(tr[:, 0].mean(), tr[:, 1].mean())
    P, N = project(tf, tr[:, :2]), project(tf, trip.nodes)
    lo, hi = P.min(0) - pad_m, P.max(0) + pad_m
    inside = np.all((N >= lo) & (N <= hi), axis=1)
    sel = trip.arcs[inside[trip.arcs[:, 0]] & inside[trip.arcs[:, 1]]]
    net = folium.FeatureGroup(name="Road network", show=True)
    for a, b in sel[:20000]:
        folium.PolyLine([(trip.nodes[a, 1], trip.nodes[a, 0]), (trip.nodes[b, 1], trip.nodes[b, 0])],
                        color="#9e9e9e", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    if trip.route is not None and len(trip.route):
        rl = folium.FeatureGroup(name="Reference route", show=True)
        for arc in trip.route:
            a, b = trip.arcs[arc]
            folium.PolyLine([(trip.nodes[a, 1], trip.nodes[a, 0]), (trip.nodes[b, 1], trip.nodes[b, 0])],
                            color="#2e7d32", weight=5, opacity=0.8).add_to(rl)
        rl.add_to(m)
    gps = folium.FeatureGroup(name="GPS observations", show=True)
    for lon, lat, t in tr[:: max(1, len(tr) // 1500)]:
        folium.CircleMarker((lat, lon), radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-trips", nargs="+", default=["00000010", "00000080", "00000033"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    raw = ROOT / cfg["paths"]["raw"]
    tabs, figs, maps = (ROOT / "results" / x for x in ("tables", "figures", "maps"))
    for p in (tabs, figs, maps):
        p.mkdir(parents=True, exist_ok=True)

    rows = []
    for part in ("train", "student_test"):
        for d in list_trips(raw / part):
            trip = load_trip(d)
            rows.append(trip_stats(trip, part))
            if trip.trip_id in args.map_trips:
                folium_map(trip, maps / f"trip_{trip.trip_id}.html")
                print("map", trip.trip_id)
    inv = pd.DataFrame(rows)
    inv.to_csv(tabs / "eda_inventory.csv", index=False)
    summary = inv.groupby("partition")[["n_points", "n_arcs", "max_gap_s", "path_km", "duplicate_arcs",
                                         "weak_components", "share_two_way_arcs"]].describe().T
    summary.to_csv(tabs / "eda_summary.csv")
    print(inv.groupby("partition").agg(trips=("trip_id", "count"), points=("n_points", "sum"),
                                       nan=("nan_values", "sum"), gaps10=("gaps_over_10s", "sum")))
    anomalies = inv[(inv.max_gap_s > 300) | (inv.route_continuity < 1) | (inv.duplicate_arcs > 50)]
    anomalies.to_csv(tabs / "eda_anomalies.csv", index=False)
    print(f"{len(anomalies)} trips flagged -> eda_anomalies.csv")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for part, g in inv.groupby("partition"):
        ax[0, 0].hist(g.n_points, bins=30, alpha=0.6, label=part)
        ax[0, 1].hist(np.log10(g.n_arcs), bins=30, alpha=0.6, label=part)
        ax[1, 0].hist(np.log10(g.max_gap_s), bins=30, alpha=0.6, label=part)
        ax[1, 1].scatter(inv.loc[g.index, "lon_min"], inv.loc[g.index, "lat_min"], s=12, label=part)
    ax[0, 0].set_title("GPS points per trip"); ax[0, 1].set_title("log10(road arcs per trip)")
    ax[1, 0].set_title("log10(max time gap, s)"); ax[1, 1].set_title("Trip locations (lon, lat)")
    for a in ax.ravel():
        a.legend()
    plt.tight_layout(); plt.savefig(figs / "eda_overview.png", dpi=150); plt.close()

    # label-derivation quality and candidate coverage (needs build_dataset.py output)
    pts_path = ROOT / cfg["paths"]["processed"] / "points_train.parquet"
    if pts_path.exists():
        pts = pd.read_parquet(pts_path)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(np.clip(pts.true_dist, 0, 200), bins=80)
        ax[0].set_title("Distance GPS point -> derived reference arc (m, train)")
        miss = pts.groupby("trip_id").cand_hit.mean().rsub(1).sort_values(ascending=False)
        ax[1].bar(miss.index[:15], miss.values[:15]); ax[1].tick_params(axis="x", rotation=60)
        ax[1].set_title("Candidate miss rate, worst 15 train trips")
        plt.tight_layout(); plt.savefig(figs / "eda_labels_candidates.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
