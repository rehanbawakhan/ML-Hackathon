"""Build labelled candidate/feature tables for the train and student_test partitions.

Usage (from repo root):
    python scripts/build_dataset.py
    python scripts/build_dataset.py --partitions train --limit 5     # quick test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.features import build_trip_frame  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    raw = ROOT / cfg["paths"]["raw"]
    out = ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)

    for part in args.partitions:
        cands, pts, log = [], [], []
        trips = list_trips(raw / part)[: args.limit]
        for k, d in enumerate(trips, 1):
            t0 = time.time()
            trip = load_trip(d)
            df, points, net = build_trip_frame(trip, cfg, with_labels=True)
            df["partition"] = part
            points["partition"] = part
            cands.append(df)
            pts.append(points)
            miss = 1 - points["cand_hit"].mean()
            log.append({"trip_id": trip.trip_id, "partition": part, "n_points": len(points),
                        "n_candidates": len(df), "n_arcs": len(trip.arcs),
                        "candidate_miss_rate": miss,
                        "median_label_dist_m": points["true_dist"].median(),
                        "seconds": time.time() - t0})
            print(f"[{part} {k}/{len(trips)}] {trip.trip_id} pts={len(points)} "
                  f"cands={len(df)} miss={miss:.3f} {time.time()-t0:.1f}s", flush=True)
        pd.concat(cands).to_parquet(out / f"candidates_{part}.parquet", index=False)
        pd.concat(pts).to_parquet(out / f"points_{part}.parquet", index=False)
        pd.DataFrame(log).to_csv(ROOT / "results" / "tables" / f"build_log_{part}.csv", index=False)
        print(f"saved {part}")


if __name__ == "__main__":
    main()
