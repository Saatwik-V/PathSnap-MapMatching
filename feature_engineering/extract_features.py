"""
Feature Engineering - PathSnap
Expands the 6 base geometric features from data_processing.candidate_gen into the
12-feature representation used for model training and inference (kinematic +
geometric + topological-rank features), and builds the train/test feature tables.

Base (from candidate_gen, per GPS-point x candidate-arc pair):
    perp_dist, proj_ratio, heading_error, arc_length, dist_to_start, dist_to_end,
    candidate_rank

Added here (kinematic / derived):
    heading_cos          cos(radians(heading_error))
    speed_mps            displacement between consecutive GPS points / max(delta_t, 0.1s)
    delta_t              timestamp difference (s) since the previous observation
    dist_diff_endpoints  abs(dist_to_start - dist_to_end)
    traversal_time_est   arc_length / max(speed_mps, 0.5 m/s)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.candidate_gen import process_trip_candidates

FEATURE_COLS = [
    "perp_dist",
    "proj_ratio",
    "heading_error",
    "heading_cos",
    "arc_length",
    "dist_to_start",
    "dist_to_end",
    "speed_mps",
    "delta_t",
    "candidate_rank",
    "dist_diff_endpoints",
    "traversal_time_est",
]


def _point_kinematics(track: pd.DataFrame) -> pd.DataFrame:
    """Per-GPS-point delta_t and speed_mps, from consecutive rows of the raw .track file."""
    lon = track["lon"].values
    lat = track["lat"].values
    t = track["time"].values

    mean_lat = lat.mean()
    lat_m = 111139.0
    lon_m = 111139.0 * np.cos(np.radians(mean_lat))

    x = lon * lon_m
    y = lat * lat_m

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dt = np.diff(t, prepend=t[0])
    dx[0] = 0.0
    dy[0] = 0.0
    dt[0] = 0.0

    disp = np.hypot(dx, dy)
    delta_t = np.maximum(dt, 0.0)
    speed_mps = disp / np.maximum(delta_t, 0.1)

    return pd.DataFrame({
        "obs_idx": np.arange(len(track)),
        "delta_t": delta_t,
        "speed_mps": speed_mps,
    })


def extract_features_for_trip(trip_id: str, base_dir: str, k_candidates: int = 10) -> pd.DataFrame:
    """Generates candidates for one trip and attaches the full 12-feature set."""
    candidates_df = process_trip_candidates(trip_id, base_dir=base_dir, k_candidates=k_candidates)
    if candidates_df.empty:
        return candidates_df

    track_path = os.path.join(base_dir, trip_id, f"{trip_id}.track")
    track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "time"])
    kinematics = _point_kinematics(track)

    df = candidates_df.merge(kinematics, on="obs_idx", how="left")

    df["heading_cos"] = np.cos(np.radians(df["heading_error"]))
    df["dist_diff_endpoints"] = (df["dist_to_start"] - df["dist_to_end"]).abs()
    df["traversal_time_est"] = df["arc_length"] / df["speed_mps"].clip(lower=0.5)

    ordered_cols = ["trip_id", "obs_idx", "timestamp", "candidate_arc_id"] + FEATURE_COLS + ["label_y"]
    return df[ordered_cols].round(4)


def build_dataset(
    data_dir: str,
    output_csv: str,
    max_trips: int = None,
    k_candidates: int = 10,
) -> pd.DataFrame:
    """Builds the 12-feature candidate dataset across all (or the first max_trips) trips in data_dir."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    trip_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if max_trips is not None:
        trip_dirs = trip_dirs[:max_trips]

    print(f"Extracting {len(FEATURE_COLS)} features for {len(trip_dirs)} trips from {data_dir}...")
    dfs = []
    for idx, t in enumerate(trip_dirs):
        df_trip = extract_features_for_trip(t, base_dir=data_dir, k_candidates=k_candidates)
        if not df_trip.empty:
            dfs.append(df_trip)
            print(f"  [{idx + 1}/{len(trip_dirs)}] Trip {t}: {len(df_trip):,} rows.")

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(all_df):,} rows ({len(FEATURE_COLS)} features) to: {output_csv}")
        return all_df
    return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="Student folder/train")
    parser.add_argument("--test_dir", default="Student folder/student_test")
    parser.add_argument("--train_out", default="results/train_features.csv")
    parser.add_argument("--test_out", default="results/test_features.csv")
    parser.add_argument("--k_candidates", type=int, default=10)
    args = parser.parse_args()

    build_dataset(args.train_dir, args.train_out, k_candidates=args.k_candidates)
    build_dataset(args.test_dir, args.test_out, k_candidates=args.k_candidates)
