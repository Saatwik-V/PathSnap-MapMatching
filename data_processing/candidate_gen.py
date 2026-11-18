"""
Candidate Generation Pipeline - PathSnap
Authors: Vikas Jadar & Saatwik
Finds the nearest candidate road segments per GPS observation and computes the 6 base
geometric features (perp_dist, proj_ratio, heading_error, arc_length, dist_to_start,
dist_to_end) plus candidate_rank and the ground-truth label.

The full 12-feature table (adding kinematic/derived features) used for model training
and inference is built on top of this by feature_engineering/extract_features.py -- see
that module for the complete feature list.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def process_trip_candidates(
    trip_id: str,
    base_dir: str = "Student folder/train",
    k_candidates: int = 10,
    max_radius_m: float = 60.0
) -> pd.DataFrame:
    """
    Extracts candidate road segments and features for a single trip.
    """
    trip_path = os.path.join(base_dir, trip_id)
    track_p = os.path.join(trip_path, f"{trip_id}.track")
    nodes_p = os.path.join(trip_path, f"{trip_id}.nodes")
    arcs_p = os.path.join(trip_path, f"{trip_id}.arcs")
    route_p = os.path.join(trip_path, f"{trip_id}.route")

    if not (os.path.exists(track_p) and os.path.exists(nodes_p) and os.path.exists(arcs_p)):
        return pd.DataFrame()

    track = pd.read_csv(track_p, sep=r"\s+", header=None, names=["lon", "lat", "time"])
    nodes = pd.read_csv(nodes_p, sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(arcs_p, sep=r"\s+", header=None, names=["u", "v"])
    
    route_arcs = []
    if os.path.exists(route_p):
        route = pd.read_csv(route_p, sep=r"\s+", header=None, names=["arc_id"])
        route_arcs = set(route["arc_id"].values)

    # Dynamic local metric projection (meters)
    mean_lat = track["lat"].mean()
    lat_m = 111139.0
    lon_m = 111139.0 * np.cos(np.radians(mean_lat))

    node_x = nodes["lon"].values * lon_m
    node_y = nodes["lat"].values * lat_m

    u_indices = arcs["u"].values
    v_indices = arcs["v"].values

    u_x = node_x[u_indices]
    u_y = node_y[u_indices]
    v_x = node_x[v_indices]
    v_y = node_y[v_indices]

    dx = v_x - u_x
    dy = v_y - u_y
    arc_lengths = np.hypot(dx, dy)
    arc_lengths_safe = np.where(arc_lengths == 0, 1e-6, arc_lengths)

    # Arc midpoints for fast KD-Tree spatial index
    mid_x = (u_x + v_x) * 0.5
    mid_y = (u_y + v_y) * 0.5
    tree = cKDTree(np.column_stack([mid_x, mid_y]))

    pt_x = track["lon"].values * lon_m
    pt_y = track["lat"].values * lat_m
    N_pts = len(track)

    # GPS headings (velocity vector)
    vx = np.gradient(pt_x)
    vy = np.gradient(pt_y)
    gps_headings = np.arctan2(vy, vx)

    rows = []
    # Query spatial tree
    for i in range(N_pts):
        px, py = pt_x[i], pt_y[i]
        t_sec = track["time"].iloc[i]
        
        # Query k nearest arcs
        dists, cand_arc_ids = tree.query([px, py], k=k_candidates)
        if np.isscalar(cand_arc_ids):
            cand_arc_ids = [cand_arc_ids]

        for rank, arc_id in enumerate(cand_arc_ids):
            ax, ay = u_x[arc_id], u_y[arc_id]
            bx, by = v_x[arc_id], v_y[arc_id]
            seg_dx = dx[arc_id]
            seg_dy = dy[arc_id]
            L = arc_lengths_safe[arc_id]

            # Vector from start node u to GPS point p
            ap_x = px - ax
            ap_y = py - ay

            # Projection along segment
            proj = (ap_x * seg_dx + ap_y * seg_dy) / (L * L)
            t_clamped = max(0.0, min(1.0, proj))

            # Closest point on segment
            closest_x = ax + t_clamped * seg_dx
            closest_y = ay + t_clamped * seg_dy
            perp_dist = np.hypot(px - closest_x, py - closest_y)

            # Distances to endpoints
            dist_to_start = np.hypot(px - ax, py - ay)
            dist_to_end = np.hypot(px - bx, py - by)

            # Heading error
            arc_heading = np.arctan2(seg_dy, seg_dx)
            h_diff = np.abs(np.degrees(gps_headings[i] - arc_heading))
            heading_error = min(h_diff, 360.0 - h_diff)
            if heading_error > 180.0:
                heading_error = 360.0 - heading_error

            # Label assignment
            is_true_route = 1 if arc_id in route_arcs else 0

            rows.append({
                "trip_id": trip_id,
                "obs_idx": i,
                "timestamp": t_sec,
                "candidate_arc_id": int(arc_id),
                "candidate_rank": rank,
                "perp_dist": round(float(perp_dist), 2),
                "proj_ratio": round(float(t_clamped), 3),
                "heading_error": round(float(heading_error), 2),
                "arc_length": round(float(L), 2),
                "dist_to_start": round(float(dist_to_start), 2),
                "dist_to_end": round(float(dist_to_end), 2),
                "label_y": is_true_route
            })

    return pd.DataFrame(rows)

def build_dataset(
    data_dir: str = "Student folder/train",
    output_csv: str = "results/train_features.csv",
    max_trips: int = 10
):
    """
    Builds the candidate features dataset across trips.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    trip_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    selected_trips = trip_dirs[:max_trips]
    
    print(f"Building candidate feature dataset for {len(selected_trips)} trips from {data_dir}...")
    dfs = []
    for idx, t in enumerate(selected_trips):
        df_trip = process_trip_candidates(t, base_dir=data_dir)
        if not df_trip.empty:
            dfs.append(df_trip)
            print(f"  [{idx+1}/{len(selected_trips)}] Trip {t}: {len(df_trip):,} candidate rows generated.")

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully generated {len(all_df):,} rows.")
        print(f"Saved to: {output_csv}")
        return all_df
    return pd.DataFrame()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=int, default=5, help="Number of trips to process for initial test")
    args = parser.parse_args()
    build_dataset(max_trips=args.trips)
