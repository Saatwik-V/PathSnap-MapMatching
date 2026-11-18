#!/usr/bin/env python3
"""
PathSnap: Spatial Candidate Generation & Ground-Truth Alignment Pipeline
Author: Vikas Jadar (Data Pipeline & Feature Extraction Specialist)

Generates candidate road segments for each GPS observation:
1. Spatially indexes graph arcs using cKDTree over segment midpoints.
2. Extracts top-K nearest road segment candidates per GPS point via exact metric distance.
3. Resolves reference ground-truth alignment from the sequential .route walk.
4. Computes 6 kinematic/geometric features via feature_engineering.extract_features.
5. Evaluates Candidate Miss Rate and writes results/train_features.csv.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Add repo root to module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_engineering.extract_features import (
    project_local_metric,
    compute_vehicle_headings,
    compute_candidate_features
)

def find_dataset_dir():
    candidates = [
        "Student folder",
        os.path.join("ML hackathon", "Student folder"),
        os.path.join("..", "Student folder"),
        os.path.join("..", "ML hackathon", "Student folder")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train")):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate 'Student folder/train' in repository.")

def align_ground_truth_route(track_x, track_y, track_t, ax1, ay1, ax2, ay2, route):
    """
    Maps each GPS point to its reference ground-truth arc in the .route sequence
    using a normalized temporal window along the chronological route walk.
    """
    T = len(track_x)
    M = len(route)
    if M == 0:
        return np.zeros(T, dtype=int), np.zeros(T, dtype=float)
        
    duration = track_t[-1] - track_t[0]
    t_norm = (track_t - track_t[0]) / max(1e-6, duration)
    
    assigned_route = np.zeros(T, dtype=int)
    alignment_dists = np.zeros(T, dtype=float)
    
    # Adaptive window: search within +/-20% of route length or at least 30 arcs
    w = max(30, int(0.20 * M))
    
    for p in range(T):
        px, py = track_x[p], track_y[p]
        center = int(t_norm[p] * M)
        w_start = max(0, center - w)
        w_end = min(M, center + w + 1)
        
        sx1, sy1 = ax1[w_start:w_end], ay1[w_start:w_end]
        sx2, sy2 = ax2[w_start:w_end], ay2[w_start:w_end]
        
        dx = sx2 - sx1
        dy = sy2 - sy1
        L2 = dx * dx + dy * dy
        
        # Projection factor
        t = np.clip(((px - sx1) * dx + (py - sy1) * dy) / np.maximum(L2, 1e-9), 0.0, 1.0)
        proj_x = sx1 + t * dx
        proj_y = sy1 + t * dy
        
        dists = np.hypot(px - proj_x, py - proj_y)
        best_local = np.argmin(dists)
        
        assigned_route[p] = route[w_start + best_local]
        alignment_dists[p] = dists[best_local]
        
    return assigned_route, alignment_dists

def process_trip(trip_dir, trip_id, k_candidates=10, k_search=150, is_test=False):
    """
    Generates candidates and extracts features for a single trip.
    """
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    track_data = np.loadtxt(track_path)
    nodes_data = np.loadtxt(nodes_path)
    arcs_data = np.loadtxt(arcs_path, dtype=int)
    
    has_route = os.path.exists(route_path) and not is_test
    if has_route:
        route_data = np.loadtxt(route_path, dtype=int)
        if route_data.ndim == 0:
            route_data = np.array([route_data.item()])
    else:
        route_data = np.array([], dtype=int)
        
    n_pts = len(track_data)
    n_arcs = len(arcs_data)
    
    lons = track_data[:, 0]
    lats = track_data[:, 1]
    times = track_data[:, 2]
    
    # 1. Metric projection
    track_x, track_y, ref_lon, ref_lat = project_local_metric(lons, lats)
    nodes_x, nodes_y, _, _ = project_local_metric(nodes_data[:, 0], nodes_data[:, 1], ref_lon, ref_lat)
    
    # 2. Vehicle headings
    v_headings = compute_vehicle_headings(track_x, track_y)
    
    # 3. Road network segments
    from_nodes = arcs_data[:, 0]
    to_nodes = arcs_data[:, 1]
    ax1 = nodes_x[from_nodes]
    ay1 = nodes_y[from_nodes]
    ax2 = nodes_x[to_nodes]
    ay2 = nodes_y[to_nodes]
    
    # Midpoints for fast spatial indexing
    mid_x = 0.5 * (ax1 + ax2)
    mid_y = 0.5 * (ay1 + ay2)
    
    tree = cKDTree(np.column_stack([mid_x, mid_y]))
    
    # 4. Ground-truth route alignment (if available)
    if has_route and len(route_data) > 0:
        r_from = arcs_data[route_data, 0]
        r_to = arcs_data[route_data, 1]
        r_ax1, r_ay1 = nodes_x[r_from], nodes_y[r_from]
        r_ax2, r_ay2 = nodes_x[r_to], nodes_y[r_to]
        true_arcs, align_dists = align_ground_truth_route(
            track_x, track_y, times, r_ax1, r_ay1, r_ax2, r_ay2, route_data
        )
    else:
        true_arcs = np.full(n_pts, -1, dtype=int)
        align_dists = np.zeros(n_pts, dtype=float)
        
    # 5. Spatial candidate extraction
    k_query = min(k_search, n_arcs)
    _, query_neighbors = tree.query(np.column_stack([track_x, track_y]), k=k_query)
    
    trip_records = []
    miss_count = 0
    
    for p in range(n_pts):
        px = track_x[p]
        py = track_y[p]
        vh = v_headings[p]
        cand_indices = query_neighbors[p]
        if np.ndim(cand_indices) == 0:
            cand_indices = np.array([cand_indices])
            
        # Point-to-segment perpendicular distance for queried candidates
        c_ax1 = ax1[cand_indices]
        c_ay1 = ay1[cand_indices]
        c_ax2 = ax2[cand_indices]
        c_ay2 = ay2[cand_indices]
        
        dx = c_ax2 - c_ax1
        dy = c_ay2 - c_ay1
        L2 = np.maximum(dx * dx + dy * dy, 1e-9)
        t = np.clip(((px - c_ax1) * dx + (py - c_ay1) * dy) / L2, 0.0, 1.0)
        proj_x = c_ax1 + t * dx
        proj_y = c_ay1 + t * dy
        dists = np.hypot(px - proj_x, py - proj_y)
        
        # Sort by exact perpendicular distance and take top-K
        top_k_order = np.argsort(dists)[:k_candidates]
        top_k_arcs = cand_indices[top_k_order]
        
        true_arc = true_arcs[p]
        if has_route:
            if true_arc not in top_k_arcs:
                miss_count += 1
                
        # Extract features for these top-K candidates
        feats = compute_candidate_features(
            px=px,
            py=py,
            v_heading=vh,
            ax1=ax1[top_k_arcs],
            ay1=ay1[top_k_arcs],
            ax2=ax2[top_k_arcs],
            ay2=ay2[top_k_arcs]
        )
        
        for k_idx in range(len(top_k_arcs)):
            arc_id = int(top_k_arcs[k_idx])
            is_match = 1 if (has_route and arc_id == true_arc) else 0
            
            trip_records.append({
                "trip_id": str(trip_id),
                "point_idx": int(p),
                "arc_id": arc_id,
                "perp_dist": round(float(feats["perp_dist"][k_idx]), 4),
                "proj_ratio": round(float(feats["proj_ratio"][k_idx]), 4),
                "heading_error": round(float(feats["heading_error"][k_idx]), 2),
                "arc_length": round(float(feats["arc_length"][k_idx]), 2),
                "dist_to_start": round(float(feats["dist_to_start"][k_idx]), 2),
                "dist_to_end": round(float(feats["dist_to_end"][k_idx]), 2),
                "label_y": is_match,
                "y": is_match
            })
            
    miss_rate = (miss_count / n_pts) if n_pts > 0 else 0.0
    return pd.DataFrame(trip_records), n_pts, miss_count, miss_rate

def main():
    parser = argparse.ArgumentParser(description="PathSnap Candidate Generation & Feature Extraction")
    parser.add_argument("--trips", type=str, default="10",
                        help="Number of trips to process (e.g. '10', '80', or 'all'). Default: 10")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train",
                        help="Dataset partition to process ('train' or 'test'). Default: train")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest candidates per GPS point (default: 10)")
    parser.add_argument("--k_search", type=int, default=150,
                        help="KDTree neighbors to pre-filter before exact distance (default: 150)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional custom output path for features CSV")
    
    args = parser.parse_args()
    start_time = time.time()
    
    base_dir = find_dataset_dir()
    split_folder = "train" if args.split == "train" else "student_test"
    target_dir = os.path.join(base_dir, split_folder)
    
    all_trips = sorted([
        d for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d))
    ])
    
    if args.trips.lower() == "all":
        selected_trips = all_trips
    else:
        num = min(int(args.trips), len(all_trips))
        selected_trips = all_trips[:num]
        
    print(f"============================================================")
    print(f" PathSnap Candidate Generation & Feature Extraction Pipeline")
    print(f" Partition: {args.split.upper()} | Selected Trips: {len(selected_trips)}")
    print(f" Candidates per Observation (K): {args.k}")
    print(f"============================================================\n")
    
    total_points = 0
    total_misses = 0
    trip_dfs = []
    
    for idx, tid in enumerate(selected_trips, 1):
        t0 = time.time()
        trip_dir = os.path.join(target_dir, tid)
        df_trip, n_pts, misses, m_rate = process_trip(
            trip_dir=trip_dir,
            trip_id=tid,
            k_candidates=args.k,
            k_search=args.k_search,
            is_test=(args.split == "test")
        )
        t_elapsed = time.time() - t0
        
        trip_dfs.append(df_trip)
        total_points += n_pts
        total_misses += misses
        
        status_line = (
            f"[{idx:2d}/{len(selected_trips)}] Trip {tid}: {n_pts:5d} pts -> "
            f"{len(df_trip):6d} cand rows | Misses: {misses:3d} ({m_rate:.2%}) "
            f"| Time: {t_elapsed:.2f}s"
        )
        print(status_line)
        
    df_all = pd.concat(trip_dfs, ignore_index=True)
    
    # Define output destination
    os.makedirs("results", exist_ok=True)
    if args.output:
        out_csv = args.output
    else:
        out_csv = os.path.join("results", f"{args.split}_features.csv")
        
    df_all.to_csv(out_csv, index=False)
    
    overall_miss_rate = total_misses / max(1, total_points)
    overall_recall = 1.0 - overall_miss_rate
    elapsed_total = time.time() - start_time
    
    print("\n------------------------------------------------------------")
    print(f" PIPELINE SUMMARY & QUALITY SCORECARD")
    print("------------------------------------------------------------")
    print(f"Total Trips Processed:     {len(selected_trips)}")
    print(f"Total GPS Observations:    {total_points:,}")
    print(f"Total Candidate Rows:      {len(df_all):,}")
    print(f"Positive Labels (y=1):     {(df_all['label_y'] == 1).sum():,} ({(df_all['label_y'] == 1).mean():.2%})")
    print(f"Negative Labels (y=0):     {(df_all['label_y'] == 0).sum():,} ({(df_all['label_y'] == 0).mean():.2%})")
    print(f"Candidate Miss Rate:       {overall_miss_rate:.2%} ({total_misses:,}/{total_points:,})")
    print(f"Candidate Spatial Recall:  {overall_recall:.2%}")
    print(f"Total Execution Time:      {elapsed_total:.2f}s ({elapsed_total/len(selected_trips):.2f}s/trip)")
    print(f"Output Dataset:            {out_csv}")
    print(f"Columns Verified:          {list(df_all.columns)}")
    print("------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
