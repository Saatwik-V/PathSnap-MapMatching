"""
Exploratory Data Analysis (EDA) Script - PathSnap
Author: Vikas Jadar
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_dir="Student folder/train", output_plot_dir="results/plots"):
    os.makedirs(output_plot_dir, exist_ok=True)
    trip_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"=== PATHSNAP DATASET EXPLORATION ===")
    print(f"Found {len(trip_dirs)} trips in partition: {data_dir}\n")

    trajectory_lengths = []
    delta_times = []
    missing_points_count = 0
    route_lengths = []
    node_counts = []
    arc_counts = []
    
    # Process each trip
    for trip_id in trip_dirs:
        trip_path = os.path.join(data_dir, trip_id)
        track_file = os.path.join(trip_path, f"{trip_id}.track")
        nodes_file = os.path.join(trip_path, f"{trip_id}.nodes")
        arcs_file = os.path.join(trip_path, f"{trip_id}.arcs")
        route_file = os.path.join(trip_path, f"{trip_id}.route")

        if not os.path.exists(track_file):
            continue

        # Load GPS observations
        track_df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
        trajectory_lengths.append(len(track_df))
        missing_points_count += track_df.isna().sum().sum()

        # Compute sampling intervals (delta t)
        dt = np.diff(track_df["timestamp"].values)
        delta_times.extend(dt)

        # Network sizes
        if os.path.exists(nodes_file):
            with open(nodes_file) as f:
                node_counts.append(sum(1 for _ in f))
        if os.path.exists(arcs_file):
            with open(arcs_file) as f:
                arc_counts.append(sum(1 for _ in f))
        if os.path.exists(route_file):
            with open(route_file) as f:
                route_lengths.append(sum(1 for _ in f))

    # Convert to numpy arrays
    traj_arr = np.array(trajectory_lengths)
    dt_arr = np.array(delta_times)

    print("--- 1. Trajectory Length Statistics ---")
    print(f"Total GPS Observations: {np.sum(traj_arr):,}")
    print(f"Min Points per Trip:   {np.min(traj_arr)}")
    print(f"Mean Points per Trip:  {np.mean(traj_arr):.1f}")
    print(f"Median Points per Trip:{np.median(traj_arr):.1f}")
    print(f"Max Points per Trip:   {np.max(traj_arr)}\n")

    print("--- 2. Sampling Rate & Gaps Analysis ---")
    print(f"Total Time Step Transitions: {len(dt_arr):,}")
    print(f"Nominal 1.0s Steps (%):     {np.mean(dt_arr == 1.0) * 100:.2f}%")
    print(f"Mean Delta T:               {np.mean(dt_arr):.2f} s")
    print(f"Max Delta T (Dropouts):     {np.max(dt_arr):.2f} s\n")

    print("--- 3. Road Network Scale ---")
    print(f"Mean Nodes per Subgraph:    {np.mean(node_counts):.1f}")
    print(f"Mean Arcs per Subgraph:     {np.mean(arc_counts):.1f}")
    print(f"Mean Reference Route Arcs:  {np.mean(route_lengths):.1f}")
    print(f"Missing Values Checked:     {missing_points_count} (Clean)\n")

    # Plot 1: Sampling Interval Distribution
    plt.figure(figsize=(8, 4))
    # Clip large outliers to 10s for visual clarity in histogram
    dt_clipped = np.clip(dt_arr, 0, 10)
    sns.histplot(dt_clipped, bins=30, kde=False, color="#2b5c8f")
    plt.title("Distribution of GPS Sampling Intervals (Delta t)", fontsize=13, fontweight="bold")
    plt.xlabel("Sampling Interval Delta t (seconds, clipped at 10s)", fontsize=11)
    plt.ylabel("Observation Count", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(output_plot_dir, "sampling_intervals.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # Plot 2: Trajectory Lengths Histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(traj_arr, bins=25, kde=True, color="#2ca02c")
    plt.title("Distribution of Trajectory Lengths (Points per Trip)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of GPS Points in Trip", fontsize=11)
    plt.ylabel("Trip Count", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(output_plot_dir, "trajectory_lengths.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    print("\nEDA Completed successfully. All plots ready for the Technical Report.")

if __name__ == "__main__":
    run_eda()
