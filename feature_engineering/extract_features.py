#!/usr/bin/env python3
"""
PathSnap: Feature Engineering Engine (Phase 2 Expanded 12-Feature Specification)
Author: Vikas Jadar (Data Pipeline & Feature Extraction Specialist)

Calculates the 12 fundamental geometric, kinematic, and topological features for candidate road arcs:
1. perp_dist: Orthogonal perpendicular distance from GPS observation to candidate road segment (meters).
2. proj_ratio: Fractional projection along the segment vector [0.0, 1.0].
3. heading_error: Absolute angular deviation between vehicle trajectory and road arc [0 deg, 180 deg].
4. arc_length: Physical metric length of candidate road segment (meters).
5. dist_to_start: Euclidean metric distance from GPS observation to arc start node (meters).
6. dist_to_end: Euclidean metric distance from GPS observation to arc end node (meters).
7. speed_mps: Instantaneous vehicle speed derived from consecutive GPS points (meters/second).
8. heading_cos: Directional cosine of heading error cos(heading_error) in [-1.0, 1.0].
9. delta_t: Sampling time interval between consecutive observations (seconds).
10. candidate_rank: Ordinal rank index of candidate arc sorted by perpendicular distance (0-based or 1-based).
11. dist_diff_endpoints: Absolute difference between endpoint distances |dist_to_start - dist_to_end| (meters).
12. traversal_time_est: Estimated time required to traverse candidate segment at current speed: arc_length / max(speed, 0.5) (seconds).
"""

import numpy as np

EARTH_RADIUS = 6371000.0  # Mean spherical Earth radius in meters

FEATURE_COLUMNS_12 = [
    "perp_dist",
    "proj_ratio",
    "heading_error",
    "arc_length",
    "dist_to_start",
    "dist_to_end",
    "speed_mps",
    "heading_cos",
    "delta_t",
    "candidate_rank",
    "dist_diff_endpoints",
    "traversal_time_est"
]

def project_local_metric(lons, lats, ref_lon=None, ref_lat=None):
    """
    Projects WGS84 (lon, lat) degrees to local Cartesian metric (x, y) coordinates
    using an Equirectangular projection centered at the trajectory's centroid.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    
    if ref_lon is None:
        ref_lon = float(np.mean(lons))
    if ref_lat is None:
        ref_lat = float(np.mean(lats))
        
    lat0_rad = np.radians(ref_lat)
    lon0_rad = np.radians(ref_lon)
    
    x = EARTH_RADIUS * (np.radians(lons) - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS * (np.radians(lats) - lat0_rad)
    
    return x, y, ref_lon, ref_lat

def compute_kinematics(track_x, track_y, track_t):
    """
    Computes instantaneous vehicle movement kinematics from track coordinates:
    - headings: Instantaneous heading angles in degrees [0, 360)
    - speeds: Instantaneous speed in meters per second
    - delta_ts: Sampling interval delta_t in seconds
    """
    n = len(track_x)
    headings = np.zeros(n, dtype=float)
    speeds = np.zeros(n, dtype=float)
    delta_ts = np.ones(n, dtype=float)
    
    if n < 2:
        return headings, speeds, delta_ts
        
    track_t = np.asarray(track_t, dtype=float)
    dt_raw = np.diff(track_t)
    dt_safe = np.where(dt_raw <= 0, 1.0, dt_raw)
    
    delta_ts[1:] = dt_safe
    delta_ts[0] = dt_safe[0] if len(dt_safe) > 0 else 1.0
    
    dx = np.diff(track_x)
    dy = np.diff(track_y)
    step_dist = np.hypot(dx, dy)
    step_speed = step_dist / dt_safe
    
    speeds[1:] = step_speed
    speeds[0] = step_speed[0] if len(step_speed) > 0 else 0.0
    
    # Headings
    step_headings = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    headings[1:] = step_headings
    headings[0] = step_headings[0] if len(step_headings) > 0 else 0.0
    
    # Stationary filter: hold heading when speed < 0.1 m/s
    last_h = headings[0]
    for i in range(n):
        if speeds[i] >= 0.1:
            last_h = headings[i]
        headings[i] = last_h
        
    return headings, speeds, delta_ts

def compute_candidate_features(
    px, py, v_heading, ax1, ay1, ax2, ay2,
    speed_mps=None, delta_t=None, candidate_rank=None
):
    """
    Vectorized computation of the 12 features for candidate pairs.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    v_heading = np.asarray(v_heading, dtype=float)
    ax1 = np.asarray(ax1, dtype=float)
    ay1 = np.asarray(ay1, dtype=float)
    ax2 = np.asarray(ax2, dtype=float)
    ay2 = np.asarray(ay2, dtype=float)
    
    N = len(ax1)
    
    if speed_mps is None:
        speed_mps = np.zeros(N, dtype=float)
    else:
        speed_mps = np.broadcast_to(np.asarray(speed_mps, dtype=float), N)
        
    if delta_t is None:
        delta_t = np.ones(N, dtype=float)
    else:
        delta_t = np.broadcast_to(np.asarray(delta_t, dtype=float), N)
        
    if candidate_rank is None:
        candidate_rank = np.arange(N, dtype=float)
    else:
        candidate_rank = np.asarray(candidate_rank, dtype=float)

    # Segment vectors
    seg_dx = ax2 - ax1
    seg_dy = ay2 - ay1
    
    # 4. arc_length
    arc_length = np.hypot(seg_dx, seg_dy)
    L2 = np.maximum(arc_length * arc_length, 1e-12)
    
    # Point-to-start vector
    pt_dx = px - ax1
    pt_dy = py - ay1
    
    # 2. proj_ratio
    raw_proj = (pt_dx * seg_dx + pt_dy * seg_dy) / L2
    proj_ratio = np.clip(raw_proj, 0.0, 1.0)
    
    # Projected coordinates on segment
    proj_x = ax1 + proj_ratio * seg_dx
    proj_y = ay1 + proj_ratio * seg_dy
    
    # 1. perp_dist
    perp_dist = np.hypot(px - proj_x, py - proj_y)
    
    # 5. dist_to_start & 6. dist_to_end
    dist_to_start = np.hypot(px - ax1, py - ay1)
    dist_to_end = np.hypot(px - ax2, py - ay2)
    
    # 3. heading_error
    seg_heading = (np.degrees(np.arctan2(seg_dy, seg_dx)) + 360.0) % 360.0
    diff = np.abs(v_heading - seg_heading) % 360.0
    heading_error = np.where(diff > 180.0, 360.0 - diff, diff)
    
    # 8. heading_cos: directional alignment in [-1.0, 1.0]
    heading_cos = np.cos(np.radians(heading_error))
    
    # 11. dist_diff_endpoints: |dist_to_start - dist_to_end|
    dist_diff_endpoints = np.abs(dist_to_start - dist_to_end)
    
    # 12. traversal_time_est: arc_length / max(speed_mps, 0.5)
    speed_safe = np.maximum(speed_mps, 0.5)
    traversal_time_est = arc_length / speed_safe
    
    return {
        "perp_dist": perp_dist,
        "proj_ratio": proj_ratio,
        "heading_error": heading_error,
        "arc_length": arc_length,
        "dist_to_start": dist_to_start,
        "dist_to_end": dist_to_end,
        "speed_mps": speed_mps,
        "heading_cos": heading_cos,
        "delta_t": delta_t,
        "candidate_rank": candidate_rank,
        "dist_diff_endpoints": dist_diff_endpoints,
        "traversal_time_est": traversal_time_est
    }
