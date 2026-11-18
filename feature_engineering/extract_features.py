#!/usr/bin/env python3
"""
PathSnap: Feature Engineering Engine
Author: Vikas Jadar (Data Pipeline & Feature Extraction Specialist)

Calculates the 6 fundamental geometric and kinematic features for candidate road arcs:
1. perp_dist: Orthogonal perpendicular distance from GPS point to road segment (meters).
2. proj_ratio: Fractional projection along the segment vector [0.0, 1.0].
3. heading_error: Absolute angular deviation between vehicle trajectory and road arc [0 deg, 180 deg].
4. arc_length: Physical metric length of the candidate road segment (meters).
5. dist_to_start: Euclidean distance from GPS point to arc start node (meters).
6. dist_to_end: Euclidean distance from GPS point to arc end node (meters).
"""

import numpy as np

EARTH_RADIUS = 6371000.0  # Mean spherical Earth radius in meters

def project_local_metric(lons, lats, ref_lon=None, ref_lat=None):
    """
    Projects WGS84 (lon, lat) degrees to local Cartesian metric (x, y) coordinates
    using an Equirectangular projection centered at the trajectory's centroid.
    Accurate to within <0.1% over trajectory extents (<100 km).
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

def compute_vehicle_headings(track_x, track_y):
    """
    Computes instantaneous vehicle movement heading in degrees [0, 360)
    from consecutive track points.
    Uses forward difference for first point, backward for last, and central for interior.
    For stationary points (displacement < 0.1m), retains previous heading.
    """
    n = len(track_x)
    headings = np.zeros(n, dtype=float)
    if n < 2:
        return headings
        
    # Forward differences
    dx = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    
    # Interior: central differences
    dx[1:-1] = track_x[2:] - track_x[:-2]
    dy[1:-1] = track_y[2:] - track_y[:-2]
    
    # Boundary points
    dx[0] = track_x[1] - track_x[0]
    dy[0] = track_y[1] - track_y[0]
    dx[-1] = track_x[-1] - track_x[-2]
    dy[-1] = track_y[-1] - track_y[-2]
    
    # Compute angles in degrees [0, 360)
    angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    
    # Handle stationary points where displacement is near zero
    displacements = np.hypot(dx, dy)
    last_valid = angles[0]
    for i in range(n):
        if displacements[i] > 0.1:
            last_valid = angles[i]
        headings[i] = last_valid
        
    return headings

def compute_candidate_features(px, py, v_heading, ax1, ay1, ax2, ay2):
    """
    Vectorized computation of the 6 geometric features for candidate pairs.
    
    Parameters:
    - px, py: GPS point coordinates (scalars or arrays of shape N)
    - v_heading: Vehicle heading in degrees [0, 360) (scalar or array of shape N)
    - ax1, ay1: Candidate arc start node coordinates (arrays of shape N)
    - ax2, ay2: Candidate arc end node coordinates (arrays of shape N)
    
    Returns:
    - dict with keys:
      ['perp_dist', 'proj_ratio', 'heading_error', 'arc_length', 'dist_to_start', 'dist_to_end']
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    v_heading = np.asarray(v_heading, dtype=float)
    ax1 = np.asarray(ax1, dtype=float)
    ay1 = np.asarray(ay1, dtype=float)
    ax2 = np.asarray(ax2, dtype=float)
    ay2 = np.asarray(ay2, dtype=float)
    
    # Segment vectors
    seg_dx = ax2 - ax1
    seg_dy = ay2 - ay1
    
    # 4. arc_length
    arc_length = np.hypot(seg_dx, seg_dy)
    L2 = np.maximum(arc_length * arc_length, 1e-12)
    
    # Vector from start to point
    pt_dx = px - ax1
    pt_dy = py - ay1
    
    # 2. proj_ratio (projection parameter t along segment)
    raw_proj = (pt_dx * seg_dx + pt_dy * seg_dy) / L2
    proj_ratio = np.clip(raw_proj, 0.0, 1.0)
    
    # Projected coordinates on segment
    proj_x = ax1 + proj_ratio * seg_dx
    proj_y = ay1 + proj_ratio * seg_dy
    
    # 1. perp_dist (distance from GPS point to projected point)
    perp_dist = np.hypot(px - proj_x, py - proj_y)
    
    # 5. dist_to_start & 6. dist_to_end
    dist_to_start = np.hypot(px - ax1, py - ay1)
    dist_to_end = np.hypot(px - ax2, py - ay2)
    
    # 3. heading_error
    seg_heading = (np.degrees(np.arctan2(seg_dy, seg_dx)) + 360.0) % 360.0
    diff = np.abs(v_heading - seg_heading) % 360.0
    heading_error = np.where(diff > 180.0, 360.0 - diff, diff)
    
    return {
        "perp_dist": perp_dist,
        "proj_ratio": proj_ratio,
        "heading_error": heading_error,
        "arc_length": arc_length,
        "dist_to_start": dist_to_start,
        "dist_to_end": dist_to_end
    }
