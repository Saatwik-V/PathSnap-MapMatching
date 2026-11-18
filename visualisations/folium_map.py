"""
Folium Geographical Visualizer - PathSnap
Author: Uttam
Generates multi-layer interactive maps for trips.
"""

import os
import folium
import numpy as np
import pandas as pd

LEGEND_HTML = """
<div style="
    position: fixed;
    bottom: 24px;
    left: 12px;
    z-index: 9999;
    background-color: rgba(255, 255, 255, 0.92);
    padding: 12px 14px;
    border-radius: 8px;
    border: 1px solid #999;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 1.6;
">
    <b>PathSnap Legend</b><br>
    <span style="color:#1f77b4; font-size:16px;">&#9679;</span> Raw Noisy GPS (1 Hz observations)<br>
    <span style="color:#2ca02c; font-weight:bold;">&#8213;</span> Ground-Truth Reference Route<br>
    <span style="color:#d62728; font-weight:bold;">&#8213;&#8213;</span> PathSnap Snapped &amp; Decoded Path<br>
    <span style="color:#999999;">&#8213;</span> Bounding Road Network Graph
</div>
"""


def _compute_point_speeds(track_df: pd.DataFrame) -> np.ndarray:
    """Per-point speed (m/s) from consecutive GPS points, for hover tooltips."""
    lon = track_df["lon"].values
    lat = track_df["lat"].values
    t = track_df["timestamp"].values if "timestamp" in track_df.columns else track_df["time"].values

    mean_lat = lat.mean()
    lat_m = 111139.0
    lon_m = 111139.0 * np.cos(np.radians(mean_lat))

    x = lon * lon_m
    y = lat * lat_m

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dt = np.diff(t, prepend=t[0])
    dx[0] = dy[0] = dt[0] = 0.0

    disp = np.hypot(dx, dy)
    return disp / np.maximum(dt, 0.1)


def build_trip_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame = None,
    arcs_df: pd.DataFrame = None,
    route_arcs: list = None,
    predicted_arcs: list = None,
    trip_id: str = "Trip"
) -> folium.Map:
    """
    Builds a 4-layer interactive Folium map:
    1. Blue CircleMarkers: Raw noisy GPS points (hover: obs index, time, speed)
    2. Grey Polylines: Local bounding road network arcs
    3. Green Bold Line: Reference ground-truth route
    4. Red Dashed Line: PathSnap predicted route
    Auto-fits zoom bounds to the trip's GPS extent and overlays a floating legend.
    """
    time_col = "timestamp" if "timestamp" in track_df.columns else "time"
    speeds = _compute_point_speeds(track_df)

    min_lat, max_lat = track_df["lat"].min(), track_df["lat"].max()
    min_lon, max_lon = track_df["lon"].min(), track_df["lon"].max()

    m = folium.Map(
        location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
        tiles="OpenStreetMap"
    )
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Layer 0: Bounding road-network graph (grey). Rendered as a SINGLE multi-segment
    # PolyLine (one DOM element) instead of one folium.PolyLine per arc -- dense local
    # networks can have tens of thousands of arcs near a trip's bounding box, and one
    # element per arc was producing 10-25MB HTML files per trip.
    if nodes_df is not None and arcs_df is not None:
        fg_graph = folium.FeatureGroup(name="0. Road Network Graph (Grey)")
        pad = 0.0015  # ~165m
        node_mask = (
            (nodes_df["lat"] >= min_lat - pad) & (nodes_df["lat"] <= max_lat + pad) &
            (nodes_df["lon"] >= min_lon - pad) & (nodes_df["lon"] <= max_lon + pad)
        )
        visible_node_ids = set(nodes_df.index[node_mask])
        relevant_arcs = arcs_df[
            arcs_df["u"].isin(visible_node_ids) | arcs_df["v"].isin(visible_node_ids)
        ]
        max_arcs = 4000
        if len(relevant_arcs) > max_arcs:
            relevant_arcs = relevant_arcs.sample(max_arcs, random_state=42)

        segments = [
            [
                [nodes_df.iloc[int(arc["u"])]["lat"], nodes_df.iloc[int(arc["u"])]["lon"]],
                [nodes_df.iloc[int(arc["v"])]["lat"], nodes_df.iloc[int(arc["v"])]["lon"]],
            ]
            for _, arc in relevant_arcs.iterrows()
        ]
        if segments:
            folium.PolyLine(locations=segments, color="#999999", weight=1.5, opacity=0.5).add_to(fg_graph)
        fg_graph.add_to(m)

    # Layer 1: Raw GPS Points (subsampled to keep the exported HTML lightweight)
    fg_gps = folium.FeatureGroup(name="1. Raw GPS Observations (Blue Dots)")
    max_markers = 500
    step = max(1, len(track_df) // max_markers)
    for idx in range(0, len(track_df), step):
        row = track_df.iloc[idx]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            tooltip=f"Obs #{idx}<br>Time: {row[time_col]}s<br>Speed: {speeds[idx]:.2f} m/s",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Layer 2: Reference Ground Truth Route (Green)
    if nodes_df is not None and arcs_df is not None and route_arcs is not None:
        fg_ref = folium.FeatureGroup(name="2. Ground Truth Route (Green)")
        ref_coords = []
        for arc_id in route_arcs:
            if arc_id < len(arcs_df):
                u = int(arcs_df.iloc[arc_id]["u"])
                v = int(arcs_df.iloc[arc_id]["v"])
                u_coord = [nodes_df.iloc[u]["lat"], nodes_df.iloc[u]["lon"]]
                v_coord = [nodes_df.iloc[v]["lat"], nodes_df.iloc[v]["lon"]]
                ref_coords.append(u_coord)
                ref_coords.append(v_coord)

        if ref_coords:
            folium.PolyLine(
                locations=ref_coords,
                color="#2ca02c",
                weight=5,
                opacity=0.85,
                tooltip="Ground Truth Path"
            ).add_to(fg_ref)
        fg_ref.add_to(m)

    # Layer 3: PathSnap Predicted Reconstructed Route (Red)
    if nodes_df is not None and arcs_df is not None and predicted_arcs is not None:
        fg_pred = folium.FeatureGroup(name="3. PathSnap Predicted Route (Red)")
        pred_coords = []
        for arc_id in predicted_arcs:
            if arc_id < len(arcs_df):
                u = int(arcs_df.iloc[arc_id]["u"])
                v = int(arcs_df.iloc[arc_id]["v"])
                pred_coords.append([nodes_df.iloc[u]["lat"], nodes_df.iloc[u]["lon"]])
                pred_coords.append([nodes_df.iloc[v]["lat"], nodes_df.iloc[v]["lon"]])

        if pred_coords:
            folium.PolyLine(
                locations=pred_coords,
                color="#d62728",
                weight=5,
                dash_array="6, 6",
                opacity=0.9,
                tooltip="PathSnap Prediction"
            ).add_to(fg_pred)
        fg_pred.add_to(m)

    # Add Layer Control to easily toggle layers
    folium.LayerControl(collapsed=False).add_to(m)

    # Floating legend (fixed position, always visible regardless of toggled layers)
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))

    return m
