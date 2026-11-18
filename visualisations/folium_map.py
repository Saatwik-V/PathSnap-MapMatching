"""
Folium Geographical Visualizer - PathSnap
Author: Uttam
Generates multi-layer interactive maps for trips.
"""

import os
import folium
import numpy as np
import pandas as pd

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
    1. Blue CircleMarkers: Raw noisy GPS points
    2. Grey Polylines: Local bounding road network arcs
    3. Green Bold Line: Reference ground-truth route
    4. Red Dashed Line: PathSnap predicted route
    """
    # Compute center of map
    mean_lat = track_df["lat"].mean()
    mean_lon = track_df["lon"].mean()

    m = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=14,
        tiles="CartoDB positron"
    )

    # Layer 1: Raw GPS Points
    fg_gps = folium.FeatureGroup(name="1. Raw GPS Observations (Blue Dots)")
    for idx, row in track_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            popup=f"Obs #{idx}<br>Time: {row['timestamp']}s"
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
                weight=3,
                dash_array="6, 6",
                opacity=0.9,
                tooltip="PathSnap Prediction"
            ).add_to(fg_pred)
        fg_pred.add_to(m)

    # Add Layer Control to easily toggle layers
    folium.LayerControl(collapsed=False).add_to(m)

    return m
