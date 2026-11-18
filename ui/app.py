"""
PathSnap Evaluator Dashboard - Laptop 2 (Client UI)
Author: Uttam
"""

import os
import requests
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import sys

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualisations.folium_map import build_trip_map

st.set_page_config(
    page_title="PathSnap | Evaluator Portal",
    page_icon="🗺️",
    layout="wide"
)

# Header
st.title("🗺️ PathSnap: Map-Matching & Route Reconstruction")
st.caption("PES University | UE24CS352A Machine Learning Hackathon | Evaluator Portal")

# Sidebar Configuration
st.sidebar.header("🔌 Server Connection")
api_base_url = st.sidebar.text_input(
    "Laptop 1 API Base URL",
    value="http://localhost:8000",
    help="Enter the IP of Laptop 1 running FastAPI (e.g., http://192.168.1.15:8000)"
)

# Check Connection
is_connected = False
try:
    health_resp = requests.get(f"{api_base_url}/health", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("✅ Connected to Laptop 1 API")
        is_connected = True
    else:
        st.sidebar.warning("⚠️ API reached but status not 200")
except Exception:
    st.sidebar.error("❌ Cannot reach Laptop 1 API. Ensure uvicorn is running.")

# Fetch Trips
available_trips = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]
if is_connected:
    try:
        trips_resp = requests.get(f"{api_base_url}/trips", timeout=2)
        if trips_resp.status_code == 200:
            available_trips = trips_resp.json().get("trips", available_trips)
    except Exception:
        pass

# Main Area Controls
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    selected_trip = st.selectbox(
        "Select an Evaluation Trip to Reconstruct:",
        options=available_trips,
        index=0
    )
with col_ctrl2:
    st.write("")
    st.write("")
    evaluate_button = st.button("🚀 Snap Route & Evaluate", type="primary", use_container_width=True)

# Evaluation Execution
if evaluate_button:
    with st.spinner(f"Running inference on Laptop 1 for Trip {selected_trip}..."):
        try:
            pred_resp = requests.post(f"{api_base_url}/predict/{selected_trip}", timeout=10)
            if pred_resp.status_code == 200:
                data = pred_resp.json()
                
                # Metrics Row
                metrics = data["metrics"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Top-1 Accuracy", f"{metrics['top1_accuracy'] * 100:.1f}%")
                m2.metric("Macro F1-Score", f"{metrics['macro_f1']:.3f}")
                m3.metric("Route Continuity", f"{metrics['route_continuity']:.1f}%")
                m4.metric("Inference Latency", f"{metrics['inference_latency_ms']:.1f} ms")

                st.divider()

                # Map Visualization
                st.subheader(f"Interactive Trajectory Map — Trip {selected_trip}")
                
                # Load local trip files for background map layers
                base_dir = "Student folder/student_test"
                trip_dir = os.path.join(base_dir, selected_trip)
                
                if os.path.exists(os.path.join(trip_dir, f"{selected_trip}.track")):
                    track_df = pd.read_csv(os.path.join(trip_dir, f"{selected_trip}.track"), sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
                    nodes_df = pd.read_csv(os.path.join(trip_dir, f"{selected_trip}.nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
                    arcs_df = pd.read_csv(os.path.join(trip_dir, f"{selected_trip}.arcs"), sep=r"\s+", header=None, names=["u", "v"])
                    route_df = pd.read_csv(os.path.join(trip_dir, f"{selected_trip}.route"), sep=r"\s+", header=None, names=["arc_id"])
                    
                    folium_map = build_trip_map(
                        track_df=track_df,
                        nodes_df=nodes_df,
                        arcs_df=arcs_df,
                        route_arcs=route_df["arc_id"].values.tolist(),
                        predicted_arcs=data.get("predicted_arc_ids", []),
                        trip_id=selected_trip
                    )
                    
                    # Render map cleanly in Streamlit
                    components.html(folium_map._repr_html_(), height=550)
                else:
                    st.info("Local trajectory data not found on client. Showing metric scorecard only.")
            else:
                st.error(f"Error from API server: {pred_resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API server at {api_base_url}: {e}")
else:
    st.info("Select any of the 10 evaluation trips above and click 'Snap Route & Evaluate' to begin.")
