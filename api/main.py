"""
PathSnap Backend Service (api/main.py)
Author: Vikas Aihole (FastAPI Backend & Deployment Lead)

Mission:
Hosts the model inference, candidate generation, HMM/Viterbi sequential decoding,
and server-side ground-truth evaluation pipeline on Laptop 1.

Endpoints:
- GET  /health          : Health-check endpoint for Laptop 2 connection verification
- GET  /trips           : List of available reserved test trip IDs
- POST /predict/{trip_id}: Reconstructs vehicle trajectory & computes evaluation metrics
- GET  /predict/{trip_id}: Read-only test interface for convenience

Run with:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api.utils import (
    find_dataset_root,
    list_available_trips,
    load_trip_data,
    reconstruct_route_pipeline,
    compute_route_continuity,
    compute_ground_truth_metrics,
    get_route_coordinates,
)

# Configure Terminal Logging (Evaluators explicitly look at CMD server logs!)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pathsnap_server")

app = FastAPI(
    title="PathSnap: Distributed Map-Matching & Sequential Route Reconstruction API",
    description="Laptop 1 Backend Server hosting ML inference, HMM Viterbi decoding, and server-side ground-truth evaluation.",
    version="1.0.0",
)

# Enable CORS for cross-laptop communication (Laptop 2 Streamlit UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware that logs all incoming requests and measures execution latency.
    """
    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    url_path = request.url.path

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pass request down to endpoint
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        if url_path in ["/health", "/trips", "/"]:
            print(f"[{timestamp}] [API {method}] {url_path} | Client: {client_host} | Status: {response.status_code} | Latency: {duration_ms:.2f}ms")

        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        print(f"[{timestamp}] [API ERROR] {method} {url_path} | Client: {client_host} | Error: {str(exc)} | Latency: {duration_ms:.2f}ms")
        raise exc


@app.get("/")
def root_info():
    """
    Root summary endpoint.
    """
    return {
        "system": "PathSnap Distributed Map-Matching",
        "node": "Laptop 1 (Model Inference & Ground-Truth Evaluation)",
        "lead": "Vikas Aihole",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "trips": "/trips"
    }


@app.get("/health")
def health_check(request: Request):
    """
    Health check endpoint used by Laptop 2 to verify network connectivity.
    """
    timestamp = datetime.now().isoformat()
    client_ip = request.client.host if request.client else "127.0.0.1"

    return {
        "status": "healthy",
        "service": "PathSnap FastAPI Backend",
        "machine": "Laptop 1",
        "client_connected": client_ip,
        "timestamp": timestamp,
        "pre_lunch_checkpoint": "READY"
    }


@app.get("/trips")
def get_evaluation_trips():
    """
    Returns the list of 10 reserved evaluation trips available on the server.
    """
    data_root = find_dataset_root()
    trips = list_available_trips(data_root)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [TRIP DISCOVERY] Found {len(trips)} test trips: {trips}")

    return {
        "trips": trips,
        "count": len(trips),
        "data_root": data_root
    }


@app.post("/predict/{trip_id}")
@app.get("/predict/{trip_id}")
def predict_and_evaluate(trip_id: str, request: Request):
    """
    Core Pipeline Endpoint:
    1. Loads raw .track, .nodes, .arcs
    2. Runs geometric feature extraction & candidate generation
    3. Runs ML scoring + HMM/Viterbi DP trajectory reconstruction
    4. Evaluates predictions against private server-side .route ground truth
    5. Returns clean JSON payload to Laptop 2 (Streamlit UI)
    """
    start_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    client_ip = request.client.host if request.client else "127.0.0.1"

    print("\n" + "=" * 80)
    print(f"[{timestamp}] [EVALUATION REQUEST] POST /predict/{trip_id}")
    print(f"[{timestamp}] Client IP: {client_ip} | Trip ID: {trip_id}")

    try:
        # 1. Load Trip Data
        data = load_trip_data(trip_id)
        track_df = data["track"]
        nodes_df = data["nodes"]
        arcs_df = data["arcs"]
        gt_route_arcs = data["route"]

        num_gps = len(track_df)
        num_nodes = len(nodes_df)
        num_arcs = len(arcs_df)
        num_gt = len(gt_route_arcs)

        print(f"[{timestamp}] Data Ingestion: Loaded {num_gps:,} GPS points | {num_nodes:,} nodes | {num_arcs:,} arcs | {num_gt} ground-truth arcs")

        # 2. Check for Champion Model Artifact (prefer the Optuna-tuned model)
        model_path = os.path.join(REPO_ROOT, "results", "best_tuned_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(REPO_ROOT, "results", "best_model.joblib")
        if not os.path.exists(model_path):
            model_path = None

        # 3. Trajectory Reconstruction (Model + HMM Viterbi)
        print(f"[{timestamp}] Sequential Decoding: Running HMM Viterbi road network snapping...")
        decode_start = time.perf_counter()
        point_pred_arcs, seq_pred_arcs = reconstruct_route_pipeline(
            trip_id=trip_id,
            track_df=track_df,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            model_path=model_path
        )
        decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

        # 4. Server-Side Ground-Truth Evaluation
        continuity_pct = compute_route_continuity(point_pred_arcs, arcs_df)
        gt_metrics = compute_ground_truth_metrics(point_pred_arcs, gt_route_arcs, track_df, nodes_df, arcs_df)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        top1_acc = gt_metrics["top1_accuracy"]
        macro_f1 = gt_metrics["macro_f1"]

        # Evaluator-facing CMD logs (Crucial requirement!)
        print(f"[{timestamp}] Server-Side Ground-Truth Evaluation Results:")
        print(f"    ├── Observation Top-1 Accuracy : {top1_acc * 100:.2f}%")
        print(f"    ├── Route Continuity %         : {continuity_pct:.2f}%")
        print(f"    ├── Macro F1-Score             : {macro_f1:.4f}")
        print(f"    ├── Sequential Decode Time     : {decode_time_ms:.2f} ms")
        print(f"    └── Total Inference Latency    : {total_latency_ms:.2f} ms")
        print(f"[{timestamp}] Response Status: 200 OK | Predicted {len(seq_pred_arcs)} continuous arcs")
        print("=" * 80 + "\n")

        # 5. Extract Map Coordinates for Uttam's Folium Layers
        predicted_coords = get_route_coordinates(seq_pred_arcs, arcs_df, nodes_df)
        reference_coords = get_route_coordinates(gt_route_arcs, arcs_df, nodes_df)

        # Raw GPS points sampled for fast map rendering (up to 500 points for responsiveness)
        step = max(1, len(track_df) // 500)
        sampled_track = track_df.iloc[::step]
        raw_gps = [
            {"lat": float(r["lat"]), "lon": float(r["lon"]), "timestamp": float(r["timestamp"])}
            for _, r in sampled_track.iterrows()
        ]

        # Bounding box
        lat_min, lat_max = float(track_df["lat"].min()), float(track_df["lat"].max())
        lon_min, lon_max = float(track_df["lon"].min()), float(track_df["lon"].max())

        # Construct response payload
        return {
            "trip_id": trip_id,
            "status": "success",
            "timestamp": timestamp,
            "metrics": {
                "top1_accuracy": top1_acc,
                "route_continuity": continuity_pct,
                "macro_f1": macro_f1,
                "precision": gt_metrics.get("precision", 0.0),
                "recall": gt_metrics.get("recall", 0.0),
                "inference_latency_ms": round(total_latency_ms, 2)
            },
            "summary": {
                "num_gps_points": num_gps,
                "num_predicted_arcs": len(seq_pred_arcs),
                "num_reference_arcs": num_gt
            },
            "bounding_box": {
                "min_lat": lat_min,
                "max_lat": lat_max,
                "min_lon": lon_min,
                "max_lon": lon_max,
                "center": [(lat_min + lat_max) * 0.5, (lon_min + lon_max) * 0.5]
            },
            "predicted_route": {
                "arc_ids": seq_pred_arcs,
                "coordinates": predicted_coords
            },
            "reference_route": {
                "arc_ids": gt_route_arcs,
                "coordinates": reference_coords
            },
            "raw_gps": raw_gps
        }

    except Exception as e:
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        print(f"[{timestamp}] [ERROR] Evaluation failed for trip {trip_id}: {str(e)}")
        print("=" * 80 + "\n")
        raise HTTPException(status_code=500, detail=str(e))