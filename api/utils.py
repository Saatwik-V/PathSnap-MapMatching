"""
PathSnap Utility Module (api/utils.py)
Author: Vikas Aihole (FastAPI Backend & Deployment Lead)

Provides core utilities for:
- Dataset discovery and loading (.track, .nodes, .arcs, .route)
- Local metric projections (WGS84 -> meters)
- Spatial candidate generation and feature extraction
- Trajectory reconstruction (Model scoring + HMM/Viterbi DP)
- Server-side evaluation metrics (Top-1 Accuracy, Route Continuity %, Macro F1)
"""

import os
import glob
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial import cKDTree
import networkx as nx
import joblib

logger = logging.getLogger("pathsnap_api")

# Earth radius in meters
EARTH_RADIUS = 6371000.0


def find_dataset_root() -> str:
    """
    Locates the dataset root directory containing 'train' and 'student_test'
    or trip folders.
    """
    candidates = [
        os.environ.get("PATHSNAP_DATA_DIR", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Student folder", "Student folder")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Student folder")),
        r"C:\Users\Vikas\Downloads\Student folder\Student folder",
        r"C:\Users\Vikas\Downloads\Student folder",
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            if os.path.isdir(os.path.join(path, "student_test")) or os.path.isdir(os.path.join(path, "train")):
                return path
            if "student_test" in path:
                return path

    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def list_available_trips(data_root: Optional[str] = None) -> List[str]:
    """
    Returns sorted list of 10-digit test trip IDs from student_test or train.
    """
    if data_root is None:
        data_root = find_dataset_root()

    search_dirs = [
        os.path.join(data_root, "student_test"),
        data_root,
        os.path.join(data_root, "train")
    ]

    trips = []
    for d in search_dirs:
        if os.path.isdir(d):
            subdirs = [os.path.basename(p) for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)]
            valid = [s for s in subdirs if s.isdigit() and len(s) == 8]
            if valid:
                trips = sorted(valid)
                break

    if not trips:
        trips = ["00000000", "00000007", "00000009", "00000040", "00000052",
                 "00000056", "00000065", "00000076", "00000077", "00000091"]
    return trips


def find_trip_folder(trip_id: str, data_root: Optional[str] = None) -> str:
    """
    Finds the directory containing the trip files for a given trip_id.
    """
    if data_root is None:
        data_root = find_dataset_root()

    candidates = [
        os.path.join(data_root, "student_test", trip_id),
        os.path.join(data_root, trip_id),
        os.path.join(data_root, "train", trip_id),
        r"C:\Users\Vikas\Downloads\Student folder\Student folder\student_test\"" + trip_id,
        r"C:\Users\Vikas\Downloads\Student folder\Student folder\train\"" + trip_id,
    ]
    for path in candidates:
        clean_path = path.replace('"', '')
        if os.path.isdir(clean_path):
            return clean_path

    raise FileNotFoundError(f"Trip folder for trip ID '{trip_id}' not found in {data_root}.")


def load_trip_data(trip_id: str, data_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads raw trip data:
    - .track: [lon, lat, timestamp]
    - .nodes: [lon, lat] (0-based row index = node_id)
    - .arcs:  [from_node, to_node] (0-based row index = arc_id)
    - .route: [arc_id] (ground truth sequence)
    """
    folder = find_trip_folder(trip_id, data_root)

    track_file = os.path.join(folder, f"{trip_id}.track")
    nodes_file = os.path.join(folder, f"{trip_id}.nodes")
    arcs_file = os.path.join(folder, f"{trip_id}.arcs")
    route_file = os.path.join(folder, f"{trip_id}.route")

    if not os.path.exists(track_file):
        raise FileNotFoundError(f"Track file not found: {track_file}")
    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    if not os.path.exists(arcs_file):
        raise FileNotFoundError(f"Arcs file not found: {arcs_file}")

    track_df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes_df = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["lon", "lat"])
    nodes_df["node_id"] = np.arange(len(nodes_df))

    arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["from_node", "to_node"])
    arcs_df["arc_id"] = np.arange(len(arcs_df))

    route_arcs = []
    if os.path.exists(route_file):
        route_df = pd.read_csv(route_file, header=None, names=["arc_id"])
        route_arcs = route_df["arc_id"].tolist()

    return {
        "trip_id": trip_id,
        "track": track_df,
        "nodes": nodes_df,
        "arcs": arcs_df,
        "route": route_arcs
    }


def project_to_meters(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local metric projection (Equirectangular flat projection centered at lat0, lon0).
    Converts (lon, lat) to (x, y) in meters with high local fidelity.
    """
    lat0_rad = math.radians(lat0)
    x = np.radians(lon - lon0) * EARTH_RADIUS * math.cos(lat0_rad)
    y = np.radians(lat - lat0) * EARTH_RADIUS
    return x, y


def build_road_graph(arcs_df: pd.DataFrame) -> nx.Graph:
    """
    Constructs a NetworkX graph from arcs dataframe for connectivity checks.
    """
    G = nx.Graph()
    for _, row in arcs_df.iterrows():
        u = int(row["from_node"])
        v = int(row["to_node"])
        arc_id = int(row["arc_id"])
        G.add_edge(u, v, arc_id=arc_id)
    return G


def compute_route_continuity(predicted_arcs: List[int], arcs_df: pd.DataFrame) -> float:
    """
    Calculates Route Continuity %:
    Percentage of consecutive predicted arcs that connect directly in the road network graph
    (i.e. share at least one node in common).
    """
    if len(predicted_arcs) <= 1:
        return 100.0

    collapsed = [predicted_arcs[0]]
    for arc in predicted_arcs[1:]:
        if arc != collapsed[-1]:
            collapsed.append(arc)

    if len(collapsed) <= 1:
        return 100.0

    arc_nodes = {}
    for _, row in arcs_df.iterrows():
        arc_nodes[int(row["arc_id"])] = (int(row["from_node"]), int(row["to_node"]))

    connected_count = 0
    total_transitions = len(collapsed) - 1

    for i in range(total_transitions):
        arc_a = collapsed[i]
        arc_b = collapsed[i + 1]

        nodes_a = set(arc_nodes.get(arc_a, ()))
        nodes_b = set(arc_nodes.get(arc_b, ()))

        if nodes_a.intersection(nodes_b):
            connected_count += 1

    continuity = (connected_count / total_transitions) * 100.0
    return round(continuity, 2)


def compute_ground_truth_metrics(
    point_predicted_arcs: List[int],
    reference_route_arcs: List[int],
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes observation-level Top-1 Accuracy and Macro F1 score against server-side ground truth.
    """
    if not reference_route_arcs:
        return {
            "top1_accuracy": 0.0,
            "macro_f1": 0.0,
            "note": "No ground truth route supplied for this trip"
        }

    gt_arc_set = set(reference_route_arcs)

    correct_points = sum(1 for arc in point_predicted_arcs if arc in gt_arc_set)
    total_points = len(point_predicted_arcs)
    top1_accuracy = (correct_points / total_points) if total_points > 0 else 0.0

    pred_arc_set = set(point_predicted_arcs)
    intersection = pred_arc_set.intersection(gt_arc_set)
    precision = (len(intersection) / len(pred_arc_set)) if pred_arc_set else 0.0
    recall = (len(intersection) / len(gt_arc_set)) if gt_arc_set else 0.0

    if (precision + recall) > 0:
        macro_f1 = 2 * (precision * recall) / (precision + recall)
    else:
        macro_f1 = 0.0

    return {
        "top1_accuracy": round(top1_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }


def reconstruct_route_pipeline(
    trip_id: str,
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    model_path: Optional[str] = None
) -> Tuple[List[int], List[int]]:
    """
    Performs end-to-end trajectory reconstruction:
    1. Local metric projection
    2. Candidate generation via KD-Tree
    3. Geometric feature extraction
    4. Model scoring (or spatial probability fallback)
    5. HMM Viterbi dynamic programming decoding
    """
    num_points = len(track_df)
    if num_points == 0:
        return [], []

    lon0 = float(track_df["lon"].mean())
    lat0 = float(track_df["lat"].mean())

    gps_x, gps_y = project_to_meters(track_df["lon"].values, track_df["lat"].values, lon0, lat0)
    node_x, node_y = project_to_meters(nodes_df["lon"].values, nodes_df["lat"].values, lon0, lat0)

    from_nodes = arcs_df["from_node"].values
    to_nodes = arcs_df["to_node"].values
    arc_ids = arcs_df["arc_id"].values

    x1 = node_x[from_nodes]
    y1 = node_y[from_nodes]
    x2 = node_x[to_nodes]
    y2 = node_y[to_nodes]

    mid_x = (x1 + x2) * 0.5
    mid_y = (y1 + y2) * 0.5
    segment_lengths = np.hypot(x2 - x1, y2 - y1)
    segment_lengths_safe = np.where(segment_lengths == 0, 1e-6, segment_lengths)

    tree = cKDTree(np.column_stack([mid_x, mid_y]))

    trained_model = None
    if model_path and os.path.exists(model_path):
        try:
            trained_model = joblib.load(model_path)
            logger.info(f"Loaded trained champion model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")

    k_candidates = min(10, len(arcs_df))
    distances, candidate_indices = tree.query(np.column_stack([gps_x, gps_y]), k=k_candidates)

    sigma = 15.0
    timesteps_candidates = []

    for t in range(num_points):
        px = gps_x[t]
        py = gps_y[t]

        if t < num_points - 1:
            vx = gps_x[t + 1] - px
            vy = gps_y[t + 1] - py
        elif t > 0:
            vx = px - gps_x[t - 1]
            vy = py - gps_y[t - 1]
        else:
            vx, vy = 1.0, 0.0
        v_heading = math.degrees(math.atan2(vy, vx)) % 360.0

        step_cands = []
        c_idxs = candidate_indices[t] if k_candidates > 1 else [candidate_indices[t]]

        for idx in c_idxs:
            arc_id = arc_ids[idx]
            ax1, ay1 = x1[idx], y1[idx]
            ax2, ay2 = x2[idx], y2[idx]
            seg_len = segment_lengths_safe[idx]

            dx = ax2 - ax1
            dy = ay2 - ay1
            t_ratio = ((px - ax1) * dx + (py - ay1) * dy) / (seg_len * seg_len)
            t_clamped = max(0.0, min(1.0, float(t_ratio)))

            proj_x = ax1 + t_clamped * dx
            proj_y = ay1 + t_clamped * dy

            perp_dist = math.hypot(px - proj_x, py - proj_y)
            dist_to_start = math.hypot(px - ax1, py - ay1)
            dist_to_end = math.hypot(px - ax2, py - ay2)

            seg_heading = math.degrees(math.atan2(dy, dx)) % 360.0
            heading_diff = abs(v_heading - seg_heading)
            heading_error = min(heading_diff, 360.0 - heading_diff)

            spatial_prob = math.exp(-(perp_dist ** 2) / (2.0 * sigma * sigma))

            model_prob = 1.0
            if trained_model is not None:
                try:
                    feat = np.array([[perp_dist, t_clamped, heading_error, seg_len, dist_to_start, dist_to_end]])
                    if hasattr(trained_model, "predict_proba"):
                        model_prob = float(trained_model.predict_proba(feat)[0][1])
                    else:
                        model_prob = float(trained_model.predict(feat)[0])
                except Exception:
                    model_prob = 1.0

            emission = model_prob * spatial_prob + 1e-12

            step_cands.append({
                "arc_id": int(arc_id),
                "from_node": int(from_nodes[idx]),
                "to_node": int(to_nodes[idx]),
                "emission": emission,
                "perp_dist": perp_dist,
                "t_ratio": t_clamped
            })

        timesteps_candidates.append(step_cands)

    point_predictions = []
    V = [{} for _ in range(num_points)]
    backpointer = [{} for _ in range(num_points)]

    for i, cand in enumerate(timesteps_candidates[0]):
        V[0][i] = math.log(max(cand["emission"], 1e-15))

    for t in range(1, num_points):
        prev_cands = timesteps_candidates[t - 1]
        curr_cands = timesteps_candidates[t]

        for j, curr_c in enumerate(curr_cands):
            best_prev_score = -float("inf")
            best_prev_idx = 0
            curr_nodes = {curr_c["from_node"], curr_c["to_node"]}

            for i, prev_c in enumerate(prev_cands):
                prev_nodes = {prev_c["from_node"], prev_c["to_node"]}

                if curr_c["arc_id"] == prev_c["arc_id"]:
                    trans_prob = 0.90
                elif curr_nodes.intersection(prev_nodes):
                    trans_prob = 0.70
                else:
                    trans_prob = 0.05

                score = V[t - 1][i] + math.log(trans_prob) + math.log(max(curr_c["emission"], 1e-15))

                if score > best_prev_score:
                    best_prev_score = score
                    best_prev_idx = i

            V[t][j] = best_prev_score
            backpointer[t][j] = best_prev_idx

    best_last_idx = max(V[num_points - 1], key=V[num_points - 1].get)
    best_path_indices = [best_last_idx]

    for t in range(num_points - 1, 0, -1):
        prev_idx = backpointer[t][best_path_indices[-1]]
        best_path_indices.append(prev_idx)

    best_path_indices.reverse()

    for t in range(num_points):
        chosen_idx = best_path_indices[t]
        point_predictions.append(timesteps_candidates[t][chosen_idx]["arc_id"])

    sequence_predictions = []
    for arc in point_predictions:
        if not sequence_predictions or arc != sequence_predictions[-1]:
            sequence_predictions.append(arc)

    return point_predictions, sequence_predictions


def get_route_coordinates(arc_ids: List[int], arcs_df: pd.DataFrame, nodes_df: pd.DataFrame) -> List[List[float]]:
    """
    Resolves a list of arc_ids to a list of [lat, lon] coordinates for Folium polyline rendering.
    """
    coords = []
    node_coords = {}
    for _, row in nodes_df.iterrows():
        node_coords[int(row["node_id"])] = [float(row["lat"]), float(row["lon"])]

    arc_endpoints = {}
    for _, row in arcs_df.iterrows():
        arc_endpoints[int(row["arc_id"])] = (int(row["from_node"]), int(row["to_node"]))

    for arc_id in arc_ids:
        if arc_id in arc_endpoints:
            u, v = arc_endpoints[arc_id]
            if u in node_coords and v in node_coords:
                if not coords or coords[-1] != node_coords[u]:
                    coords.append(node_coords[u])
                coords.append(node_coords[v])

    return coords