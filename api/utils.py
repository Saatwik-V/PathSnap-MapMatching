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
import sys
import glob
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger("pathsnap_api")

# Earth radius in meters
EARTH_RADIUS = 6371000.0


def find_dataset_root() -> str:
    """
    Locates the dataset root directory containing 'train', 'student_test', and/or
    'evaluator_test' trip folders.
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
            if any(os.path.isdir(os.path.join(path, p)) for p in ("evaluator_test", "student_test", "train")):
                return path
            if "student_test" in path or "evaluator_test" in path:
                return path

    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def list_available_trips(data_root: Optional[str] = None) -> List[str]:
    """
    Returns sorted list of 10-digit test trip IDs.

    'evaluator_test' is checked first: per the official Dataset Exploration / Assessment
    policy docs, it is the final-evaluation-only partition and -- once present on Laptop 1
    for the live demo -- is what the API and UI must actually serve. 'student_test' (our
    own development-stage partition, used throughout this repo's offline experiments) is
    the fallback so local dev/testing keeps working when 'evaluator_test' isn't deployed.
    """
    if data_root is None:
        data_root = find_dataset_root()

    search_dirs = [
        os.path.join(data_root, "evaluator_test"),
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
        # Last-resort placeholder if no trip directories are found at all on disk --
        # the evaluator_test IDs, since that is the partition the live demo serves from.
        trips = ["00000015", "00000017", "00000023", "00000030", "00000049",
                 "00000059", "00000067", "00000073", "00000087", "00000094"]
    return trips


def find_trip_folder(trip_id: str, data_root: Optional[str] = None) -> str:
    """
    Finds the directory containing the trip files for a given trip_id.
    'evaluator_test' is checked first for the same reason as in list_available_trips.
    """
    if data_root is None:
        data_root = find_dataset_root()

    candidates = [
        os.path.join(data_root, "evaluator_test", trip_id),
        os.path.join(data_root, "student_test", trip_id),
        os.path.join(data_root, trip_id),
        os.path.join(data_root, "train", trip_id),
        r"C:\Users\Vikas\Downloads\Student folder\Student folder\evaluator_test\"" + trip_id,
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
    Delegates to the canonical hmm_viterbi.viterbi pipeline (12-feature candidate
    generation via feature_engineering.extract_features + classifier scoring +
    HMM/Viterbi decoding), so the API serves predictions from the exact same code
    path used for training and offline evaluation.

    track_df / nodes_df / arcs_df are accepted for backward-compatible call signature
    (the caller already loaded them for other purposes) but are not reused here --
    hmm_viterbi.viterbi re-reads the trip's raw files itself, since it also needs the
    .route file and full road-network graph, which load_trip_data's caller may not
    always have attached.
    """
    from hmm_viterbi.viterbi import reconstruct_route as _reconstruct_route

    trip_folder = find_trip_folder(trip_id)
    base_dir = os.path.dirname(trip_folder)

    resolved_model_path = model_path
    if not resolved_model_path or not os.path.exists(resolved_model_path):
        tuned_path = os.path.join(REPO_ROOT, "results", "best_tuned_model.joblib")
        resolved_model_path = tuned_path if os.path.exists(tuned_path) else model_path

    result = _reconstruct_route(trip_id, base_dir=base_dir, model_path=resolved_model_path)
    if "error" in result:
        logger.warning(f"Route reconstruction failed for trip {trip_id}: {result['error']}")
        return [], []

    # Cast to native Python int: FastAPI's jsonable_encoder cannot serialize numpy.int64
    # inside a plain list, and the underlying candidate/arc ids are numpy ints throughout
    # the hmm_viterbi / pandas pipeline.
    point_predictions = [int(a) for a in result["predicted_arc_ids"]]
    # Connected route = per-timestep decode with deterministic gap-repair bridge arcs
    # spliced in (hmm_viterbi.repair_route_gaps); used for continuity and map drawing.
    connected = [int(a) for a in result.get("repaired_route_arc_ids", point_predictions)]
    sequence_predictions = []
    for arc in connected:
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