"""
Production Artifact Batch Generator - PathSnap
Runs the tuned model + HMM/Viterbi decoder on all 10 student-test trips and exports
standalone interactive Folium HTML maps, so evaluators can open them in any browser
without running Python.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_viterbi.viterbi import reconstruct_route
from visualisations.folium_map import build_trip_map


def generate_trip_map(trip_id: str, base_dir: str, model_path: str, output_dir: str) -> str:
    trip_path = os.path.join(base_dir, trip_id)
    track_df = pd.read_csv(os.path.join(trip_path, f"{trip_id}.track"), sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes_df = pd.read_csv(os.path.join(trip_path, f"{trip_id}.nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
    arcs_df = pd.read_csv(os.path.join(trip_path, f"{trip_id}.arcs"), sep=r"\s+", header=None, names=["u", "v"])

    route_path = os.path.join(trip_path, f"{trip_id}.route")
    route_arcs = []
    if os.path.exists(route_path):
        route_arcs = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()

    result = reconstruct_route(trip_id, base_dir=base_dir, model_path=model_path)
    predicted_arcs = result.get("repaired_route_arc_ids", result.get("predicted_arc_ids", []))

    fmap = build_trip_map(
        track_df=track_df,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        route_arcs=route_arcs,
        predicted_arcs=predicted_arcs,
        trip_id=trip_id,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"trip_{trip_id}_map.html")
    fmap.save(out_path)

    continuity = result.get("route_continuity_pct", float("nan"))
    accuracy = result.get("pointwise_accuracy", float("nan"))
    print(f"  {trip_id}: continuity={continuity:.2f}% accuracy={accuracy:.3f} -> {out_path}")
    return out_path


def generate_all(
    base_dir: str = "Student folder/student_test",
    model_path: str = "results/best_tuned_model.joblib",
    output_dir: str = "results/maps",
) -> list:
    trips = sorted(os.listdir(base_dir))
    print(f"Generating production Folium maps for {len(trips)} student-test trips...")
    paths = []
    for trip_id in trips:
        try:
            paths.append(generate_trip_map(trip_id, base_dir, model_path, output_dir))
        except Exception as e:
            print(f"  {trip_id}: FAILED ({e})")
    print(f"\nSaved {len(paths)} maps to: {output_dir}")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="Student folder/student_test")
    parser.add_argument("--model_path", default="results/best_tuned_model.joblib")
    parser.add_argument("--output_dir", default="results/maps")
    args = parser.parse_args()
    generate_all(args.base_dir, args.model_path, args.output_dir)
