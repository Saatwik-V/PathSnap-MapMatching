"""
HMM + Viterbi Sequential Route Reconstruction - PathSnap
Converts independent per-GPS-point candidate scores into a single
road-network-consistent arc sequence.

Hidden states  : candidate arc_ids at each GPS timestep.
Emission       : P(O_t | a_i) propto P_model(y=1 | features_i) * exp(-perp_dist_i^2 / (2*sigma^2))
Transition     : P(a_{t+1} | a_t) propto exp(-|d_graph(a_t, a_{t+1}) - d_greatcircle(p_t, p_{t+1})| / beta)
                 d_graph = 0 when arcs connect directly; shortest-path length via NetworkX otherwise;
                 floored to a minimum probability of 1e-6 when no path exists.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.candidate_gen import process_trip_candidates

FEATURE_COLS = [
    "perp_dist",
    "proj_ratio",
    "heading_error",
    "arc_length",
    "dist_to_start",
    "dist_to_end",
]

MIN_PROB = 1e-6
EARTH_RADIUS_M = 6371000.0


def _haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_trip_graph(trip_id: str, base_dir: str):
    """Builds a directed road-network graph for one trip from .nodes / .arcs."""
    trip_path = os.path.join(base_dir, trip_id)
    nodes = pd.read_csv(os.path.join(trip_path, f"{trip_id}.nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(os.path.join(trip_path, f"{trip_id}.arcs"), sep=r"\s+", header=None, names=["u", "v"])

    lon = nodes["lon"].values
    lat = nodes["lat"].values

    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))

    u_idx = arcs["u"].values
    v_idx = arcs["v"].values
    lengths = _haversine_m(lon[u_idx], lat[u_idx], lon[v_idx], lat[v_idx])

    for arc_id, (u, v, length) in enumerate(zip(u_idx, v_idx, lengths)):
        G.add_edge(int(u), int(v), weight=float(length), arc_id=int(arc_id))

    arc_endpoints = {int(aid): (int(u), int(v)) for aid, (u, v) in enumerate(zip(u_idx, v_idx))}
    arc_lengths = {int(aid): float(l) for aid, l in enumerate(lengths)}
    return G, arc_endpoints, arc_lengths, nodes, arcs


GRAPH_SEARCH_CUTOFF_M = 400.0  # beyond this, two arcs are treated as effectively disconnected


class GraphDistanceIndex:
    """
    Caches one bounded (cutoff-limited) single-source Dijkstra search per distinct
    'end node' of a candidate arc, instead of running a fresh shortest-path query for
    every (arc_i, arc_j) pair on every Viterbi call. This is what makes it practical
    to grid-search / re-decode many (sigma, beta) values over the same trip: the
    expensive graph search runs exactly once per trip, independent of those parameters.
    """

    def __init__(self, G: nx.DiGraph, cutoff: float = GRAPH_SEARCH_CUTOFF_M):
        self.G = G
        self.cutoff = cutoff
        self._cache = {}

    def _dist_map_from(self, source_node: int) -> dict:
        if source_node not in self._cache:
            try:
                self._cache[source_node] = nx.single_source_dijkstra_path_length(
                    self.G, source_node, cutoff=self.cutoff, weight="weight"
                )
            except nx.NodeNotFound:
                self._cache[source_node] = {}
        return self._cache[source_node]

    def distance(self, arc_i: int, arc_j: int, arc_endpoints: dict) -> float:
        """Shortest-path distance (meters) travelling from the end of arc_i to the start of arc_j."""
        if arc_i == arc_j:
            return 0.0
        _, v_i = arc_endpoints[arc_i]
        u_j, _ = arc_endpoints[arc_j]
        return self._dist_map_from(v_i).get(u_j, np.inf)


def load_classifier(model_path: str):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle.get("scaler", None), bundle.get("model_name", "model")


def score_candidates(candidates_df: pd.DataFrame, model, scaler):
    X = candidates_df[FEATURE_COLS].values
    if scaler is not None:
        X = scaler.transform(X)
    probs = model.predict_proba(X)[:, 1]
    out = candidates_df.copy()
    out["prob"] = probs
    return out


def route_continuity_pct(arc_sequence: list, arc_endpoints: dict) -> float:
    """Percentage of consecutive predicted arcs that connect directly (to_node == from_node)."""
    if len(arc_sequence) < 2:
        return 100.0
    connected = 0
    total = 0
    for a, b in zip(arc_sequence[:-1], arc_sequence[1:]):
        if a == b:
            connected += 1
            total += 1
            continue
        _, v_a = arc_endpoints[a]
        u_b, _ = arc_endpoints[b]
        total += 1
        if v_a == u_b:
            connected += 1
    return 100.0 * connected / total if total else 100.0


def prepare_trip_context(
    trip_id: str,
    base_dir: str = "Student folder/train",
    model_path: str = "results/best_tuned_model.joblib",
    k_candidates: int = 10,
) -> dict:
    """
    Runs candidate generation + classifier scoring + graph construction for one trip.
    This is the expensive, (sigma, beta)-independent part of the pipeline, split out so
    decode_viterbi() can be called repeatedly (e.g. during a parameter sweep) without
    re-running candidate generation or re-scoring the classifier each time.
    """
    model, scaler, model_name = load_classifier(model_path)

    candidates_df = process_trip_candidates(trip_id, base_dir=base_dir, k_candidates=k_candidates)
    if candidates_df.empty:
        return {"trip_id": trip_id, "error": "no candidates generated"}

    scored_df = score_candidates(candidates_df, model, scaler)
    G, arc_endpoints, arc_lengths, nodes, arcs = load_trip_graph(trip_id, base_dir)
    dist_index = GraphDistanceIndex(G)

    track_path = os.path.join(base_dir, trip_id, f"{trip_id}.track")
    track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "time"])

    route_path = os.path.join(base_dir, trip_id, f"{trip_id}.route")
    route_arcs = set()
    if os.path.exists(route_path):
        route_arcs = set(pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].values)

    obs_groups = scored_df.sort_values(["obs_idx", "prob"], ascending=[True, False]).groupby("obs_idx")
    timesteps = sorted(scored_df["obs_idx"].unique())

    candidates_per_t = []
    for t in timesteps:
        g = obs_groups.get_group(t)
        cand_list = list(zip(g["candidate_arc_id"].values, g["prob"].values, g["perp_dist"].values))
        candidates_per_t.append(cand_list)

    # Great-circle distance between consecutive raw GPS points.
    gc_dist = np.zeros(len(timesteps))
    for i in range(1, len(timesteps)):
        p_prev = track.iloc[timesteps[i - 1]]
        p_curr = track.iloc[timesteps[i]]
        gc_dist[i] = _haversine_m(p_prev["lon"], p_prev["lat"], p_curr["lon"], p_curr["lat"])

    return {
        "trip_id": trip_id,
        "candidates_per_t": candidates_per_t,
        "timesteps": timesteps,
        "gc_dist": gc_dist,
        "arc_endpoints": arc_endpoints,
        "dist_index": dist_index,
        "route_arcs": route_arcs,
        "model_name": model_name,
    }


def decode_viterbi(ctx: dict, sigma: float = 10.0, beta: float = 100.0) -> dict:
    """
    Cheap step: given a precomputed trip context (see prepare_trip_context), decodes the
    HMM with the given (sigma, beta) and returns the predicted sequence plus metrics,
    compared against the classifier-only (no-HMM) baseline.
    """
    candidates_per_t = ctx["candidates_per_t"]
    arc_endpoints = ctx["arc_endpoints"]
    dist_index: GraphDistanceIndex = ctx["dist_index"]
    gc_dist = ctx["gc_dist"]
    route_arcs = ctx["route_arcs"]
    T = len(candidates_per_t)

    # ---- Classifier-only baseline: pick arg-max probability candidate at each timestep ----
    classifier_only_seq = [max(cands, key=lambda c: c[1])[0] for cands in candidates_per_t]

    def emission(prob, perp_dist):
        spatial = np.exp(-(perp_dist ** 2) / (2 * sigma ** 2))
        return max(prob * spatial, MIN_PROB)

    def transition(arc_i, arc_j, d_gc):
        d_graph = dist_index.distance(arc_i, arc_j, arc_endpoints)
        if not np.isfinite(d_graph):
            return MIN_PROB
        return max(np.exp(-abs(d_graph - d_gc) / beta), MIN_PROB)

    # ---- Viterbi recurrence in log-space ----
    V = [dict() for _ in range(T)]
    backptr = [dict() for _ in range(T)]

    for arc_id, prob, perp_dist in candidates_per_t[0]:
        V[0][arc_id] = np.log(emission(prob, perp_dist))
        backptr[0][arc_id] = None

    for t in range(1, T):
        d_gc = gc_dist[t]
        for arc_j, prob_j, perp_j in candidates_per_t[t]:
            best_score, best_prev = -np.inf, None
            log_em = np.log(emission(prob_j, perp_j))
            for arc_i, v_i in V[t - 1].items():
                log_tr = np.log(transition(arc_i, arc_j, d_gc))
                score = v_i + log_tr
                if score > best_score:
                    best_score, best_prev = score, arc_i
            V[t][arc_j] = best_score + log_em
            backptr[t][arc_j] = best_prev

    # ---- Backtrack ----
    last_arc = max(V[T - 1], key=V[T - 1].get)
    hmm_seq = [last_arc]
    for t in range(T - 1, 0, -1):
        last_arc = backptr[t][hmm_seq[-1]]
        hmm_seq.append(last_arc)
    hmm_seq.reverse()

    # ---- Metrics (weak reference label: predicted arc is a member of the route's arc set,
    #      consistent with the label_y convention used throughout candidate_gen / training) ----
    def pointwise_accuracy(seq):
        if not route_arcs:
            return np.nan
        return float(np.mean([1 if a in route_arcs else 0 for a in seq]))

    return {
        "trip_id": ctx["trip_id"],
        "n_timesteps": T,
        "sigma": sigma,
        "beta": beta,
        "predicted_arc_ids": hmm_seq,
        "classifier_only_arc_ids": classifier_only_seq,
        "pointwise_accuracy": pointwise_accuracy(hmm_seq),
        "classifier_only_pointwise_accuracy": pointwise_accuracy(classifier_only_seq),
        "route_continuity_pct": route_continuity_pct(hmm_seq, arc_endpoints),
        "classifier_only_route_continuity_pct": route_continuity_pct(classifier_only_seq, arc_endpoints),
        "model_name": ctx["model_name"],
    }


def reconstruct_route(
    trip_id: str,
    base_dir: str = "Student folder/train",
    model_path: str = "results/best_tuned_model.joblib",
    sigma: float = 10.0,
    beta: float = 100.0,
    k_candidates: int = 10,
) -> dict:
    """
    Convenience one-shot wrapper: prepares the trip context and decodes it once.
    This is the function the FastAPI layer should call per request.
    """
    ctx = prepare_trip_context(trip_id, base_dir=base_dir, model_path=model_path, k_candidates=k_candidates)
    if "error" in ctx:
        return ctx
    return decode_viterbi(ctx, sigma=sigma, beta=beta)


def evaluate_trips(trip_ids: list, base_dir: str, model_path: str) -> pd.DataFrame:
    rows = []
    for tid in trip_ids:
        r = reconstruct_route(tid, base_dir=base_dir, model_path=model_path)
        if "error" in r:
            continue
        rows.append({
            "trip_id": r["trip_id"],
            "n_timesteps": r["n_timesteps"],
            "classifier_only_pointwise_acc": r["classifier_only_pointwise_accuracy"],
            "hmm_pointwise_acc": r["pointwise_accuracy"],
            "classifier_only_continuity_pct": r["classifier_only_route_continuity_pct"],
            "hmm_continuity_pct": r["route_continuity_pct"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="Student folder/train")
    parser.add_argument("--model_path", default="results/best_tuned_model.joblib")
    parser.add_argument("--n_trips", type=int, default=10)
    parser.add_argument("--output", default="results/hmm_vs_classifier.csv")
    args = parser.parse_args()

    trips = sorted(os.listdir(args.base_dir))[: args.n_trips]
    df = evaluate_trips(trips, args.base_dir, args.model_path)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nMean classifier-only continuity: {df['classifier_only_continuity_pct'].mean():.2f}%")
    print(f"Mean HMM+Viterbi continuity:      {df['hmm_continuity_pct'].mean():.2f}%")
    print(f"\nSaved per-trip comparison to: {args.output}")
