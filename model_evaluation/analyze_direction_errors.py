"""
Diagnoses WHY per-trip Top-1 varies while route continuity stays high.

For every trip it reports strict Top-1 (predicted arc is on the reference route), Top-1 when
direction is ignored (the arc or its opposite-direction twin is on the route), and the share
of wrong predictions that are exactly the reverse-direction twin of a route arc. A large
gap between strict and direction-agnostic Top-1 means the decoder found the right street but
assigned its opposite-direction arc -- a connected, drivable-looking path that continuity cannot flag.
Run lengths show whether those errors are sustained wrong-way stretches or point-by-point flips.

This is a post-hoc diagnostic only: nothing here feeds training, tuning or model selection.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_viterbi.viterbi import prepare_trip_context, decode_viterbi


def analyse_trip(base_dir, trip_id, model_path):
    ctx = prepare_trip_context(trip_id, base_dir, model_path)
    r = decode_viterbi(ctx)
    route, ep = ctx["route_arcs"], ctx["arc_endpoints"]
    pred = r["predicted_arc_ids"]
    reverse_twins = {(v, u) for u, v in (ep[a] for a in route)}
    route_undirected = {frozenset(ep[a]) for a in route}
    wrong = [a for a in pred if a not in route]
    wrong_flags = np.array([a not in route for a in pred])
    # Share of observations for which none of the 10 candidate arcs is on the reference route:
    # no decoder can be right there, so this is a ceiling set by candidate generation / GPS noise.
    no_true_candidate = np.mean([not any(c[0] in route for c in cands) for cands in ctx["candidates_per_t"]])
    first = min(60, len(pred))
    runs, cur = [], 0                      # lengths of maximal runs of consecutive wrong points
    for w in wrong_flags:
        if w:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    runs = np.array(runs) if runs else np.array([0])
    return {
        "partition": os.path.basename(base_dir),
        "trip_id": trip_id,
        "n_points": len(pred),
        "strict_top1": round(r["pointwise_accuracy"], 4),
        "direction_agnostic_top1": round(float(np.mean([frozenset(ep[a]) in route_undirected for a in pred])), 4),
        "wrong_points": len(wrong),
        "wrong_that_are_reverse_twin": round(sum(ep[a] in reverse_twins for a in wrong) / max(1, len(wrong)), 4),
        "no_true_candidate_share": round(float(no_true_candidate), 4),
        "wrong_rate_first_60_points": round(float(wrong_flags[:first].mean()), 4),
        "wrong_rate_after_first_60": round(float(wrong_flags[first:].mean()), 4) if len(pred) > first else float("nan"),
        "wrong_runs": int((runs > 0).sum()),
        "median_wrong_run_len": float(np.median(runs[runs > 0])) if (runs > 0).any() else 0.0,
        "longest_wrong_run": int(runs.max()),
        "share_of_wrong_in_runs_ge5": round(float(runs[runs >= 5].sum() / max(1, runs.sum())), 4),
        "predicted_arc_set_precision": round(len(set(pred) & route) / max(1, len(set(pred))), 4),
        "reference_arc_recall": round(len(set(pred) & route) / max(1, len(route)), 4),
    }


if __name__ == "__main__":
    model = "results/best_tuned_model.joblib"
    rows = []
    for part in ("student_test", "evaluator_test"):
        base = os.path.join("Student folder", part)
        if not os.path.isdir(base):
            continue
        for t in sorted(os.listdir(base)):
            rows.append(analyse_trip(base, t, model))
            print(rows[-1])
    pd.DataFrame(rows).to_csv("results/direction_error_analysis.csv", index=False)
