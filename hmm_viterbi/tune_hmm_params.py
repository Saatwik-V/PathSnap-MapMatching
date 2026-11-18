"""
HMM (sigma, beta) sweep on the student-test partition ONLY (never evaluator_test).
Context prep (candidate gen + scoring + graph) runs once per trip; each (sigma, beta)
decode reuses it. Selection rule: maximise mean pointwise accuracy first (the quantity
the classifier+HMM is actually trying to get right), break ties/near-ties by repaired
continuity, so continuity is never bought with per-point accuracy.
"""
import os
import sys
import itertools
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_viterbi.viterbi import prepare_trip_context, decode_viterbi

SIGMAS = [5.0, 10.0, 15.0, 25.0]
BETAS = [25.0, 50.0, 100.0, 200.0, 400.0]


def sweep(base_dir="Student folder/student_test", model_path="results/best_tuned_model.joblib",
          output="results/hmm_param_sweep.csv"):
    trips = sorted(os.listdir(base_dir))
    ctxs = [prepare_trip_context(t, base_dir, model_path) for t in trips]
    ctxs = [c for c in ctxs if "error" not in c]
    rows = []
    for sigma, beta in itertools.product(SIGMAS, BETAS):
        res = [decode_viterbi(c, sigma=sigma, beta=beta) for c in ctxs]
        rows.append({
            "sigma": sigma, "beta": beta,
            "mean_pointwise_acc": sum(r["pointwise_accuracy"] for r in res) / len(res),
            "mean_pre_repair_continuity": sum(r["pre_repair_route_continuity_pct"] for r in res) / len(res),
            "mean_repaired_continuity": sum(r["route_continuity_pct"] for r in res) / len(res),
            "gaps_repaired": sum(r["n_gaps_repaired"] for r in res),
            "gaps_unrepairable": sum(r["n_gaps_unrepairable"] for r in res),
        })
        print(rows[-1])
    df = pd.DataFrame(rows).sort_values(["mean_pointwise_acc", "mean_repaired_continuity"], ascending=False)
    df.to_csv(output, index=False)
    print("\nTop 5:\n", df.head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    sweep()
