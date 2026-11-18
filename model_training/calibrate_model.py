"""
Probability calibration of the Optuna-tuned HistGradientBoosting champion, with a FAIR
three-way comparison (untuned / tuned / tuned+calibrated) all measured on the same
student-test partition (evaluator_test is never touched).

Why calibrate: the HMM multiplies classifier probabilities into its emission term, so
over/under-confident probabilities directly distort Viterbi decoding, not just log loss.
Isotonic regression is fit on held-out trips only: CalibratedClassifierCV is given
GroupKFold-by-trip splits, so no trip is ever used to both fit the base model and
calibrate it (same leakage rule as the CV benchmark).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, f1_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.extract_features import FEATURE_COLS
from hmm_viterbi.viterbi import evaluate_trips


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def top1(test_df, p):
    d = test_df.assign(prob=p).sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False])
    return float(d.groupby(["trip_id", "obs_idx"]).head(1)["label_y"].mean())


def main(train_csv="results/train_features.csv", test_csv="results/test_features.csv",
         test_dir="Student folder/student_test"):
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    X, y, g = tr[FEATURE_COLS].values, tr["label_y"].values, tr["trip_id"].values
    Xt, yt = te[FEATURE_COLS].values, te["label_y"].values

    best = json.load(open("results/optuna_best_params.json"))["best_params"]
    untuned = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, l2_regularization=0.1, random_state=42).fit(X, y)
    tuned = joblib.load("results/best_tuned_model.joblib")["model"]

    splits = list(GroupKFold(n_splits=5).split(X, y, groups=g))
    base = HistGradientBoostingClassifier(random_state=42, **best)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=splits).fit(X, y)
    joblib.dump({"model": calibrated, "scaler": None, "model_name": "histgb_tuned_isotonic", "params": best},
                "results/best_calibrated_model.joblib")

    trips = sorted(te["trip_id"].astype(str).str.zfill(8).unique())
    rows = []
    for name, m, path in [("HistGB untuned", untuned, "results/_tmp_untuned.joblib"),
                          ("HistGB tuned (Optuna)", tuned, "results/best_tuned_model.joblib"),
                          ("HistGB tuned + isotonic calibration", calibrated, "results/best_calibrated_model.joblib")]:
        p = m.predict_proba(Xt)[:, 1]
        if path.endswith("_tmp_untuned.joblib"):
            joblib.dump({"model": m, "scaler": None, "model_name": name}, path)
        hmm = evaluate_trips(trips, test_dir, path)
        if path.endswith("_tmp_untuned.joblib"):
            os.remove(path)
        rows.append({
            "Variant": name,
            "Student-Test Log Loss": round(log_loss(yt, p), 4),
            "Student-Test Brier": round(brier_score_loss(yt, p), 4),
            "Student-Test ECE": round(ece(yt, p), 4),
            "Student-Test Macro F1": round(f1_score(yt, (p >= 0.5).astype(int), average="macro"), 4),
            "Student-Test Top-1 Accuracy": f"{top1(te, p) * 100:.2f}%",
            "HMM Pointwise Acc": f"{hmm['hmm_pointwise_acc'].mean() * 100:.2f}%",
            "HMM Route Continuity": f"{hmm['hmm_continuity_pct'].mean():.2f}%",
        })
        print(rows[-1])
    pd.DataFrame(rows).to_csv("results/calibration_comparison.csv", index=False)


if __name__ == "__main__":
    main()
