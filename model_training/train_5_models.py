"""
5-Model Family Benchmark Suite & Comparative Study - PathSnap
Mandatory Hackathon Deliverable | Strict GroupKFold Leakage Prevention
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE_COLS = [
    "perp_dist",
    "proj_ratio",
    "heading_error",
    "heading_cos",
    "arc_length",
    "dist_to_start",
    "dist_to_end",
    "speed_mps",
    "delta_t",
    "candidate_rank",
    "dist_diff_endpoints",
    "traversal_time_est",
]

def run_5_model_study(
    csv_path: str = "results/train_features.csv",
    output_dir: str = "results",
    plot_dir: str = "results/plots"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("===============================================================")
    print("        PATHSNAP: 5-MODEL ALGORITHM BENCHMARK STUDY            ")
    print("===============================================================")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Candidate dataset {csv_path} not found. Run candidate_gen.py first.")

    df = pd.read_csv(csv_path)
    n_trips = df["trip_id"].nunique()
    print(f"Loaded {len(df):,} candidate rows across {n_trips} unique trips.")
    print(f"Target distribution: {df['label_y'].value_counts(normalize=True).round(3).to_dict()}\n")

    X = df[FEATURE_COLS].values
    y = df["label_y"].values
    groups = df["trip_id"].values

    # Define 5 genuine distinct algorithm families
    model_definitions = [
        {
            "name": "Logistic Regression",
            "family": "Linear & Probabilistic",
            "config": "max_iter=1000, class_weight='balanced'",
            "scaler": True,
            "clf": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        },
        {
            "name": "Distance-Weighted k-NN",
            "family": "Instance-based",
            "config": "k=5, weights='distance', algorithm='kd_tree'",
            "scaler": True,
            "clf": KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="kd_tree")
        },
        {
            "name": "Random Forest",
            "family": "Ensemble (Bagging)",
            "config": "n_estimators=100, max_depth=12, n_jobs=-1",
            "scaler": False,
            "clf": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        },
        {
            "name": "Histogram Gradient Boosting",
            "family": "Boosting",
            "config": "max_iter=100, learning_rate=0.08, l2_regularization=0.1",
            "scaler": False,
            "clf": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, l2_regularization=0.1, random_state=42)
        },
        {
            "name": "Multi-Layer Perceptron (MLP)",
            "family": "Neural Network",
            "config": "hidden_layers=(64, 32), early_stopping=True, max_iter=150",
            "scaler": True,
            "clf": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, early_stopping=True, random_state=42)
        }
    ]

    n_splits = min(5, n_trips) if n_trips >= 3 else 2
    print(f"Enforcing GroupKFold with {n_splits} splits (grouped by trip_id)...\n")
    gkf = GroupKFold(n_splits=n_splits)

    results_table = []
    trained_models = {}

    for m_info in model_definitions:
        m_name = m_info["name"]
        m_family = m_info["family"]
        m_config = m_info["config"]
        use_scaler = m_info["scaler"]
        clf = m_info["clf"]

        print(f"--> Training Family: {m_family} | Model: {m_name}")
        fold_f1s = []
        fold_top1s = []
        fold_losses = []
        train_times = []
        inf_times = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]
            val_df = df.iloc[val_idx].copy()

            if use_scaler:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_va = sc.transform(X_va)

            t0 = time.time()
            clf.fit(X_tr, y_tr)
            t_train = time.time() - t0
            train_times.append(t_train)

            t0 = time.time()
            probs = clf.predict_proba(X_va)[:, 1]
            t_inf = time.time() - t0
            inf_times.append((t_inf / len(X_va)) * 1000)  # ms per obs

            preds = (probs >= 0.5).astype(int)
            macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            fold_f1s.append(macro_f1)

            # Observation-level Top-1 Accuracy: candidate with highest prob per GPS observation
            val_df["prob"] = probs
            top1_picks = val_df.sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False])
            top1_picks = top1_picks.groupby(["trip_id", "obs_idx"]).head(1)
            obs_acc = top1_picks["label_y"].mean()
            fold_top1s.append(obs_acc)

            try:
                ll = log_loss(y_val, probs)
                fold_losses.append(ll)
            except Exception:
                fold_losses.append(np.nan)

        avg_f1 = float(np.mean(fold_f1s))
        avg_top1 = float(np.mean(fold_top1s))
        avg_ll = float(np.nanmean(fold_losses))
        avg_tr_time = float(np.mean(train_times))
        avg_inf_ms = float(np.mean(inf_times))

        print(f"    Macro F1: {avg_f1:.4f} | Top-1 Acc: {avg_top1*100:.2f}% | Log Loss: {avg_ll:.4f} | Inf Latency: {avg_inf_ms:.3f}ms")

        results_table.append({
            "Model": m_name,
            "Algorithm Family": m_family,
            "Key Configuration": m_config,
            "CV Macro F1": round(avg_f1, 4),
            "Top-1 Accuracy": f"{round(avg_top1 * 100, 2)}%",
            "Log Loss": round(avg_ll, 4),
            "Train Time (s)": round(avg_tr_time, 2),
            "Inference Latency (ms)": round(avg_inf_ms, 3)
        })

        # Fit on entire dataset to store final model
        if use_scaler:
            sc_full = StandardScaler()
            X_full = sc_full.fit_transform(X)
            clf.fit(X_full, y)
            trained_models[m_name] = (clf, sc_full, avg_f1)
        else:
            clf.fit(X, y)
            trained_models[m_name] = (clf, None, avg_f1)

    # ---- Student-test Top-1 Accuracy / Macro F1 + Route Continuity, per model ----
    # The Deliverables report table requires these per model, not just CV metrics.
    # Route continuity needs the full candidate-gen + HMM/Viterbi pipeline, so each
    # fitted model is temporarily dumped to disk and passed through hmm_viterbi.viterbi
    # (reused as-is, rather than duplicating the decoder here).
    student_test_csv = os.path.join(output_dir, "test_features.csv")
    hmm_base_dir = "Student folder/student_test"
    tmp_model_path = os.path.join(output_dir, "_tmp_model_eval.joblib")

    continuity_by_model = {}
    student_f1_by_model = {}
    student_top1_by_model = {}

    if os.path.exists(student_test_csv) and os.path.isdir(hmm_base_dir):
        from hmm_viterbi.viterbi import evaluate_trips
        test_df = pd.read_csv(student_test_csv)
        Xt = test_df[FEATURE_COLS].values
        yt = test_df["label_y"].values
        trips = sorted(os.listdir(hmm_base_dir))

        for m_name, (clf, scaler, _) in trained_models.items():
            Xt_in = scaler.transform(Xt) if scaler is not None else Xt
            probs_t = clf.predict_proba(Xt_in)[:, 1]
            preds_t = (probs_t >= 0.5).astype(int)
            student_f1_by_model[m_name] = f1_score(yt, preds_t, average="macro", zero_division=0)

            eval_df = test_df.copy()
            eval_df["prob"] = probs_t
            top1_picks = eval_df.sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False])
            top1_picks = top1_picks.groupby(["trip_id", "obs_idx"]).head(1)
            student_top1_by_model[m_name] = top1_picks["label_y"].mean()

            joblib.dump({"model": clf, "scaler": scaler, "model_name": m_name}, tmp_model_path)
            try:
                cont_df = evaluate_trips(trips, hmm_base_dir, tmp_model_path)
                continuity_by_model[m_name] = cont_df["hmm_continuity_pct"].mean() if not cont_df.empty else np.nan
            except Exception as e:
                print(f"    [warn] HMM continuity evaluation failed for {m_name}: {e}")
                continuity_by_model[m_name] = np.nan

            print(f"    {m_name}: Student-Test Top-1 {student_top1_by_model[m_name]*100:.2f}% | "
                  f"Macro F1 {student_f1_by_model[m_name]:.4f} | Route Continuity {continuity_by_model[m_name]:.2f}%")

        if os.path.exists(tmp_model_path):
            os.remove(tmp_model_path)

        for row in results_table:
            name = row["Model"]
            row["Student-Test Top-1 Accuracy"] = f"{round(student_top1_by_model.get(name, float('nan')) * 100, 2)}%"
            row["Student-Test Macro F1"] = round(student_f1_by_model.get(name, float("nan")), 4)
            row["Route Continuity"] = f"{round(continuity_by_model.get(name, float('nan')), 2)}%"
    else:
        print("    [warn] Skipping student-test/HMM evaluation: test_features.csv or student_test trips not found.")

    # Compile Table
    res_df = pd.DataFrame(results_table)
    csv_out = os.path.join(output_dir, "five_model_comparison.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"\nSaved consolidated 5-model comparison table to: {csv_out}")
    print("\n" + res_df.to_string(index=False))

    # Identify Winning Model (Champion)
    best_m_name = max(trained_models.keys(), key=lambda k: trained_models[k][2])
    best_clf, best_scaler, best_score = trained_models[best_m_name]
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump({"model": best_clf, "scaler": best_scaler, "model_name": best_m_name}, best_model_path)
    print(f"\nChampion Model: {best_m_name} (Macro F1 = {best_score:.4f}) saved to {best_model_path}")

    # Generate Mandatory Comparison Plots for the Technical Report
    plot_comparisons(res_df, plot_dir)

def plot_comparisons(df: pd.DataFrame, plot_dir: str):
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # 1. Macro F1 Plot
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(df["Model"], df["CV Macro F1"], color=palette, edgecolor="black", alpha=0.85)
    plt.title("Cross-Validation Macro F1-Score Across 5 Algorithm Families", fontsize=12, fontweight="bold")
    plt.ylabel("CV Macro F1-Score", fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    p1 = os.path.join(plot_dir, "model_macro_f1_comparison.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Top-1 Accuracy Plot
    plt.figure(figsize=(9, 4.5))
    acc_floats = df["Top-1 Accuracy"].str.rstrip("%").astype(float)
    bars = plt.bar(df["Model"], acc_floats, color=palette, edgecolor="black", alpha=0.85)
    plt.title("Observation-Level Top-1 Accuracy Across 5 Algorithm Families", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    p2 = os.path.join(plot_dir, "model_top1_acc_comparison.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Log Loss Comparison Plot
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(df["Model"], df["Log Loss"], color=palette, edgecolor="black", alpha=0.85)
    plt.title("Log Loss (Cross-Entropy) Comparison Across Models", fontsize=12, fontweight="bold")
    plt.ylabel("Log Loss (Lower is better)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    p3 = os.path.join(plot_dir, "model_logloss_comparison.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

if __name__ == "__main__":
    run_5_model_study()
