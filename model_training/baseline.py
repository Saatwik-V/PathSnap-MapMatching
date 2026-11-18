"""
Baseline Model Training & Group-Aware Evaluation - PathSnap
Author: Saatwik
Evaluates initial baseline models across candidate road segment features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, log_loss

FEATURE_COLS = [
    "perp_dist",
    "proj_ratio",
    "heading_error",
    "arc_length",
    "dist_to_start",
    "dist_to_end"
]

def evaluate_baseline_models(csv_path="results/train_features.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        print(f"Error: Feature dataset not found at {csv_path}. Please run candidate_gen.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} candidate rows from {csv_path}")
    print(f"Trips included: {df['trip_id'].nunique()} distinct trips")
    print(f"Class distribution: {df['label_y'].value_counts(normalize=True).to_dict()}\n")

    X = df[FEATURE_COLS].values
    y = df["label_y"].values
    groups = df["trip_id"].values

    models = {
        "Logistic Regression (Standardized)": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    }

    n_splits = min(3, df["trip_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    results = []

    for name, model in models.items():
        print(f"--- Evaluating {name} ({n_splits}-Fold Group CV) ---")
        f1_scores = []
        top1_accs = []
        log_losses = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            val_df = df.iloc[val_idx]

            scaler = StandardScaler()
            if "Standardized" in name:
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)

            fold_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            f1_scores.append(fold_f1)

            # Compute observation-level Top-1 Accuracy:
            # For each (trip_id, obs_idx), pick candidate with highest probability
            val_df = val_df.copy()
            val_df["pred_score"] = probs
            
            # Find candidate with max score per observation
            top_preds = val_df.sort_values(["trip_id", "obs_idx", "pred_score"], ascending=[True, True, False])
            top_preds = top_preds.groupby(["trip_id", "obs_idx"]).head(1)
            obs_acc = top_preds["label_y"].mean()
            top1_accs.append(obs_acc)

            try:
                ll = log_loss(y_val, probs)
                log_losses.append(ll)
            except Exception:
                pass

        mean_f1 = np.mean(f1_scores)
        mean_top1 = np.mean(top1_accs)
        mean_ll = np.mean(log_losses) if log_losses else 0.0

        print(f"  CV Macro F1:        {mean_f1:.4f}")
        print(f"  Obs Top-1 Accuracy: {mean_top1 * 100:.2f}%")
        print(f"  Log Loss:           {mean_ll:.4f}\n")

        results.append({
            "Model": name,
            "CV Macro F1": round(mean_f1, 4),
            "Top-1 Accuracy": f"{round(mean_top1 * 100, 2)}%",
            "Log Loss": round(mean_ll, 4)
        })

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, "baseline_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved baseline comparison to: {summary_path}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    # Fit best model on all data and save artifact for pre-lunch demo
    rf_best = models["Random Forest (100 Trees)"]
    rf_best.fit(X, y)
    model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(rf_best, model_path)
    print(f"\nSaved champion model artifact to: {model_path}")

if __name__ == "__main__":
    evaluate_baseline_models()
