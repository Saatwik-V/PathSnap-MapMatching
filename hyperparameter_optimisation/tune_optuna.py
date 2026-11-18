"""
Bayesian Hyperparameter Optimisation (Optuna) - PathSnap
Tunes the champion model family selected by model_training/train_5_models.py
using group-aware cross-validation (grouped by trip_id) to avoid leakage.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss

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

# The 5-model benchmark (model_training/train_5_models.py) on the full 80-trip
# training set picked MLP as the top CV Macro F1 (0.7984), with HistGradientBoosting
# a close second (0.7879, a 0.0105 gap) and ~20x cheaper per fit (2.6s vs ~50s on this
# dataset size). A 30-trial x 3-fold Optuna search over MLP would take over an hour;
# the same search over HistGradientBoosting takes ~4 minutes. Given the Models and
# Training Guidelines doc explicitly lists "inference speed" and "computational
# efficiency" as model-selection criteria (not CV F1 alone), HistGradientBoosting is
# the practical choice to tune -- document this trade-off in the report.
CHAMPION_FAMILY = "histgb"  # "histgb" or "mlp"


def _build_model(family: str, trial: optuna.Trial):
    if family == "histgb":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1e-1, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        }
        clf = HistGradientBoostingClassifier(
            max_iter=150, random_state=42, **params
        )
        return clf, params, False  # no scaler needed
    elif family == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 2)
        if n_layers == 1:
            hidden = (trial.suggest_int("layer1_size", 16, 128),)
        else:
            hidden = (
                trial.suggest_int("layer1_size", 32, 128),
                trial.suggest_int("layer2_size", 8, 64),
            )
        params = {
            "hidden_layer_sizes": hidden,
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        }
        clf = MLPClassifier(
            max_iter=200, early_stopping=True, random_state=42, **params
        )
        return clf, params, True  # needs scaler
    else:
        raise ValueError(f"Unknown champion family: {family}")


def _cv_macro_f1(clf_builder, X, y, groups, n_splits=3, needs_scaler=False):
    gkf = GroupKFold(n_splits=n_splits)
    fold_f1, fold_top1, fold_ll = [], [], []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if needs_scaler:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)

        clf = clf_builder()
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        preds = (probs >= 0.5).astype(int)

        fold_f1.append(f1_score(y_va, preds, average="macro", zero_division=0))
        try:
            fold_ll.append(log_loss(y_va, probs))
        except Exception:
            fold_ll.append(np.nan)
    return float(np.mean(fold_f1)), float(np.nanmean(fold_ll))


def run_optuna_study(
    csv_path: str = "results/train_features.csv",
    output_dir: str = "results",
    family: str = CHAMPION_FAMILY,
    n_trials: int = 30,
    n_splits: int = 3,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y = df["label_y"].values
    groups = df["trip_id"].astype(str).values

    print(f"Loaded {len(df):,} rows across {df['trip_id'].nunique()} trips.")
    print(f"Tuning champion family: {family} | GroupKFold(n_splits={n_splits}) | trials={n_trials}")

    trial_log = []

    def objective(trial: optuna.Trial) -> float:
        clf, params, needs_scaler = _build_model(family, trial)
        f1, ll = _cv_macro_f1(
            lambda: _build_model(family, trial)[0],
            X, y, groups, n_splits=n_splits, needs_scaler=needs_scaler,
        )
        trial.set_user_attr("log_loss", ll)
        trial_log.append({"trial": trial.number, **params, "cv_macro_f1": f1, "log_loss": ll})
        return f1

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_time = time.time() - t0

    trials_df = pd.DataFrame(trial_log)
    trials_csv = os.path.join(output_dir, "optuna_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"\nSaved {len(trials_df)} trials to: {trials_csv}")
    print(f"Optuna search wall time: {search_time:.1f}s")

    best_params = study.best_params
    print(f"Best CV Macro F1: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # ---- Refit tuned model on FULL training data, needs_scaler per family ----
    needs_scaler = family == "mlp"
    best_trial = study.best_trial
    tuned_clf, _, _ = _build_model(family, best_trial)

    if needs_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        t0 = time.time()
        tuned_clf.fit(X_scaled, y)
        tuned_train_time = time.time() - t0
    else:
        scaler = None
        t0 = time.time()
        tuned_clf.fit(X, y)
        tuned_train_time = time.time() - t0

    tuned_path = os.path.join(output_dir, "best_tuned_model.joblib")
    joblib.dump(
        {"model": tuned_clf, "scaler": scaler, "model_name": f"{family}_tuned", "params": best_params},
        tuned_path,
    )
    print(f"Saved tuned model to: {tuned_path}")

    # ---- Build Tuned vs Untuned comparison table ----
    comparison_csv = os.path.join(output_dir, "five_model_comparison.csv")
    untuned_row = None
    if os.path.exists(comparison_csv):
        comp_df = pd.read_csv(comparison_csv)
        name_map = {"histgb": "Histogram Gradient Boosting", "mlp": "Multi-Layer Perceptron (MLP)"}
        match = comp_df[comp_df["Model"] == name_map.get(family, "")]
        if not match.empty:
            untuned_row = match.iloc[0].to_dict()

    # Evaluate tuned model on student-testing partition (held-out trips)
    test_path = os.path.join(output_dir, "test_features.csv")
    tuned_top1, tuned_f1_test, tuned_ll_test, tuned_latency_ms = (np.nan,) * 4
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        Xt = test_df[FEATURE_COLS].values
        yt = test_df["label_y"].values
        Xt_in = scaler.transform(Xt) if needs_scaler else Xt

        t0 = time.time()
        probs_t = tuned_clf.predict_proba(Xt_in)[:, 1]
        tuned_latency_ms = (time.time() - t0) / len(Xt) * 1000

        preds_t = (probs_t >= 0.5).astype(int)
        tuned_f1_test = f1_score(yt, preds_t, average="macro", zero_division=0)
        try:
            tuned_ll_test = log_loss(yt, probs_t)
        except Exception:
            pass

        test_df = test_df.copy()
        test_df["prob"] = probs_t
        top1_picks = test_df.sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False])
        top1_picks = top1_picks.groupby(["trip_id", "obs_idx"]).head(1)
        tuned_top1 = top1_picks["label_y"].mean()

    comparison_rows = [{
        "Variant": f"{family} (UNTUNED)" if untuned_row else f"{family} (baseline unavailable)",
        "CV Macro F1": untuned_row.get("CV Macro F1") if untuned_row else np.nan,
        "Student-Test Top-1 Accuracy": untuned_row.get("Top-1 Accuracy") if untuned_row else "N/A",
        "Log Loss": untuned_row.get("Log Loss") if untuned_row else np.nan,
        "Inference Latency (ms)": untuned_row.get("Inference Latency (ms)") if untuned_row else np.nan,
    }, {
        "Variant": f"{family} (TUNED via Optuna, {n_trials} trials)",
        "CV Macro F1": round(study.best_value, 4),
        "Student-Test Top-1 Accuracy": f"{round(tuned_top1 * 100, 2)}%" if not np.isnan(tuned_top1) else "N/A",
        "Log Loss": round(tuned_ll_test, 4) if not np.isnan(tuned_ll_test) else "N/A",
        "Inference Latency (ms)": round(tuned_latency_ms, 3) if not np.isnan(tuned_latency_ms) else "N/A",
    }]

    tuned_vs_untuned_df = pd.DataFrame(comparison_rows)
    tuned_vs_untuned_csv = os.path.join(output_dir, "tuned_vs_untuned.csv")
    tuned_vs_untuned_df.to_csv(tuned_vs_untuned_csv, index=False)
    print(f"\nTuned vs Untuned comparison:\n{tuned_vs_untuned_df.to_string(index=False)}")
    print(f"Saved to: {tuned_vs_untuned_csv}")

    with open(os.path.join(output_dir, "optuna_best_params.json"), "w") as f:
        json.dump({
            "family": family,
            "best_params": best_params,
            "best_cv_macro_f1": study.best_value,
            "n_trials": n_trials,
            "n_splits": n_splits,
            "search_time_s": search_time,
        }, f, indent=2)

    return study, tuned_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default=CHAMPION_FAMILY, choices=["histgb", "mlp"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--splits", type=int, default=3)
    args = parser.parse_args()
    run_optuna_study(family=args.family, n_trials=args.trials, n_splits=args.splits)
