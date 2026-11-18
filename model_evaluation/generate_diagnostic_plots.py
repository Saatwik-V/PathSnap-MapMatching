"""
Diagnostic Publication Plots - PathSnap
Generates the 3 mandatory 300 DPI diagnostic figures for the technical report:
  1. results/plots/confusion_matrix_champion.png
  2. results/plots/tuned_vs_untuned_comparison.png
  3. results/plots/hmm_continuity_gain.png
All figures are built from the actual CSV outputs of the training / tuning / HMM
evaluation scripts -- never fabricated.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.extract_features import FEATURE_COLS
from hmm_viterbi.viterbi import reconstruct_route

sns.set_style("whitegrid")
DPI = 300


def plot_confusion_matrix(
    test_csv: str = "results/test_features.csv",
    model_path: str = "results/best_tuned_model.joblib",
    output_path: str = "results/plots/confusion_matrix_champion.png",
):
    df = pd.read_csv(test_csv)
    bundle = joblib.load(model_path)
    model, scaler, model_name = bundle["model"], bundle.get("scaler"), bundle.get("model_name", "model")

    X = df[FEATURE_COLS].values
    y = df["label_y"].values
    if scaler is not None:
        X = scaler.transform(X)
    y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)

    cm = confusion_matrix(y, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not True Arc", "True Arc"])
    disp.plot(ax=ax, cmap="Blues", values_format=".2%", colorbar=False)
    ax.set_title(f"Normalized Confusion Matrix\nChampion Model: {model_name} (student-test set)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"Saved: {output_path}")


def plot_tuned_vs_untuned(
    csv_path: str = "results/tuned_vs_untuned.csv",
    output_path: str = "results/plots/tuned_vs_untuned_comparison.png",
):
    df = pd.read_csv(csv_path)
    acc = df["Student-Test Top-1 Accuracy"].astype(str).str.rstrip("%").astype(float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#9e9e9e", "#2ca02c"]
    bars = ax.bar(df["Variant"], acc, color=colors, edgecolor="black", alpha=0.88)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Student-Test Top-1 Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Bayesian Hyperparameter Tuning: Untuned vs Tuned Model", fontsize=12, fontweight="bold")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hmm_continuity_gain(
    csv_path: str = "results/hmm_vs_classifier_studenttest.csv",
    output_path: str = "results/plots/hmm_continuity_gain.png",
):
    df = pd.read_csv(csv_path, dtype={"trip_id": str})
    df["trip_id"] = df["trip_id"].str.zfill(8)
    x = np.arange(len(df))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, df["classifier_only_continuity_pct"], width, label="Classifier-Only", color="#9e9e9e", edgecolor="black")
    ax.bar(x, df["pre_repair_hmm_continuity_pct"], width, label="Classifier + HMM/Viterbi", color="#6baed6", edgecolor="black")
    ax.bar(x + width, df["hmm_continuity_pct"], width, label="HMM/Viterbi + Connectivity Repair", color="#2ca02c", edgecolor="black")

    mean_base = df["classifier_only_continuity_pct"].mean()
    mean_pre = df["pre_repair_hmm_continuity_pct"].mean()
    mean_hmm = df["hmm_continuity_pct"].mean()
    ax.axhline(mean_base, color="#555555", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(mean_hmm, color="#1a7a1a", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["trip_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Route Continuity (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Route Continuity: Classifier-Only ({mean_base:.1f}%) vs HMM ({mean_pre:.1f}%) vs HMM+Repair ({mean_hmm:.1f}%)\n"
        f"10 Student-Test Trips",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"Saved: {output_path}")


def plot_static_route_map(
    trip_id: str = "00000007",
    base_dir: str = "Student folder/student_test",
    model_path: str = "results/best_tuned_model.joblib",
    output_path: str = "results/plots/static_route_map.png",
):
    """
    A matplotlib-rendered static geographical map (raw GPS, road network, reference
    route, predicted route) for embedding directly in the PDF report -- ReportLab
    cannot embed the interactive Folium HTML, so this serves as the "screenshot or
    exported view" the Deliverables doc requires in the report itself. The full
    interactive versions live in results/maps/.
    """
    trip_path = os.path.join(base_dir, trip_id)
    track = pd.read_csv(os.path.join(trip_path, f"{trip_id}.track"), sep=r"\s+", header=None, names=["lon", "lat", "time"])
    nodes = pd.read_csv(os.path.join(trip_path, f"{trip_id}.nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(os.path.join(trip_path, f"{trip_id}.arcs"), sep=r"\s+", header=None, names=["u", "v"])
    route_path = os.path.join(trip_path, f"{trip_id}.route")
    route_arcs = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()

    result = reconstruct_route(trip_id, base_dir=base_dir, model_path=model_path)
    predicted_arcs = result.get("repaired_route_arc_ids", result.get("predicted_arc_ids", []))

    def arc_lines(arc_ids):
        xs, ys = [], []
        for a in arc_ids:
            if a >= len(arcs):
                continue
            u, v = int(arcs.iloc[a]["u"]), int(arcs.iloc[a]["v"])
            xs += [nodes.iloc[u]["lon"], nodes.iloc[v]["lon"], None]
            ys += [nodes.iloc[u]["lat"], nodes.iloc[v]["lat"], None]
        return xs, ys

    fig, ax = plt.subplots(figsize=(9, 7))
    ref_x, ref_y = arc_lines(route_arcs)
    pred_x, pred_y = arc_lines(predicted_arcs)

    ax.scatter(track["lon"], track["lat"], s=5, color="#1f77b4", alpha=0.35, label="Raw Noisy GPS Observations", zorder=1)
    ax.plot(ref_x, ref_y, color="#2ca02c", linewidth=4, alpha=0.9, solid_capstyle="round", label="Ground-Truth Reference Route", zorder=2)
    ax.plot(pred_x, pred_y, color="#d62728", linewidth=2, linestyle="--", alpha=1.0, label="PathSnap Predicted Route (HMM+Viterbi)", zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"PathSnap Route Reconstruction — Trip {trip_id}\n"
        f"Route Continuity: {result.get('route_continuity_pct', float('nan')):.1f}% | "
        f"Pointwise Accuracy: {result.get('pointwise_accuracy', float('nan')):.3f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"Saved: {output_path}")


def plot_student_test_comparison(
    csv_path: str = "results/five_model_comparison.csv",
    output_path: str = "results/plots/student_test_top1_f1_comparison.png",
):
    df = pd.read_csv(csv_path)
    top1 = df["Student-Test Top-1 Accuracy"].astype(str).str.rstrip("%").astype(float) / 100
    short = df["Model"].str.replace("Histogram Gradient Boosting", "HistGB").str.replace("Multi-Layer Perceptron (MLP)", "MLP", regex=False) \
        .str.replace("Distance-Weighted k-NN", "k-NN")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b1 = ax.bar(x - 0.2, top1, 0.4, label="Student-Test Top-1 Accuracy", color="#4c78a8", edgecolor="black")
    b2 = ax.bar(x + 0.2, df["Student-Test Macro F1"], 0.4, label="Student-Test Macro F1", color="#f58518", edgecolor="black")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=12)
    ax.set_ylim(0.6, 0.92); ax.set_ylabel("Score")
    ax.set_title("Held-out Student-Test Performance (10 trips, 191,600 candidate rows)", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI); plt.close()
    print(f"Saved: {output_path}")


def plot_model_selection_tradeoff(
    csv_path: str = "results/five_model_comparison.csv",
    output_path: str = "results/plots/model_selection_tradeoff.png",
):
    df = pd.read_csv(csv_path)
    top1 = df["Student-Test Top-1 Accuracy"].astype(str).str.rstrip("%").astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(df["Train Time (s)"], df["Student-Test Macro F1"], s=(top1 - 70) * 40 + 60, c="#4c78a8", edgecolor="black", alpha=0.85)
    for _, r in df.iterrows():
        ax.annotate(r["Model"].replace("Histogram Gradient Boosting", "HistGB").replace("Multi-Layer Perceptron (MLP)", "MLP"),
                    (r["Train Time (s)"], r["Student-Test Macro F1"]), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 600)
    ax.set_ylim(0.752, 0.798)
    ax.set_xlabel("Training time on 2.08M rows (s, log scale)")
    ax.set_ylabel("Student-Test Macro F1")
    ax.set_title("Model selection: accuracy vs training cost\n(marker size = Student-Test Top-1 Accuracy)", fontweight="bold", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI); plt.close()
    print(f"Saved: {output_path}")


def plot_optuna_history(
    csv_path: str = "results/optuna_trials.csv",
    output_path: str = "results/plots/optuna_history.png",
):
    df = pd.read_csv(csv_path)
    num, val = "trial", "cv_macro_f1"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(df[num], df[val], "o", color="#9e9e9e", label="Trial CV Macro F1")
    ax.plot(df[num], df[val].cummax(), "-", color="#2ca02c", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("CV Macro F1 (GroupKFold, 3 folds)")
    ax.set_title("Optuna TPE optimisation history (HistGradientBoosting, 30 trials)", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI); plt.close()
    print(f"Saved: {output_path}")


def plot_training_histories(
    mlp_path: str = "results/best_model.joblib",
    histgb_path: str = "results/best_tuned_model.joblib",
    output_path: str = "results/plots/training_histories.png",
):
    """
    Per-iteration histories read straight from the fitted model objects (no retraining, nothing
    synthesised). Only models that actually store one are plotted: MLP (loss_curve_ and
    validation_scores_) and HistGradientBoosting (train_score_/validation_score_, stored as
    negative loss). Logistic Regression (lbfgs), k-NN and Random Forest keep no epoch history.
    """
    mlp = joblib.load(mlp_path)["model"]
    hgb = joblib.load(histgb_path)["model"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ep = np.arange(1, len(mlp.loss_curve_) + 1)
    axes[0].plot(ep, mlp.loss_curve_, color="#d62728", label="Training loss (log loss)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss", color="#d62728")
    ax2 = axes[0].twinx()
    ax2.plot(ep, mlp.validation_scores_, color="#1f77b4", label="Validation accuracy")
    ax2.set_ylabel("Validation accuracy", color="#1f77b4"); ax2.grid(False)
    axes[0].set_title(f"MLP: stopped by early stopping at epoch {mlp.n_iter_} of {mlp.max_iter}", fontsize=10, fontweight="bold")

    it = np.arange(1, len(hgb.train_score_) + 1)
    axes[1].plot(it, -hgb.train_score_, color="#2ca02c", label="Training loss")
    axes[1].plot(it, -hgb.validation_score_, color="#ff7f0e", label="Validation loss (internal 10% split)")
    axes[1].set_xlabel("Boosting iteration"); axes[1].set_ylabel("Log loss")
    axes[1].set_title("Tuned HistGradientBoosting: loss per iteration", fontsize=10, fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI); plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_tuned_vs_untuned()
    plot_hmm_continuity_gain()
    plot_static_route_map()
    plot_student_test_comparison()
    plot_model_selection_tradeoff()
    plot_optuna_history()
    plot_training_histories()
