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
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, df["classifier_only_continuity_pct"], width, label="Classifier-Only", color="#9e9e9e", edgecolor="black")
    ax.bar(x + width / 2, df["hmm_continuity_pct"], width, label="Classifier + HMM/Viterbi", color="#2ca02c", edgecolor="black")

    mean_base = df["classifier_only_continuity_pct"].mean()
    mean_hmm = df["hmm_continuity_pct"].mean()
    ax.axhline(mean_base, color="#555555", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(mean_hmm, color="#1a7a1a", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["trip_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Route Continuity (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Route Continuity: Classifier-Only (mean {mean_base:.1f}%) vs Classifier+HMM (mean {mean_hmm:.1f}%)\n"
        f"10 Student-Test Trips",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_tuned_vs_untuned()
    plot_hmm_continuity_gain()
