#!/usr/bin/env python3
"""
PathSnap: Feature Correlation Analysis & Heatmap Visualization
Author: Vikas Jadar (Data Pipeline & Feature Extraction Specialist)

Analyzes inter-feature collinearity and point-biserial correlations of all 12 candidate features
against the binary target label (label_y).
Outputs publication-quality heatmap to results/plots/feature_correlation_heatmap.png.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COLS = [
    "perp_dist",
    "proj_ratio",
    "heading_error",
    "arc_length",
    "dist_to_start",
    "dist_to_end",
    "speed_mps",
    "heading_cos",
    "delta_t",
    "candidate_rank",
    "dist_diff_endpoints",
    "traversal_time_est"
]

def analyze_feature_correlations(csv_path="results/train_features.csv", output_plot="results/plots/feature_correlation_heatmap.png"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature dataset not found: {csv_path}")
        
    print(f"Loading feature dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Target column
    target_col = "label_y" if "label_y" in df.columns else "y"
    analysis_cols = FEATURE_COLS + [target_col]
    
    # Filter available columns
    available_cols = [c for c in analysis_cols if c in df.columns]
    print(f"Auditing correlations across {len(available_cols)-1} features against '{target_col}' ({len(df):,} rows)...")
    
    sub_df = df[available_cols].copy()
    
    # Compute Pearson correlation matrix
    corr_matrix = sub_df.corr(method="pearson")
    
    # Display correlation with target label
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    print("\n------------------------------------------------------------")
    print(f" FEATURE CORRELATION WITH TARGET ({target_col})")
    print("------------------------------------------------------------")
    for feat, val in target_corr.items():
        direction = "Positive (+)" if val > 0 else "Negative (-)"
        print(f"  {feat:22s}: r = {val:+0.4f} ({direction})")
    print("------------------------------------------------------------\n")
    
    # Visual Styling
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Generate correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.75,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient ($r$)"},
        annot_kws={"size": 9, "fontweight": "normal"},
        ax=ax
    )
    
    ax.set_title("PathSnap: 12-Feature Correlation Matrix & Target Association (label_y)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"[Artifact Generated] Heatmap successfully saved to: {output_plot}")
    
    return target_corr

if __name__ == "__main__":
    analyze_feature_correlations()
