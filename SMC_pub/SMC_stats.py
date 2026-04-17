#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp

def run_session_level_ab_test(csv_path="feature_table.csv"):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        sys.exit(1)

    required_cols = {"session_id", "test_version", "performance_metric"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: missing required columns: {sorted(missing)}")
        sys.exit(1)

    # Keep only relevant rows
    df = df[["session_id", "test_version", "performance_metric"]].dropna()

    # Restrict to A/B only
    df = df[df["test_version"].isin([1, 2])].copy()

    if df.empty:
        print("Error: no rows found for test_version A/B after filtering.")
        sys.exit(1)

    # Aggregate per session
    session_summary = (
        df.groupby(["session_id", "test_version"], as_index=False)
          .agg(
              session_median_performance=("performance_metric", "median"),
              session_mean_performance=("performance_metric", "mean"),
              n_rounds=("performance_metric", "size")
          )
    )

    # Use median as primary session-level outcome
    A = session_summary.loc[
        session_summary["test_version"] == 1,
        "session_median_performance"
    ].to_numpy()

    B = session_summary.loc[
        session_summary["test_version"] == 2,
        "session_median_performance"
    ].to_numpy()

    if len(A) < 2 or len(B) < 2:
        print("Error: need at least 2 sessions in each group for comparison.")
        sys.exit(1)

    # Summary stats
    print("\n=== Session-level summary using median performance per session ===")
    print(f"Version A sessions: {len(A)}")
    print(f"Version B sessions: {len(B)}")

    print("\nVersion A")
    print(f"  mean   = {np.mean(A):.6f}")
    print(f"  median = {np.median(A):.6f}")
    print(f"  std    = {np.std(A, ddof=1):.6f}")

    print("\nVersion B")
    print(f"  mean   = {np.mean(B):.6f}")
    print(f"  median = {np.median(B):.6f}")
    print(f"  std    = {np.std(B, ddof=1):.6f}")

    # Mann–Whitney U test
    mw_stat, mw_p = mannwhitneyu(A, B, alternative="two-sided")

    # KS test
    ks_stat, ks_p = ks_2samp(A, B, alternative="two-sided")

    print("\n=== Statistical tests on session-level medians ===")
    print(f"Mann-Whitney U statistic = {mw_stat:.6f}")
    print(f"Mann-Whitney p-value     = {mw_p:.6g}")

    print(f"\nKS statistic             = {ks_stat:.6f}")
    print(f"KS p-value               = {ks_p:.6g}")

    # Directional interpretation
    med_A = np.median(A)
    med_B = np.median(B)

    print("\n=== Interpretation ===")
    print("Lower performance_metric is better.")

    if med_A < med_B:
        print("Version A appears better based on lower session-level median.")
    elif med_B < med_A:
        print("Version B appears better based on lower session-level median.")
    else:
        print("Both versions have the same session-level median.")

    # Save session summary
    session_summary.to_csv("session_level_performance_summary.csv", index=False)
    print("\nSaved session summaries to 'session_level_performance_summary.csv'.")

    # Plot histogram overlay
    plt.figure(figsize=(10, 6))
    plt.hist(A, bins=15, alpha=0.6, edgecolor="black", label="Version A")
    plt.hist(B, bins=15, alpha=0.6, edgecolor="black", label="Version B")
    plt.axvline(np.median(A), color="blue", linestyle="--", linewidth=2, label="A median")
    plt.axvline(np.median(B), color="orange", linestyle="--", linewidth=2, label="B median")
    plt.title("Session-level Distribution of Performance Metric")
    plt.xlabel("Session Median Performance Metric")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("session_level_ab_histogram.png", dpi=200)
    print("Saved histogram to 'session_level_ab_histogram.png'")
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))

    x_A = np.random.normal(1, 0.04, size=len(A))
    x_B = np.random.normal(2, 0.04, size=len(B))

    plt.scatter(x_A, A, alpha=0.8, label="Version A")
    plt.scatter(x_B, B, alpha=0.8, label="Version B")

    plt.boxplot([A, B], positions=[1, 2], widths=0.35)

    plt.xticks([1, 2], ["Version A", "Version B"])
    plt.ylabel("Session Median Performance Metric")
    plt.title("Session-level Performance by Version")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("session_level_ab_scatter_box.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "feature_table.csv"

    run_session_level_ab_test(csv_file)