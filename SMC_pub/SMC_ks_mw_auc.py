#!/usr/bin/env python3
"""
Compare TRUE and PREDICTED session-level performance by game version using
session-level AUC (area under the per-session trajectory) instead of medians.

Important behavior in this version:
1. test_version == 1 is always labeled "Version A"
2. test_version == 2 is always labeled "Version B"
3. Predictions are NEVER trimmed or outlier-filtered
4. matched_vr_eeg_row_pairs is reduced to the exact rows that exist in predictions
5. TRUE uses matched.performance on the merged row universe
6. PREDICTED uses y_pred on the same merged row universe
7. y_true is checked against matched.performance as a sanity check
8. Session summary statistic for MW/KS is AUC over trial progression within session,
   using rows sorted by vr_id and trapezoidal integration over trial index.

Outputs:
- merged_prediction_truth_rows.csv
- session_level_summary_true_auc.csv
- session_level_summary_predicted_auc.csv
- ab_comparison_auc_report.txt
- ab_scatter_box_auc_shared_scale.png
- ab_histograms_auc_shared_scale.png
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


PRED_REQUIRED_COLS = {
    "outer_fold",
    "trial_id",
    "session_id",
    "y_true",
    "y_pred",
    "abs_error",
    "source_dir",
    "session_trial_key",
}

MATCH_REQUIRED_COLS = {
    "session_id",
    "vr_id",
    "eeg_id",
    "test_version",
    "performance",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare true and predicted session-level performance by version using AUC."
    )
    parser.add_argument(
        "--predictions-csv",
        default="combined_outer_fold_predictions_trimmed.csv",
        help="Path to combined outer-fold predictions CSV.",
    )
    parser.add_argument(
        "--matched-csv",
        default="matched_vr_eeg_row_pairs.csv",
        help="Path to matched_vr_eeg_row_pairs CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="ab_comparison_auc_outputs",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scatter jitter.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser


class Tee:
    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        self.buffer.write(text)

    def flush(self) -> None:
        sys.stdout.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def reconstruct_trial_keys(matched: pd.DataFrame) -> pd.DataFrame:
    matched = matched.copy().reset_index().rename(columns={"index": "matched_row_index"})
    matched["trial_id"] = (
        matched["session_id"].astype(str)
        + "__vr"
        + matched["vr_id"].astype(str)
        + "__eeg"
        + matched["eeg_id"].astype(str)
        + "__row"
        + matched["matched_row_index"].astype(str)
    )
    matched["session_trial_key"] = (
        matched["session_id"].astype(str) + "__" + matched["trial_id"].astype(str)
    )
    return matched


def load_and_merge(predictions_csv: str, matched_csv: str) -> pd.DataFrame:
    pred = pd.read_csv(predictions_csv)
    matched = pd.read_csv(matched_csv)

    validate_columns(pred, PRED_REQUIRED_COLS, "Predictions CSV")
    validate_columns(matched, MATCH_REQUIRED_COLS, "Matched CSV")

    matched = reconstruct_trial_keys(matched)

    keep_cols = [
        "session_id",
        "trial_id",
        "session_trial_key",
        "vr_id",
        "eeg_id",
        "test_version",
        "performance",
    ]

    merged = pred.merge(
        matched[keep_cols],
        on=["trial_id", "session_trial_key"],
        how="left",
        suffixes=("_pred", "_matched"),
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            "Some prediction rows could not be matched back to matched_vr_eeg_row_pairs. "
            f"Unmatched prediction rows: {len(unmatched)}"
        )

    merged = merged.drop(columns=["_merge"]).rename(columns={"session_id_pred": "session_id"})
    if "session_id_matched" in merged.columns:
        mismatch = merged["session_id"].astype(str) != merged["session_id_matched"].astype(str)
        if mismatch.any():
            raise ValueError(
                f"Session ID mismatch after merge for {int(mismatch.sum())} rows."
            )
        merged = merged.drop(columns=["session_id_matched"])

    merged = merged[merged["test_version"].isin([1, 2])].copy()
    if merged.empty:
        raise ValueError("No rows remain after filtering to test_version in [1, 2].")

    return merged


def compute_session_auc(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(arr[0])
    x = np.arange(n, dtype=float)
    return float(np.trapezoid(arr, x=x))


def summarize_sessions_auc(
    merged: pd.DataFrame,
    value_col: str,
    auc_col_name: str,
) -> pd.DataFrame:
    rows = []
    for (session_id, test_version), group in merged.groupby(["session_id", "test_version"], sort=True):
        group = group.sort_values(["vr_id", "eeg_id"]).reset_index(drop=True)
        auc = compute_session_auc(group[value_col])
        rows.append(
            {
                "session_id": session_id,
                "test_version": test_version,
                auc_col_name: auc,
                "session_mean": float(group[value_col].mean()),
                "n_trials": int(len(group)),
                "first_vr_id": int(group["vr_id"].iloc[0]),
                "last_vr_id": int(group["vr_id"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_version", "session_id"]).reset_index(drop=True)


def extract_groups(summary: pd.DataFrame, metric_col: str) -> tuple[np.ndarray, np.ndarray]:
    A = summary.loc[summary["test_version"] == 1, metric_col].to_numpy()
    B = summary.loc[summary["test_version"] == 2, metric_col].to_numpy()
    if len(A) < 2 or len(B) < 2:
        raise ValueError("Need at least 2 sessions in each version group.")
    return A, B


def compute_stats(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    mw_stat, mw_p = mannwhitneyu(A, B, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(A, B, alternative="two-sided")
    return {
        "mean_A": float(np.mean(A)),
        "median_A": float(np.median(A)),
        "std_A": float(np.std(A, ddof=1)),
        "mean_B": float(np.mean(B)),
        "median_B": float(np.median(B)),
        "std_B": float(np.std(B, ddof=1)),
        "mw_stat": float(mw_stat),
        "mw_p": float(mw_p),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


def write_report(
    merged: pd.DataFrame,
    true_summary: pd.DataFrame,
    pred_summary: pd.DataFrame,
    out_path: str,
) -> None:
    tee = Tee()

    def p(msg: str = "") -> None:
        tee.write(msg + "\n")

    true_A, true_B = extract_groups(true_summary, "session_auc_true")
    pred_A, pred_B = extract_groups(pred_summary, "session_auc_predicted")

    true_stats = compute_stats(true_A, true_B)
    pred_stats = compute_stats(pred_A, pred_B)

    diff = np.abs(merged["performance"].to_numpy() - merged["y_true"].to_numpy())

    p("=== SANITY CHECKS ===")
    p("Version mapping used throughout the script:")
    p("  test_version == 1 -> Version A")
    p("  test_version == 2 -> Version B")
    p("")
    p(f"Prediction rows kept: {len(merged)}")
    p("No outlier trimming is applied in this script.")
    p("matched_vr_eeg_row_pairs is restricted to rows that appear in predictions.")
    p("Session statistic for MW/KS: AUC over the within-session trajectory.")
    p("Rows are ordered by vr_id (then eeg_id) within each session.")
    p("AUC uses trapezoidal integration over trial index 0..n-1.")
    p("")
    p("Truth consistency check between matched.performance and y_true:")
    p(f"  max abs difference  = {diff.max():.12f}")
    p(f"  mean abs difference = {diff.mean():.12f}")
    p(f"  allclose(atol=1e-6) = {bool(np.allclose(merged['performance'], merged['y_true'], atol=1e-6))}")
    p("")
    p("Row counts by version on merged universe:")
    counts = merged["test_version"].value_counts().sort_index()
    p(f"  Version A rows (1): {int(counts.get(1, 0))}")
    p(f"  Version B rows (2): {int(counts.get(2, 0))}")
    p("")
    p("Session counts by version on merged universe:")
    p(f"  Version A sessions (1): {len(true_A)}")
    p(f"  Version B sessions (2): {len(true_B)}")
    p("")
    p("AUC caveat:")
    p("  Raw AUC depends on both trajectory height and number of trials in the session.")
    p("  Since session lengths differ, this is a trajectory-size summary, not a pure shape-only metric.")

    p("\n=== TRUE SESSION-LEVEL SUMMARY (using matched.performance AUC) ===")
    p(f"Version A mean   = {true_stats['mean_A']:.6f}")
    p(f"Version A median = {true_stats['median_A']:.6f}")
    p(f"Version A std    = {true_stats['std_A']:.6f}")
    p(f"Version B mean   = {true_stats['mean_B']:.6f}")
    p(f"Version B median = {true_stats['median_B']:.6f}")
    p(f"Version B std    = {true_stats['std_B']:.6f}")
    p(f"Mann-Whitney U   = {true_stats['mw_stat']:.6f}")
    p(f"Mann-Whitney p   = {true_stats['mw_p']:.6g}")
    p(f"KS statistic     = {true_stats['ks_stat']:.6f}")
    p(f"KS p-value       = {true_stats['ks_p']:.6g}")
    p(f"AUC(A)-AUC(B) median difference = {true_stats['median_A'] - true_stats['median_B']:.6f}")
    p("Interpretation: lower is better if lower performance values are better.")

    p("\n=== PREDICTED SESSION-LEVEL SUMMARY (using y_pred AUC) ===")
    p(f"Version A mean   = {pred_stats['mean_A']:.6f}")
    p(f"Version A median = {pred_stats['median_A']:.6f}")
    p(f"Version A std    = {pred_stats['std_A']:.6f}")
    p(f"Version B mean   = {pred_stats['mean_B']:.6f}")
    p(f"Version B median = {pred_stats['median_B']:.6f}")
    p(f"Version B std    = {pred_stats['std_B']:.6f}")
    p(f"Mann-Whitney U   = {pred_stats['mw_stat']:.6f}")
    p(f"Mann-Whitney p   = {pred_stats['mw_p']:.6g}")
    p(f"KS statistic     = {pred_stats['ks_stat']:.6f}")
    p(f"KS p-value       = {pred_stats['ks_p']:.6g}")
    p(f"AUC(A)-AUC(B) median difference = {pred_stats['median_A'] - pred_stats['median_B']:.6f}")
    p("Interpretation: lower is better if lower predicted performance values are better.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())


def save_scatter_box(
    true_A: np.ndarray,
    true_B: np.ndarray,
    pred_A: np.ndarray,
    pred_B: np.ndarray,
    output_path: str,
    seed: int,
    show_plots: bool,
) -> None:
    rng = np.random.default_rng(seed)

    all_vals = np.concatenate([true_A, true_B, pred_A, pred_B])
    y_pad = 0.05 * (all_vals.max() - all_vals.min()) if all_vals.max() > all_vals.min() else 0.25
    y_min = all_vals.min() - y_pad
    y_max = all_vals.max() + y_pad

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    panels = [
        (axes[0], true_A, true_B, "True Session-level AUC by Version"),
        (axes[1], pred_A, pred_B, "Predicted Session-level AUC by Version"),
    ]

    for ax, A, B, title in panels:
        x_A = rng.normal(1, 0.04, size=len(A))
        x_B = rng.normal(2, 0.04, size=len(B))
        ax.scatter(x_A, A, alpha=0.8)
        ax.scatter(x_B, B, alpha=0.8)
        ax.boxplot([A, B], positions=[1, 2], widths=0.35)
        ax.set_xticks([1, 2], ["Version A", "Version B"])
        ax.set_ylabel("Session AUC of Performance Metric")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def save_histograms(
    true_A: np.ndarray,
    true_B: np.ndarray,
    pred_A: np.ndarray,
    pred_B: np.ndarray,
    output_path: str,
    show_plots: bool,
) -> None:
    all_vals = np.concatenate([true_A, true_B, pred_A, pred_B])
    bins = np.linspace(all_vals.min(), all_vals.max(), 12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    panels = [
        (axes[0], true_A, true_B, "True Session-level AUC Distribution"),
        (axes[1], pred_A, pred_B, "Predicted Session-level AUC Distribution"),
    ]

    for ax, A, B, title in panels:
        ax.hist(A, bins=bins, alpha=0.6, edgecolor="black", label="Version A")
        ax.hist(B, bins=bins, alpha=0.6, edgecolor="black", label="Version B")
        ax.axvline(np.median(A), linestyle="--", linewidth=2, label="A median")
        ax.axvline(np.median(B), linestyle="--", linewidth=2, label="B median")
        ax.set_title(title)
        ax.set_xlabel("Session AUC of Performance Metric")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    merged = load_and_merge(args.predictions_csv, args.matched_csv)

    true_summary = summarize_sessions_auc(merged, "performance", "session_auc_true")
    pred_summary = summarize_sessions_auc(merged, "y_pred", "session_auc_predicted")

    merged_path = os.path.join(args.output_dir, "merged_prediction_truth_rows.csv")
    true_summary_path = os.path.join(args.output_dir, "session_level_summary_true_auc.csv")
    pred_summary_path = os.path.join(args.output_dir, "session_level_summary_predicted_auc.csv")
    report_path = os.path.join(args.output_dir, "ab_comparison_auc_report.txt")
    scatter_path = os.path.join(args.output_dir, "ab_scatter_box_auc_shared_scale.png")
    hist_path = os.path.join(args.output_dir, "ab_histograms_auc_shared_scale.png")

    merged.to_csv(merged_path, index=False)
    true_summary.to_csv(true_summary_path, index=False)
    pred_summary.to_csv(pred_summary_path, index=False)

    write_report(merged, true_summary, pred_summary, report_path)

    true_A, true_B = extract_groups(true_summary, "session_auc_true")
    pred_A, pred_B = extract_groups(pred_summary, "session_auc_predicted")

    save_scatter_box(true_A, true_B, pred_A, pred_B, scatter_path, args.seed, args.show_plots)
    save_histograms(true_A, true_B, pred_A, pred_B, hist_path, args.show_plots)

    print(f"Saved merged rows to '{merged_path}'.")
    print(f"Saved true AUC session summary to '{true_summary_path}'.")
    print(f"Saved predicted AUC session summary to '{pred_summary_path}'.")
    print(f"Saved report to '{report_path}'.")
    print(f"Saved scatter-box plot to '{scatter_path}'.")
    print(f"Saved histograms to '{hist_path}'.")


if __name__ == "__main__":
    main()
