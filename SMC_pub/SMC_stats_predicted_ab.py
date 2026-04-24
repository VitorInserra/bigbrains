#!/usr/bin/env python3
"""
Session-level A/B comparison using predicted performance scores.

This script:
1. Reads combined outer-fold predictions.
2. Reconstructs the trial keys in matched_vr_eeg_row_pairs.csv.
3. Looks up test_version for each predicted row.
4. Aggregates predicted performance by session.
5. Runs Mann-Whitney U and KS tests on session-level medians.
6. Saves a text report, merged lookup CSV, session summary CSV,
   histogram, and scatter-box plot.

Default inputs:
- combined_outer_fold_predictions_trimmed.csv
- matched_vr_eeg_row_pairs.csv
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Tuple

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
        description=(
            "Run a session-level A/B comparison on predicted performance scores "
            "using test_version recovered from matched_vr_eeg_row_pairs.csv."
        )
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
        default=".",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for jitter in the scatter-box plot.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser


class Tee:
    """Write to stdout and a text buffer at the same time."""

    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        self.buffer.write(text)

    def flush(self) -> None:
        sys.stdout.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


def validate_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def load_inputs(predictions_csv: str, matched_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        pred = pd.read_csv(predictions_csv)
    except Exception as exc:
        raise RuntimeError(f"Error reading predictions CSV '{predictions_csv}': {exc}") from exc

    try:
        matched = pd.read_csv(matched_csv)
    except Exception as exc:
        raise RuntimeError(f"Error reading matched CSV '{matched_csv}': {exc}") from exc

    validate_columns(pred, PRED_REQUIRED_COLS, "Predictions CSV")
    validate_columns(matched, MATCH_REQUIRED_COLS, "Matched CSV")

    return pred, matched


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


def merge_predictions_with_versions(pred: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    lookup_cols = [
        "session_id",
        "trial_id",
        "session_trial_key",
        "test_version",
        "performance",
    ]

    merged = pred.merge(
        matched[lookup_cols],
        on=["trial_id", "session_trial_key"],
        how="left",
        suffixes=("_pred", "_matched"),
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            "Some prediction rows could not be matched back to test_version. "
            f"Unmatched rows: {len(unmatched)}"
        )

    session_id_pred = merged["session_id_pred"].astype(str)
    session_id_matched = merged["session_id_matched"].astype(str)
    mismatch_mask = session_id_pred != session_id_matched
    if mismatch_mask.any():
        raise ValueError(
            "Session ID mismatch detected after merge for "
            f"{int(mismatch_mask.sum())} rows."
        )

    merged = merged.drop(columns=["_merge", "session_id_matched"]).rename(
        columns={"session_id_pred": "session_id"}
    )

    merged = merged[merged["test_version"].isin([1, 2])].copy()
    if merged.empty:
        raise ValueError("No merged rows remain after filtering to test_version in [1, 2].")

    return merged


def summarize_by_session(merged: pd.DataFrame) -> pd.DataFrame:
    session_summary = (
        merged.groupby(["session_id", "test_version"], as_index=False)
        .agg(
            session_median_predicted_performance=("y_pred", "median"),
            session_mean_predicted_performance=("y_pred", "mean"),
            session_median_true_performance=("y_true", "median"),
            session_mean_true_performance=("y_true", "mean"),
            mean_abs_error=("abs_error", "mean"),
            n_predicted_trials=("y_pred", "size"),
        )
        .sort_values(["test_version", "session_id"])
        .reset_index(drop=True)
    )
    return session_summary


def print_and_collect_report(
    merged: pd.DataFrame,
    session_summary: pd.DataFrame,
    out_stream: Tee,
) -> Tuple[np.ndarray, np.ndarray]:
    A = session_summary.loc[
        session_summary["test_version"] == 1,
        "session_median_predicted_performance",
    ].to_numpy()

    B = session_summary.loc[
        session_summary["test_version"] == 2,
        "session_median_predicted_performance",
    ].to_numpy()

    if len(A) < 2 or len(B) < 2:
        raise ValueError("Need at least 2 sessions in each version group for comparison.")

    def p(msg: str = "") -> None:
        out_stream.write(msg + "\n")

    p("=== Predicted session-level summary using median predicted performance per session ===")
    p(f"Predicted rows merged successfully: {len(merged)}")
    p(f"Version A trials: {(merged['test_version'] == 1).sum()}")
    p(f"Version B trials: {(merged['test_version'] == 2).sum()}")
    p(f"Version A sessions: {len(A)}")
    p(f"Version B sessions: {len(B)}")

    p("\nVersion A (predicted session medians)")
    p(f"  mean   = {np.mean(A):.6f}")
    p(f"  median = {np.median(A):.6f}")
    p(f"  std    = {np.std(A, ddof=1):.6f}")
    p(f"  min    = {np.min(A):.6f}")
    p(f"  max    = {np.max(A):.6f}")

    p("\nVersion B (predicted session medians)")
    p(f"  mean   = {np.mean(B):.6f}")
    p(f"  median = {np.median(B):.6f}")
    p(f"  std    = {np.std(B, ddof=1):.6f}")
    p(f"  min    = {np.min(B):.6f}")
    p(f"  max    = {np.max(B):.6f}")

    mw_stat, mw_p = mannwhitneyu(A, B, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(A, B, alternative="two-sided")

    p("\n=== Statistical tests on predicted session-level medians ===")
    p(f"Mann-Whitney U statistic = {mw_stat:.6f}")
    p(f"Mann-Whitney p-value     = {mw_p:.6g}")
    p("")
    p(f"KS statistic             = {ks_stat:.6f}")
    p(f"KS p-value               = {ks_p:.6g}")

    med_A = np.median(A)
    med_B = np.median(B)
    delta = med_A - med_B

    p("\n=== Interpretation ===")
    p("Lower predicted performance metric is better.")
    p(f"Median(A) - Median(B) = {delta:.6f}")
    if med_A < med_B:
        p("Version A appears better based on lower predicted session-level median.")
    elif med_B < med_A:
        p("Version B appears better based on lower predicted session-level median.")
    else:
        p("Both versions have the same predicted session-level median.")

    return A, B


def save_histogram(A: np.ndarray, B: np.ndarray, output_path: str, show_plots: bool) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(A, bins=15, alpha=0.6, edgecolor="black", label="Version A")
    plt.hist(B, bins=15, alpha=0.6, edgecolor="black", label="Version B")
    plt.axvline(np.median(A), color="blue", linestyle="--", linewidth=2, label="A median")
    plt.axvline(np.median(B), color="orange", linestyle="--", linewidth=2, label="B median")
    plt.title("Session-level Distribution of Predicted Performance Metric")
    plt.xlabel("Session Median Predicted Performance Metric")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def save_scatter_box(
    A: np.ndarray,
    B: np.ndarray,
    output_path: str,
    seed: int,
    show_plots: bool,
) -> None:
    rng = np.random.default_rng(seed)

    plt.figure(figsize=(8, 6))
    x_A = rng.normal(1, 0.04, size=len(A))
    x_B = rng.normal(2, 0.04, size=len(B))

    plt.scatter(x_A, A, alpha=0.8, label="Version A")
    plt.scatter(x_B, B, alpha=0.8, label="Version B")
    plt.boxplot([A, B], positions=[1, 2], widths=0.35)
    plt.xticks([1, 2], ["Version A", "Version B"])
    plt.ylabel("Session Median Predicted Performance Metric")
    plt.title("Predicted Session-level Performance by Version")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        pred, matched = load_inputs(args.predictions_csv, args.matched_csv)
        matched = reconstruct_trial_keys(matched)
        merged = merge_predictions_with_versions(pred, matched)
        session_summary = summarize_by_session(merged)

        merged_csv = os.path.join(args.output_dir, "predicted_trial_version_lookup.csv")
        session_csv = os.path.join(args.output_dir, "predicted_session_level_summary.csv")
        report_txt = os.path.join(args.output_dir, "predicted_session_level_ab_report.txt")
        hist_png = os.path.join(args.output_dir, "predicted_session_level_ab_histogram.png")
        scatter_png = os.path.join(args.output_dir, "predicted_session_level_ab_scatter_box.png")

        merged.to_csv(merged_csv, index=False)
        session_summary.to_csv(session_csv, index=False)

        tee = Tee()
        A, B = print_and_collect_report(merged, session_summary, tee)

        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(tee.getvalue())

        save_histogram(A, B, hist_png, args.show_plots)
        save_scatter_box(A, B, scatter_png, args.seed, args.show_plots)

        print(f"\nSaved merged prediction/version lookup to '{merged_csv}'.")
        print(f"Saved session summaries to '{session_csv}'.")
        print(f"Saved report to '{report_txt}'.")
        print(f"Saved histogram to '{hist_png}'.")
        print(f"Saved scatter-box plot to '{scatter_png}'.")

    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
