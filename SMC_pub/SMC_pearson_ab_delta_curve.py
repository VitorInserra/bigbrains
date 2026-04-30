#!/usr/bin/env python3
"""
Compare the TRUE A-vs-B trajectory gap to the PREDICTED A-vs-B trajectory gap
using a Pearson correlation on delta-curves.

Definition used here:
    delta_true(bin) = mean_true_B(bin) - mean_true_A(bin)
    delta_pred(bin) = mean_pred_B(bin) - mean_pred_A(bin)

Then compute:
    Pearson(delta_true, delta_pred)

Important behavior:
1. test_version == 1 is always labeled "Version A"
2. test_version == 2 is always labeled "Version B"
3. Predictions are NEVER trimmed or outlier-filtered
4. matched_vr_eeg_row_pairs is reduced to the exact rows that exist in predictions
5. TRUE uses matched.performance on the merged row universe
6. PREDICTED uses y_pred on the same merged row universe
7. y_true is checked against matched.performance as a sanity check
8. Curves are built over normalized within-session progress bins
9. Each session contributes equally within a bin by averaging within
   (session, bin) first, then averaging across sessions

Outputs:
- merged_prediction_truth_rows.csv
- true_version_curve.csv
- predicted_version_curve.csv
- ab_delta_curve.csv
- pearson_ab_delta_report.txt
- ab_delta_curves.png
- ab_delta_scatter.png
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a Pearson correlation between the TRUE A-vs-B delta curve "
            "and the PREDICTED A-vs-B delta curve on the overlap-only row universe."
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
        default="pearson_ab_delta_outputs",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of normalized progress bins used to build the curves.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser


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


def add_normalized_progress(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    df = df.copy()
    df = df.sort_values(["session_id", "vr_id", "eeg_id", "trial_id"]).reset_index(drop=True)

    df["trial_index"] = df.groupby("session_id").cumcount()
    df["n_trials_in_session"] = df.groupby("session_id")["trial_id"].transform("size")

    denom = (df["n_trials_in_session"] - 1).replace(0, np.nan)
    df["progress_0_1"] = (df["trial_index"] / denom).fillna(0.0)

    # Bin into [0, n_bins-1], with the last point included in the final bin.
    df["progress_bin"] = np.floor(df["progress_0_1"] * n_bins).astype(int)
    df["progress_bin"] = df["progress_bin"].clip(upper=n_bins - 1)
    df["bin_center"] = (df["progress_bin"] + 0.5) / n_bins

    return df


def build_version_curve(df: pd.DataFrame, value_col: str, curve_value_name: str) -> pd.DataFrame:
    # Give each session equal weight within each bin.
    session_bin = (
        df.groupby(["session_id", "test_version", "progress_bin", "bin_center"], as_index=False)
        .agg(session_bin_value=(value_col, "mean"))
    )

    curve = (
        session_bin.groupby(["test_version", "progress_bin", "bin_center"], as_index=False)
        .agg(
            **{curve_value_name: ("session_bin_value", "mean")},
            n_sessions=("session_id", "nunique"),
        )
        .sort_values(["test_version", "progress_bin"])
        .reset_index(drop=True)
    )
    return curve


def build_delta_curve(curve: pd.DataFrame, curve_value_name: str) -> pd.DataFrame:
    pivot = curve.pivot(index=["progress_bin", "bin_center"], columns="test_version", values=curve_value_name)
    pivot = pivot.rename(columns={1: "version_A", 2: "version_B"}).reset_index()

    required_cols = {"version_A", "version_B"}
    missing = required_cols - set(pivot.columns)
    if missing:
        raise ValueError(f"Could not build A/B delta curve; missing columns: {sorted(missing)}")

    pivot["delta_B_minus_A"] = pivot["version_B"] - pivot["version_A"]
    return pivot.sort_values("progress_bin").reset_index(drop=True)


def compute_delta_curve_correlation(true_delta: pd.DataFrame, pred_delta: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    merged_delta = true_delta.merge(
        pred_delta,
        on=["progress_bin", "bin_center"],
        suffixes=("_true", "_pred"),
        how="inner",
    )

    needed = ["delta_B_minus_A_true", "delta_B_minus_A_pred"]
    merged_delta = merged_delta.dropna(subset=needed).copy()

    if len(merged_delta) < 3:
        raise ValueError("Need at least 3 aligned bins to compute a meaningful Pearson correlation.")

    r, p = pearsonr(merged_delta["delta_B_minus_A_true"], merged_delta["delta_B_minus_A_pred"])
    return merged_delta, float(r), float(p)


def write_report(
    merged_rows: pd.DataFrame,
    true_curve: pd.DataFrame,
    pred_curve: pd.DataFrame,
    merged_delta: pd.DataFrame,
    pearson_r: float,
    pearson_p: float,
    report_path: str,
    n_bins: int,
) -> None:
    tee = Tee()

    def p(msg: str = "") -> None:
        tee.write(msg + "\n")

    diff = np.abs(merged_rows["performance"].to_numpy() - merged_rows["y_true"].to_numpy())
    counts = merged_rows["test_version"].value_counts().sort_index()

    p("=== SANITY CHECKS ===")
    p("Version mapping used throughout the script:")
    p("  test_version == 1 -> Version A")
    p("  test_version == 2 -> Version B")
    p("")
    p(f"Prediction rows kept: {len(merged_rows)}")
    p("No outlier trimming is applied in this script.")
    p("matched_vr_eeg_row_pairs is restricted to rows that appear in predictions.")
    p(f"Normalized progress bins used: {n_bins}")
    p("")
    p("Truth consistency check between matched.performance and y_true:")
    p(f"  max abs difference  = {diff.max():.12f}")
    p(f"  mean abs difference = {diff.mean():.12f}")
    p(f"  allclose(atol=1e-6) = {bool(np.allclose(merged_rows['performance'], merged_rows['y_true'], atol=1e-6))}")
    p("")
    p("Row counts by version on merged universe:")
    p(f"  Version A rows (1): {int(counts.get(1, 0))}")
    p(f"  Version B rows (2): {int(counts.get(2, 0))}")
    p("")
    p("=== DELTA-CURVE DEFINITION ===")
    p("Within each session, rows are ordered by vr_id, then eeg_id, then trial_id.")
    p("Each session is mapped onto normalized progress from 0 to 1 and binned.")
    p("Within each (session, bin), values are averaged first.")
    p("Then version-level curves are formed by averaging across sessions.")
    p("Finally:")
    p("  delta_true(bin) = VersionB_true(bin) - VersionA_true(bin)")
    p("  delta_pred(bin) = VersionB_pred(bin) - VersionA_pred(bin)")
    p("and Pearson(delta_true, delta_pred) is computed across bins.")
    p("")
    p("=== PEARSON RESULT ON A/B DELTA CURVES ===")
    p(f"Aligned bins used = {len(merged_delta)}")
    p(f"Pearson r         = {pearson_r:.6f}")
    p(f"Pearson p-value   = {pearson_p:.6g}")
    p("")
    p("=== TRUE DELTA CURVE ===")
    for _, row in merged_delta.iterrows():
        p(
            f"bin {int(row['progress_bin']):2d} (center={row['bin_center']:.2f}): "
            f"A={row['version_A_true']:.6f}, B={row['version_B_true']:.6f}, "
            f"delta={row['delta_B_minus_A_true']:.6f}"
        )
    p("")
    p("=== PREDICTED DELTA CURVE ===")
    for _, row in merged_delta.iterrows():
        p(
            f"bin {int(row['progress_bin']):2d} (center={row['bin_center']:.2f}): "
            f"A={row['version_A_pred']:.6f}, B={row['version_B_pred']:.6f}, "
            f"delta={row['delta_B_minus_A_pred']:.6f}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())


def save_delta_curves_plot(
    true_delta: pd.DataFrame,
    pred_delta: pd.DataFrame,
    output_path: str,
    show_plots: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    axes[0].plot(true_delta["bin_center"], true_delta["version_A"], marker="o", label="True A")
    axes[0].plot(true_delta["bin_center"], true_delta["version_B"], marker="o", label="True B")
    axes[0].set_title("True Version Curves")
    axes[0].set_xlabel("Normalized Trial Progress")
    axes[0].set_ylabel("Performance")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(pred_delta["bin_center"], pred_delta["version_A"], marker="o", label="Pred A")
    axes[1].plot(pred_delta["bin_center"], pred_delta["version_B"], marker="o", label="Pred B")
    axes[1].set_title("Predicted Version Curves")
    axes[1].set_xlabel("Normalized Trial Progress")
    axes[1].set_ylabel("Performance")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def save_delta_scatter_plot(
    merged_delta: pd.DataFrame,
    pearson_r: float,
    pearson_p: float,
    output_path: str,
    show_plots: bool,
) -> None:
    x = merged_delta["delta_B_minus_A_true"].to_numpy()
    y = merged_delta["delta_B_minus_A_pred"].to_numpy()

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, alpha=0.9)

    for _, row in merged_delta.iterrows():
        plt.annotate(str(int(row["progress_bin"])), (row["delta_B_minus_A_true"], row["delta_B_minus_A_pred"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.25
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.5)
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)

    plt.xlabel("True A/B Delta per Bin (B - A)")
    plt.ylabel("Predicted A/B Delta per Bin (B - A)")
    plt.title(f"A/B Delta-Curve Pearson: r={pearson_r:.3f}, p={pearson_p:.3g}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    merged = load_and_merge(args.predictions_csv, args.matched_csv)
    merged = add_normalized_progress(merged, args.n_bins)

    true_curve = build_version_curve(merged, "performance", "curve_true")
    pred_curve = build_version_curve(merged, "y_pred", "curve_pred")

    true_delta = build_delta_curve(true_curve, "curve_true")
    pred_delta = build_delta_curve(pred_curve, "curve_pred")

    merged_delta, pearson_r, pearson_p = compute_delta_curve_correlation(true_delta, pred_delta)

    merged_path = os.path.join(args.output_dir, "merged_prediction_truth_rows.csv")
    true_curve_path = os.path.join(args.output_dir, "true_version_curve.csv")
    pred_curve_path = os.path.join(args.output_dir, "predicted_version_curve.csv")
    delta_curve_path = os.path.join(args.output_dir, "ab_delta_curve.csv")
    report_path = os.path.join(args.output_dir, "pearson_ab_delta_report.txt")
    curves_plot_path = os.path.join(args.output_dir, "ab_delta_curves.png")
    scatter_plot_path = os.path.join(args.output_dir, "ab_delta_scatter.png")

    merged.to_csv(merged_path, index=False)
    true_curve.to_csv(true_curve_path, index=False)
    pred_curve.to_csv(pred_curve_path, index=False)
    merged_delta.to_csv(delta_curve_path, index=False)

    write_report(
        merged_rows=merged,
        true_curve=true_curve,
        pred_curve=pred_curve,
        merged_delta=merged_delta,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        report_path=report_path,
        n_bins=args.n_bins,
    )
    save_delta_curves_plot(true_delta, pred_delta, curves_plot_path, args.show_plots)
    save_delta_scatter_plot(merged_delta, pearson_r, pearson_p, scatter_plot_path, args.show_plots)

    print(f"Saved merged rows to '{merged_path}'.")
    print(f"Saved true version curve to '{true_curve_path}'.")
    print(f"Saved predicted version curve to '{pred_curve_path}'.")
    print(f"Saved aligned delta curve to '{delta_curve_path}'.")
    print(f"Saved report to '{report_path}'.")
    print(f"Saved curve plot to '{curves_plot_path}'.")
    print(f"Saved delta scatter plot to '{scatter_plot_path}'.")


if __name__ == "__main__":
    main()
