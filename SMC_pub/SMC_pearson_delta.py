#!/usr/bin/env python3
"""
A/B delta-curve diagnostics on the overlap-only row universe.

This script keeps the same overlap-only merge logic:
1. test_version == 1 is Version A
2. test_version == 2 is Version B
3. predictions are never trimmed
4. matched rows are restricted to rows that appear in predictions
5. truth uses matched.performance
6. prediction uses y_pred
7. y_true is checked against matched.performance
8. curves are built over normalized within-session progress bins
9. each session contributes equally within a bin

Diagnostics included:
- Delta-curve Pearson + delta scatter
- Bland-Altman plot on aligned A/B delta bins
- Session spaghetti plots (thin session curves + thick version mean curves)
- Session-bin heatmaps (True A, True B, Predicted A, Predicted B)
- Difference-only ribbon plot (true and predicted B-A curves with 95% CI ribbons)
- Cumulative difference plot (cumulative integral of B-A over normalized progress)

Outputs:
- merged_prediction_truth_rows.csv
- true_session_bin_curve.csv
- predicted_session_bin_curve.csv
- true_version_curve.csv
- predicted_version_curve.csv
- ab_delta_curve.csv
- ab_delta_ribbon_curve.csv
- ab_delta_cumulative_curve.csv
- pearson_ab_delta_report.txt
- ab_delta_curves.png
- ab_delta_scatter.png
- ab_delta_bland_altman.png
- ab_session_spaghetti.png
- ab_session_heatmaps.png
- ab_delta_ribbon.png
- ab_delta_cumulative.png
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
            "A/B delta-curve diagnostics with Bland-Altman, session spaghetti, heatmaps, "
            "difference-only ribbon, and cumulative-difference plots."
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
        help="Number of normalized progress bins.",
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
            raise ValueError(f"Session ID mismatch after merge for {int(mismatch.sum())} rows.")
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
    df["progress_bin"] = np.floor(df["progress_0_1"] * n_bins).astype(int).clip(upper=n_bins - 1)
    df["bin_center"] = (df["progress_bin"] + 0.5) / n_bins
    return df


def build_session_bin_curve(
    df: pd.DataFrame,
    value_col: str,
    session_value_name: str,
) -> pd.DataFrame:
    return (
        df.groupby(["session_id", "test_version", "progress_bin", "bin_center"], as_index=False)
        .agg(**{session_value_name: (value_col, "mean")})
        .sort_values(["test_version", "session_id", "progress_bin"])
        .reset_index(drop=True)
    )


def build_version_curve(
    session_bin_curve: pd.DataFrame,
    session_value_name: str,
    curve_value_name: str,
) -> pd.DataFrame:
    return (
        session_bin_curve.groupby(["test_version", "progress_bin", "bin_center"], as_index=False)
        .agg(
            **{curve_value_name: (session_value_name, "mean")},
            curve_sd=(session_value_name, "std"),
            n_sessions=(session_value_name, "count"),
        )
        .sort_values(["test_version", "progress_bin"])
        .reset_index(drop=True)
    )


def build_delta_curve(curve: pd.DataFrame, curve_value_name: str) -> pd.DataFrame:
    mean_pivot = curve.pivot(
        index=["progress_bin", "bin_center"],
        columns="test_version",
        values=curve_value_name,
    ).rename(columns={1: "version_A", 2: "version_B"})

    sd_pivot = curve.pivot(
        index=["progress_bin", "bin_center"],
        columns="test_version",
        values="curve_sd",
    ).rename(columns={1: "sd_A", 2: "sd_B"})

    n_pivot = curve.pivot(
        index=["progress_bin", "bin_center"],
        columns="test_version",
        values="n_sessions",
    ).rename(columns={1: "n_A", 2: "n_B"})

    pivot = pd.concat([mean_pivot, sd_pivot, n_pivot], axis=1).reset_index()

    missing = {"version_A", "version_B", "sd_A", "sd_B", "n_A", "n_B"} - set(pivot.columns)
    if missing:
        raise ValueError(f"Could not build A/B delta curve; missing columns: {sorted(missing)}")

    pivot[["sd_A", "sd_B"]] = pivot[["sd_A", "sd_B"]].fillna(0.0)
    pivot["n_A"] = pivot["n_A"].fillna(0).astype(int)
    pivot["n_B"] = pivot["n_B"].fillna(0).astype(int)

    pivot["delta_B_minus_A"] = pivot["version_B"] - pivot["version_A"]
    pivot["se_delta"] = np.sqrt(
        np.where(pivot["n_A"] > 0, (pivot["sd_A"] ** 2) / pivot["n_A"], np.nan)
        + np.where(pivot["n_B"] > 0, (pivot["sd_B"] ** 2) / pivot["n_B"], np.nan)
    )
    pivot["delta_ci_low"] = pivot["delta_B_minus_A"] - 1.96 * pivot["se_delta"]
    pivot["delta_ci_high"] = pivot["delta_B_minus_A"] + 1.96 * pivot["se_delta"]
    return pivot.sort_values("progress_bin").reset_index(drop=True)


def compute_delta_curve_correlation(
    true_delta: pd.DataFrame,
    pred_delta: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    merged_delta = true_delta.merge(
        pred_delta,
        on=["progress_bin", "bin_center"],
        suffixes=("_true", "_pred"),
        how="inner",
    )
    needed = ["delta_B_minus_A_true", "delta_B_minus_A_pred"]
    merged_delta = merged_delta.dropna(subset=needed).copy()

    if len(merged_delta) < 3:
        raise ValueError("Need at least 3 aligned bins to compute Pearson correlation.")

    r, p = pearsonr(merged_delta["delta_B_minus_A_true"], merged_delta["delta_B_minus_A_pred"])
    return merged_delta, float(r), float(p)


def compute_bland_altman_stats(merged_delta: pd.DataFrame) -> dict[str, float]:
    diff = merged_delta["delta_B_minus_A_pred"] - merged_delta["delta_B_minus_A_true"]
    avg = (merged_delta["delta_B_minus_A_pred"] + merged_delta["delta_B_minus_A_true"]) / 2.0

    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    loa_low = mean_diff - 1.96 * sd_diff
    loa_high = mean_diff + 1.96 * sd_diff

    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
        "avg_min": float(avg.min()),
        "avg_max": float(avg.max()),
    }


def build_cumulative_delta_curve(delta_df: pd.DataFrame) -> pd.DataFrame:
    delta_df = delta_df.sort_values("progress_bin").copy()
    if len(delta_df) == 0:
        return delta_df
    step = 1.0 / len(delta_df)
    out = delta_df[["progress_bin", "bin_center", "delta_B_minus_A"]].copy()
    out["cumulative_delta"] = out["delta_B_minus_A"].cumsum() * step
    return out


def write_report(
    merged_rows: pd.DataFrame,
    merged_delta: pd.DataFrame,
    pearson_r: float,
    pearson_p: float,
    ba_stats: dict[str, float],
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
    p("=== DELTA-CURVE PEARSON ===")
    p(f"Aligned bins used = {len(merged_delta)}")
    p(f"Pearson r         = {pearson_r:.6f}")
    p(f"Pearson p-value   = {pearson_p:.6g}")
    p("")
    p("=== BLAND-ALTMAN ON DELTA CURVES ===")
    p("Difference is defined as predicted delta minus true delta.")
    p(f"Mean difference (bias) = {ba_stats['mean_diff']:.6f}")
    p(f"SD of difference       = {ba_stats['sd_diff']:.6f}")
    p(f"Lower 95% LoA          = {ba_stats['loa_low']:.6f}")
    p(f"Upper 95% LoA          = {ba_stats['loa_high']:.6f}")
    p("")
    p("=== ALIGNED DELTA BINS ===")
    for _, row in merged_delta.iterrows():
        p(
            f"bin {int(row['progress_bin']):2d} (center={row['bin_center']:.2f}): "
            f"true_delta={row['delta_B_minus_A_true']:.6f}, "
            f"pred_delta={row['delta_B_minus_A_pred']:.6f}, "
            f"pred_minus_true={row['delta_B_minus_A_pred'] - row['delta_B_minus_A_true']:.6f}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())


def save_delta_curves_plot(true_delta: pd.DataFrame, pred_delta: pd.DataFrame, output_path: str, show_plots: bool) -> None:
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
        plt.annotate(
            str(int(row["progress_bin"])),
            (row["delta_B_minus_A_true"], row["delta_B_minus_A_pred"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

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


def save_bland_altman_plot(
    merged_delta: pd.DataFrame,
    ba_stats: dict[str, float],
    output_path: str,
    show_plots: bool,
) -> None:
    avg = (merged_delta["delta_B_minus_A_pred"] + merged_delta["delta_B_minus_A_true"]) / 2.0
    diff = merged_delta["delta_B_minus_A_pred"] - merged_delta["delta_B_minus_A_true"]

    plt.figure(figsize=(8, 6))
    plt.scatter(avg, diff, alpha=0.9)

    for _, row in merged_delta.iterrows():
        x = (row["delta_B_minus_A_pred"] + row["delta_B_minus_A_true"]) / 2.0
        y = row["delta_B_minus_A_pred"] - row["delta_B_minus_A_true"]
        plt.annotate(
            str(int(row["progress_bin"])),
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    plt.axhline(ba_stats["mean_diff"], linestyle="--", linewidth=2, label="Mean diff")
    plt.axhline(ba_stats["loa_low"], linestyle=":", linewidth=2, label="95% LoA")
    plt.axhline(ba_stats["loa_high"], linestyle=":", linewidth=2)

    plt.xlabel("Average of True and Predicted Delta")
    plt.ylabel("Predicted Delta - True Delta")
    plt.title("Bland–Altman Plot for A/B Delta Curves")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def save_session_spaghetti_plot(
    true_session_bin: pd.DataFrame,
    pred_session_bin: pd.DataFrame,
    true_curve: pd.DataFrame,
    pred_curve: pd.DataFrame,
    output_path: str,
    show_plots: bool,
) -> None:
    all_y = np.concatenate([
        true_session_bin["session_bin_true"].to_numpy(),
        pred_session_bin["session_bin_pred"].to_numpy(),
        true_curve["curve_true"].to_numpy(),
        pred_curve["curve_pred"].to_numpy(),
    ])
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    panel_specs = [
        (axes[0, 0], true_session_bin, true_curve, 1, "session_bin_true", "curve_true", "True Version A"),
        (axes[0, 1], true_session_bin, true_curve, 2, "session_bin_true", "curve_true", "True Version B"),
        (axes[1, 0], pred_session_bin, pred_curve, 1, "session_bin_pred", "curve_pred", "Predicted Version A"),
        (axes[1, 1], pred_session_bin, pred_curve, 2, "session_bin_pred", "curve_pred", "Predicted Version B"),
    ]

    for ax, session_df, curve_df, version, session_col, curve_col, title in panel_specs:
        sessions = session_df[session_df["test_version"] == version].copy()
        curve = curve_df[curve_df["test_version"] == version].copy()

        for _, sdf in sessions.groupby("session_id"):
            ax.plot(
                sdf["bin_center"],
                sdf[session_col],
                linewidth=1.0,
                alpha=0.35,
            )

        ax.plot(
            curve["bin_center"],
            curve[curve_col],
            linewidth=3.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Normalized Trial Progress")
        ax.set_ylabel("Performance")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def save_session_heatmaps(
    true_session_bin: pd.DataFrame,
    pred_session_bin: pd.DataFrame,
    output_path: str,
    show_plots: bool,
) -> None:
    def prep_heatmap(df: pd.DataFrame, version: int, value_col: str) -> pd.DataFrame:
        sub = df[df["test_version"] == version].copy()
        mat = sub.pivot(index="session_id", columns="progress_bin", values=value_col)
        sort_order = mat.mean(axis=1).sort_values().index
        return mat.loc[sort_order]

    mats = [
        prep_heatmap(true_session_bin, 1, "session_bin_true"),
        prep_heatmap(true_session_bin, 2, "session_bin_true"),
        prep_heatmap(pred_session_bin, 1, "session_bin_pred"),
        prep_heatmap(pred_session_bin, 2, "session_bin_pred"),
    ]

    all_vals = np.concatenate([m.to_numpy().ravel() for m in mats])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
    panel_specs = [
        (axes[0, 0], mats[0], "True Version A"),
        (axes[0, 1], mats[1], "True Version B"),
        (axes[1, 0], mats[2], "Predicted Version A"),
        (axes[1, 1], mats[3], "Predicted Version B"),
    ]

    last_im = None
    for ax, mat, title in panel_specs:
        last_im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Progress Bin")
        ax.set_ylabel("Sessions")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(c) for c in mat.columns])

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Performance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def save_delta_ribbon_plot(
    true_delta: pd.DataFrame,
    pred_delta: pd.DataFrame,
    output_path: str,
    show_plots: bool,
) -> None:
    plt.figure(figsize=(9, 6))

    x_true = true_delta["bin_center"].to_numpy()
    y_true = true_delta["delta_B_minus_A"].to_numpy()
    low_true = true_delta["delta_ci_low"].to_numpy()
    high_true = true_delta["delta_ci_high"].to_numpy()

    x_pred = pred_delta["bin_center"].to_numpy()
    y_pred = pred_delta["delta_B_minus_A"].to_numpy()
    low_pred = pred_delta["delta_ci_low"].to_numpy()
    high_pred = pred_delta["delta_ci_high"].to_numpy()

    plt.plot(x_true, y_true, linewidth=2.5, label="True B - A")
    plt.fill_between(x_true, low_true, high_true, alpha=0.2)
    plt.plot(x_pred, y_pred, linewidth=2.5, label="Predicted B - A")
    plt.fill_between(x_pred, low_pred, high_pred, alpha=0.2)
    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Normalized Trial Progress")
    plt.ylabel("B - A Delta")
    plt.title("Difference-only Ribbon Plot (95% CI)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()


def save_cumulative_delta_plot(
    true_cum: pd.DataFrame,
    pred_cum: pd.DataFrame,
    output_path: str,
    show_plots: bool,
) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(true_cum["bin_center"], true_cum["cumulative_delta"], marker="o", linewidth=2.5, label="True cumulative (B - A)")
    plt.plot(pred_cum["bin_center"], pred_cum["cumulative_delta"], marker="o", linewidth=2.5, label="Pred cumulative (B - A)")
    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Normalized Trial Progress")
    plt.ylabel("Cumulative Delta")
    plt.title("Cumulative Difference Plot")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
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

    true_session_bin = build_session_bin_curve(merged, "performance", "session_bin_true")
    pred_session_bin = build_session_bin_curve(merged, "y_pred", "session_bin_pred")

    true_curve = build_version_curve(true_session_bin, "session_bin_true", "curve_true")
    pred_curve = build_version_curve(pred_session_bin, "session_bin_pred", "curve_pred")

    true_delta = build_delta_curve(true_curve, "curve_true")
    pred_delta = build_delta_curve(pred_curve, "curve_pred")
    merged_delta, pearson_r, pearson_p = compute_delta_curve_correlation(true_delta, pred_delta)
    ba_stats = compute_bland_altman_stats(merged_delta)

    true_cumulative = build_cumulative_delta_curve(true_delta)
    pred_cumulative = build_cumulative_delta_curve(pred_delta)

    merged_path = os.path.join(args.output_dir, "merged_prediction_truth_rows.csv")
    true_session_bin_path = os.path.join(args.output_dir, "true_session_bin_curve.csv")
    pred_session_bin_path = os.path.join(args.output_dir, "predicted_session_bin_curve.csv")
    true_curve_path = os.path.join(args.output_dir, "true_version_curve.csv")
    pred_curve_path = os.path.join(args.output_dir, "predicted_version_curve.csv")
    delta_curve_path = os.path.join(args.output_dir, "ab_delta_curve.csv")
    delta_ribbon_curve_path = os.path.join(args.output_dir, "ab_delta_ribbon_curve.csv")
    delta_cumulative_curve_path = os.path.join(args.output_dir, "ab_delta_cumulative_curve.csv")
    report_path = os.path.join(args.output_dir, "pearson_ab_delta_report.txt")
    curves_plot_path = os.path.join(args.output_dir, "ab_delta_curves.png")
    scatter_plot_path = os.path.join(args.output_dir, "ab_delta_scatter.png")
    bland_altman_path = os.path.join(args.output_dir, "ab_delta_bland_altman.png")
    spaghetti_path = os.path.join(args.output_dir, "ab_session_spaghetti.png")
    heatmap_path = os.path.join(args.output_dir, "ab_session_heatmaps.png")
    delta_ribbon_path = os.path.join(args.output_dir, "ab_delta_ribbon.png")
    cumulative_path = os.path.join(args.output_dir, "ab_delta_cumulative.png")

    merged.to_csv(merged_path, index=False)
    true_session_bin.to_csv(true_session_bin_path, index=False)
    pred_session_bin.to_csv(pred_session_bin_path, index=False)
    true_curve.to_csv(true_curve_path, index=False)
    pred_curve.to_csv(pred_curve_path, index=False)
    merged_delta.to_csv(delta_curve_path, index=False)

    ribbon_export = merged_delta[[
        "progress_bin", "bin_center",
        "delta_B_minus_A_true", "delta_ci_low_true", "delta_ci_high_true",
        "delta_B_minus_A_pred", "delta_ci_low_pred", "delta_ci_high_pred",
    ]].copy()
    ribbon_export.to_csv(delta_ribbon_curve_path, index=False)

    cumulative_export = true_cumulative.merge(
        pred_cumulative,
        on=["progress_bin", "bin_center"],
        suffixes=("_true", "_pred"),
        how="outer",
    )
    cumulative_export.to_csv(delta_cumulative_curve_path, index=False)

    write_report(
        merged_rows=merged,
        merged_delta=merged_delta,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        ba_stats=ba_stats,
        report_path=report_path,
        n_bins=args.n_bins,
    )
    save_delta_curves_plot(true_delta, pred_delta, curves_plot_path, args.show_plots)
    save_delta_scatter_plot(merged_delta, pearson_r, pearson_p, scatter_plot_path, args.show_plots)
    save_bland_altman_plot(merged_delta, ba_stats, bland_altman_path, args.show_plots)
    save_session_spaghetti_plot(
        true_session_bin,
        pred_session_bin,
        true_curve,
        pred_curve,
        spaghetti_path,
        args.show_plots,
    )
    save_session_heatmaps(true_session_bin, pred_session_bin, heatmap_path, args.show_plots)
    save_delta_ribbon_plot(true_delta, pred_delta, delta_ribbon_path, args.show_plots)
    save_cumulative_delta_plot(true_cumulative, pred_cumulative, cumulative_path, args.show_plots)

    print(f"Saved merged rows to '{merged_path}'.")
    print(f"Saved true session-bin curve to '{true_session_bin_path}'.")
    print(f"Saved predicted session-bin curve to '{pred_session_bin_path}'.")
    print(f"Saved true version curve to '{true_curve_path}'.")
    print(f"Saved predicted version curve to '{pred_curve_path}'.")
    print(f"Saved aligned delta curve to '{delta_curve_path}'.")
    print(f"Saved ribbon delta curve CSV to '{delta_ribbon_curve_path}'.")
    print(f"Saved cumulative delta curve CSV to '{delta_cumulative_curve_path}'.")
    print(f"Saved report to '{report_path}'.")
    print(f"Saved curve plot to '{curves_plot_path}'.")
    print(f"Saved delta scatter plot to '{scatter_plot_path}'.")
    print(f"Saved Bland-Altman plot to '{bland_altman_path}'.")
    print(f"Saved session spaghetti plot to '{spaghetti_path}'.")
    print(f"Saved session heatmaps to '{heatmap_path}'.")
    print(f"Saved delta ribbon plot to '{delta_ribbon_path}'.")
    print(f"Saved cumulative delta plot to '{cumulative_path}'.")


if __name__ == "__main__":
    main()
