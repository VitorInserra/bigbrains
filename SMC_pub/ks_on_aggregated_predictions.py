#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


DEFAULT_CSV = Path("outputs/smc_nested_cv_a100/combined_outer_fold_predictions.csv")


def require_columns(df: pd.DataFrame, required: set[str], csv_path: Path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")


def summarize_array(name: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{name}_n": int(len(values)),
        f"{name}_mean": float(np.mean(values)),
        f"{name}_median": float(np.median(values)),
        f"{name}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{name}_min": float(np.min(values)),
        f"{name}_max": float(np.max(values)),
    }


def paired_error_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    return {
        "mean_signed_error": float(np.mean(err)),
        "median_signed_error": float(np.median(err)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "max_abs_error": float(np.max(abs_err)),
        "pearson_corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan,
    }


def ecdf_gap_table(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    sample_a = np.sort(np.asarray(sample_a, dtype=float))
    sample_b = np.sort(np.asarray(sample_b, dtype=float))

    grid = np.sort(np.unique(np.concatenate([sample_a, sample_b])))
    ecdf_a = np.searchsorted(sample_a, grid, side="right") / len(sample_a)
    ecdf_b = np.searchsorted(sample_b, grid, side="right") / len(sample_b)
    signed_gap = ecdf_a - ecdf_b
    abs_gap = np.abs(signed_gap)

    idx = int(np.argmax(abs_gap))
    ks_x = float(grid[idx])
    ks_signed_gap = float(signed_gap[idx])
    ks_abs_gap = float(abs_gap[idx])

    detail = pd.DataFrame(
        {
            "x": grid,
            "ecdf_true": ecdf_a,
            "ecdf_pred": ecdf_b,
            "signed_gap_true_minus_pred": signed_gap,
            "abs_gap": abs_gap,
        }
    )

    summary = {
        "ks_x": ks_x,
        "ks_signed_gap_true_minus_pred": ks_signed_gap,
        "ks_abs_gap": ks_abs_gap,
        "ecdf_true_at_ks_x": float(ecdf_a[idx]),
        "ecdf_pred_at_ks_x": float(ecdf_b[idx]),
    }
    return detail, summary


def write_text_report(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def plot_ecdf(detail_df: pd.DataFrame, title: str, out_path: Path) -> None:
    ks_row = detail_df.loc[detail_df["abs_gap"].idxmax()]

    plt.figure(figsize=(10, 6))
    plt.step(detail_df["x"], detail_df["ecdf_true"], where="post", label="True")
    plt.step(detail_df["x"], detail_df["ecdf_pred"], where="post", label="Predicted")
    plt.axvline(ks_row["x"], linestyle="--", linewidth=2, label=f"Max KS gap @ {ks_row['x']:.3f}")
    plt.title(title)
    plt.xlabel("Performance value")
    plt.ylabel("Empirical CDF")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist(true_vals: np.ndarray, pred_vals: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(true_vals, bins=20, alpha=0.6, edgecolor="black", label="True")
    plt.hist(pred_vals, bins=20, alpha=0.6, edgecolor="black", label="Predicted")
    plt.axvline(np.median(true_vals), linestyle="--", linewidth=2, label="True median")
    plt.axvline(np.median(pred_vals), linestyle="--", linewidth=2, label="Pred median")
    plt.title(title)
    plt.xlabel("Performance value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sorted_overlay(true_vals: np.ndarray, pred_vals: np.ndarray, title: str, out_path: Path) -> None:
    true_sorted = np.sort(np.asarray(true_vals, dtype=float))
    pred_sorted = np.sort(np.asarray(pred_vals, dtype=float))

    n = min(len(true_sorted), len(pred_sorted))
    x = np.arange(n)

    plt.figure(figsize=(12, 8))
    plt.plot(x, true_sorted[:n], label="True", linewidth=2)
    plt.plot(x, pred_sorted[:n], label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Sorted Trial Index")
    plt.ylabel("Performance (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_session_sorted_overlay(df: pd.DataFrame, out_path: Path, sort_sessions_by: str = "session_true_median") -> None:
    require_columns(df, {"session_id", "y_true", "y_pred"}, out_path)

    session_blocks = []
    session_meta = []

    grouped = (
        df.groupby("session_id", as_index=False)
        .agg(
            session_true_median=("y_true", "median"),
            session_pred_median=("y_pred", "median"),
            n_trials=("y_true", "size"),
        )
        .sort_values(sort_sessions_by, kind="stable")
        .reset_index(drop=True)
    )

    for _, row in grouped.iterrows():
        sid = row["session_id"]
        sdf = df.loc[df["session_id"] == sid, ["y_true", "y_pred"]].dropna().copy()
        sdf = sdf.sort_values("y_true", kind="stable").reset_index(drop=True)
        sdf["global_index"] = np.arange(len(sdf))
        session_blocks.append(sdf)
        session_meta.append(
            {
                "session_id": sid,
                "start_idx": int(sum(len(block) for block in session_blocks[:-1])),
                "end_idx": int(sum(len(block) for block in session_blocks) - 1),
                "n_trials": int(row["n_trials"]),
                "session_true_median": float(row["session_true_median"]),
                "session_pred_median": float(row["session_pred_median"]),
            }
        )

    if not session_blocks:
        return

    combined = pd.concat(session_blocks, ignore_index=True)
    combined["global_index"] = np.arange(len(combined))

    plt.figure(figsize=(16, 8))
    plt.plot(combined["global_index"], combined["y_true"], label="True", linewidth=2)
    plt.plot(combined["global_index"], combined["y_pred"], label="Predicted", linewidth=2)

    for meta in session_meta[:-1]:
        plt.axvline(meta["end_idx"] + 0.5, linestyle="--", linewidth=1, alpha=0.5)

    plt.title("All Sessions: Predicted vs True Performance (within-session sorted by true value)")
    plt.xlabel("Concatenated Trial Index Across Sessions")
    plt.ylabel("Performance (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    pd.DataFrame(session_meta).to_csv(out_path.with_suffix(".csv"), index=False)


def build_quantile_error_table(df: pd.DataFrame, q: int = 10) -> pd.DataFrame:
    work = df[["y_true", "y_pred"]].dropna().copy()
    work["signed_error"] = work["y_pred"] - work["y_true"]
    work["abs_error"] = work["signed_error"].abs()

    work["true_quantile_bin"] = pd.qcut(work["y_true"], q=q, duplicates="drop")

    grouped = (
        work.groupby("true_quantile_bin", observed=True)
        .agg(
            n=("y_true", "size"),
            mean_true=("y_true", "mean"),
            median_true=("y_true", "median"),
            mean_pred=("y_pred", "mean"),
            median_pred=("y_pred", "median"),
            mean_signed_error=("signed_error", "mean"),
            median_signed_error=("signed_error", "median"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )

    grouped["bin_left"] = grouped["true_quantile_bin"].apply(lambda iv: float(iv.left))
    grouped["bin_right"] = grouped["true_quantile_bin"].apply(lambda iv: float(iv.right))
    grouped["bin_label"] = grouped["true_quantile_bin"].astype(str)
    return grouped[
        [
            "bin_label",
            "bin_left",
            "bin_right",
            "n",
            "mean_true",
            "median_true",
            "mean_pred",
            "median_pred",
            "mean_signed_error",
            "median_signed_error",
            "mean_abs_error",
            "max_abs_error",
        ]
    ]


def analyze_pair(true_vals: np.ndarray, pred_vals: np.ndarray, label: str, out_dir: Path) -> dict[str, float]:
    ks_stat, ks_p = ks_2samp(true_vals, pred_vals, alternative="two-sided")
    ecdf_df, ecdf_summary = ecdf_gap_table(true_vals, pred_vals)

    summary = {
        "analysis": label,
        **summarize_array("true", true_vals),
        **summarize_array("pred", pred_vals),
        **paired_error_summary(true_vals, pred_vals),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        **ecdf_summary,
    }

    ecdf_df.to_csv(out_dir / f"{label}_ecdf_profile.csv", index=False)
    plot_ecdf(ecdf_df, f"{label.replace('_', ' ').title()}: True vs Predicted ECDF", out_dir / f"{label}_ecdf.png")
    plot_hist(true_vals, pred_vals, f"{label.replace('_', ' ').title()}: True vs Predicted", out_dir / f"{label}_hist.png")
    plot_sorted_overlay(
        true_vals,
        pred_vals,
        f"{label.replace('_', ' ').title()}: Predicted vs True Performance",
        out_dir / f"{label}_sorted_overlay.png",
    )

    return summary


def trim_by_true_quantiles(df: pd.DataFrame, lower_q: float, upper_q: float) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.copy()
    y_true = work["y_true"].dropna().to_numpy(dtype=float)
    low = float(np.quantile(y_true, lower_q))
    high = float(np.quantile(y_true, upper_q))

    keep_mask = work["y_true"].between(low, high, inclusive="both")
    trimmed = work.loc[keep_mask].copy()

    summary = {
        "rows_before": int(len(work)),
        "rows_after": int(len(trimmed)),
        "rows_removed": int(len(work) - len(trimmed)),
        "lower_q": float(lower_q),
        "upper_q": float(upper_q),
        "keep_low": low,
        "keep_high": high,
    }
    return trimmed, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KS analysis on aggregated true vs predicted outer-fold results.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        help="Path to combined_outer_fold_predictions.csv",
    )
    parser.add_argument("--lower-q", type=float, default=0.005, help="Lower quantile for trimming by y_true")
    parser.add_argument("--upper-q", type=float, default=0.995, help="Upper quantile for trimming by y_true")
    parser.add_argument("--no-trim", action="store_true", help="Disable outlier trimming")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    require_columns(df, {"y_true", "y_pred"}, csv_path)

    out_dir = csv_path.parent
    report_lines: list[str] = []

    if args.no_trim:
        analysis_df = df.copy()
        trim_summary = {
            "rows_before": int(len(df)),
            "rows_after": int(len(df)),
            "rows_removed": 0,
            "lower_q": np.nan,
            "upper_q": np.nan,
            "keep_low": float(np.nanmin(df["y_true"])),
            "keep_high": float(np.nanmax(df["y_true"])),
        }
    else:
        analysis_df, trim_summary = trim_by_true_quantiles(df, args.lower_q, args.upper_q)

    analysis_df.to_csv(out_dir / "combined_outer_fold_predictions_trimmed.csv", index=False)
    pd.DataFrame([trim_summary]).to_csv(out_dir / "trim_summary.csv", index=False)

    report_lines.append("=== Trimming summary ===")
    report_lines.append(f"rows_before               = {trim_summary['rows_before']}")
    report_lines.append(f"rows_after                = {trim_summary['rows_after']}")
    report_lines.append(f"rows_removed              = {trim_summary['rows_removed']}")
    report_lines.append(f"lower_q                   = {trim_summary['lower_q']}")
    report_lines.append(f"upper_q                   = {trim_summary['upper_q']}")
    report_lines.append(f"keep_low                  = {trim_summary['keep_low']:.6f}")
    report_lines.append(f"keep_high                 = {trim_summary['keep_high']:.6f}")
    report_lines.append("")

    trial_true = analysis_df["y_true"].dropna().to_numpy(dtype=float)
    trial_pred = analysis_df["y_pred"].dropna().to_numpy(dtype=float)
    trial_summary = analyze_pair(trial_true, trial_pred, "trial_level", out_dir)

    report_lines.append("=== Trial-level KS analysis: all aggregated out-of-fold predictions ===")
    report_lines.append(f"n_true                    = {trial_summary['true_n']}")
    report_lines.append(f"n_pred                    = {trial_summary['pred_n']}")
    report_lines.append(f"true_mean                 = {trial_summary['true_mean']:.6f}")
    report_lines.append(f"pred_mean                 = {trial_summary['pred_mean']:.6f}")
    report_lines.append(f"true_median               = {trial_summary['true_median']:.6f}")
    report_lines.append(f"pred_median               = {trial_summary['pred_median']:.6f}")
    report_lines.append(f"MAE                       = {trial_summary['mae']:.6f}")
    report_lines.append(f"RMSE                      = {trial_summary['rmse']:.6f}")
    report_lines.append(f"mean_signed_error         = {trial_summary['mean_signed_error']:.6f}")
    report_lines.append(f"pearson_corr              = {trial_summary['pearson_corr']:.6f}")
    report_lines.append(f"KS statistic              = {trial_summary['ks_statistic']:.6f}")
    report_lines.append(f"KS p-value                = {trial_summary['ks_p_value']:.6g}")
    report_lines.append(f"Max ECDF gap at x         = {trial_summary['ks_x']:.6f}")
    report_lines.append(
        f"Signed gap (true - pred)  = {trial_summary['ks_signed_gap_true_minus_pred']:.6f}"
    )
    report_lines.append(
        "Interpretation: if this signed gap is positive, the true distribution accumulates faster "
        "than the predicted distribution at that threshold. If negative, the predicted distribution "
        "accumulates faster there."
    )
    report_lines.append("")

    if "session_id" in analysis_df.columns:
        session_summary_df = (
            analysis_df.groupby("session_id", as_index=False)
            .agg(
                true_session_median=("y_true", "median"),
                pred_session_median=("y_pred", "median"),
                true_session_mean=("y_true", "mean"),
                pred_session_mean=("y_pred", "mean"),
                n_trials=("y_true", "size"),
            )
        )
        session_summary_df.to_csv(out_dir / "session_level_true_vs_pred_summary.csv", index=False)

        session_true = session_summary_df["true_session_median"].to_numpy(dtype=float)
        session_pred = session_summary_df["pred_session_median"].to_numpy(dtype=float)
        session_summary = analyze_pair(session_true, session_pred, "session_median_level", out_dir)

        report_lines.append("=== Session-level KS analysis: median per session ===")
        report_lines.append(f"n_sessions                = {len(session_summary_df)}")
        report_lines.append(f"true_session_mean         = {session_summary['true_mean']:.6f}")
        report_lines.append(f"pred_session_mean         = {session_summary['pred_mean']:.6f}")
        report_lines.append(f"true_session_median       = {session_summary['true_median']:.6f}")
        report_lines.append(f"pred_session_median       = {session_summary['pred_median']:.6f}")
        report_lines.append(f"session_MAE               = {session_summary['mae']:.6f}")
        report_lines.append(f"session_RMSE              = {session_summary['rmse']:.6f}")
        report_lines.append(f"session_KS statistic      = {session_summary['ks_statistic']:.6f}")
        report_lines.append(f"session_KS p-value        = {session_summary['ks_p_value']:.6g}")
        report_lines.append(f"session Max ECDF gap at x = {session_summary['ks_x']:.6f}")
        report_lines.append("")

        plot_per_session_sorted_overlay(
            analysis_df,
            out_dir / "all_sessions_sorted_true_vs_pred.png",
            sort_sessions_by="session_true_median",
        )
    else:
        session_summary_df = None
        report_lines.append("session_id column not found, so session-level KS was skipped.")
        report_lines.append("")

    if "outer_fold" in analysis_df.columns:
        per_fold_rows = []
        for fold, fold_df in analysis_df.groupby("outer_fold"):
            fold_true = fold_df["y_true"].to_numpy(dtype=float)
            fold_pred = fold_df["y_pred"].to_numpy(dtype=float)
            summary = analyze_pair(fold_true, fold_pred, f"outer_fold_{fold}_trial_level", out_dir)
            per_fold_rows.append(summary)

        per_fold_df = pd.DataFrame(per_fold_rows).sort_values("analysis")
        per_fold_df.to_csv(out_dir / "per_fold_ks_summary.csv", index=False)
        report_lines.append("=== Per-fold trial-level KS summary ===")
        for _, row in per_fold_df.iterrows():
            report_lines.append(
                f"{row['analysis']}: KS={row['ks_statistic']:.6f}, p={row['ks_p_value']:.6g}, "
                f"MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, x*={row['ks_x']:.6f}"
            )
        report_lines.append("")

    quantile_table = build_quantile_error_table(analysis_df, q=10)
    quantile_table.to_csv(out_dir / "prediction_error_by_true_quantile.csv", index=False)

    report_lines.append("=== Why this helps interpret KS ===")
    report_lines.append(
        "KS compares the overall true and predicted distributions after sorting, so it does not preserve "
        "which prediction belonged to which trial. That means KS is good for checking distributional mismatch, "
        "but not pointwise prediction accuracy."
    )
    report_lines.append(
        "To see how the KS divergence relates to the predictions themselves, also inspect "
        "prediction_error_by_true_quantile.csv. That file shows where the model is over- or under-predicting "
        "across the true-value range."
    )
    report_lines.append("")
    report_lines.append("Saved outputs:")
    report_lines.append("- combined_outer_fold_predictions_trimmed.csv")
    report_lines.append("- trim_summary.csv")
    report_lines.append("- trial_level_ecdf_profile.csv")
    report_lines.append("- trial_level_ecdf.png")
    report_lines.append("- trial_level_hist.png")
    report_lines.append("- trial_level_sorted_overlay.png")
    report_lines.append("- all_sessions_sorted_true_vs_pred.png        (if session_id exists)")
    report_lines.append("- all_sessions_sorted_true_vs_pred.csv        (session boundaries, if session_id exists)")
    report_lines.append("- session_level_true_vs_pred_summary.csv      (if session_id exists)")
    report_lines.append("- session_median_level_ecdf_profile.csv       (if session_id exists)")
    report_lines.append("- session_median_level_ecdf.png               (if session_id exists)")
    report_lines.append("- session_median_level_sorted_overlay.png     (if session_id exists)")
    report_lines.append("- per_fold_ks_summary.csv                     (if outer_fold exists)")
    report_lines.append("- prediction_error_by_true_quantile.csv")

    write_text_report(out_dir / "ks_test_report.txt", "\n".join(report_lines))
    pd.DataFrame([trial_summary]).to_csv(out_dir / "trial_level_ks_summary.csv", index=False)

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
