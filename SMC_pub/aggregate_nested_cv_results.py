#!/usr/bin/env python3
from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



ROOT_DIR = Path("outputs/smc_nested_cv_a100")

def parse_run_summary(summary_path: Path) -> dict[str, Any]:
    """Parse fold/session/metric info from a run_summary.txt file."""
    text = summary_path.read_text(encoding="utf-8")

    fold_match = re.search(r"OUTER FOLD\s+(\d+)\/", text)
    fold = int(fold_match.group(1)) if fold_match else None

    train_match = re.search(
        r"Outer train sessions:\s*(\[[\s\S]*?\])\s*Outer test sessions",
        text,
    )
    test_match = re.search(
        r"Outer test sessions\s*:\s*(\[[\s\S]*?\])\s*Outer train trials",
        text,
    )

    train_sessions = re.findall(r"'([^']+)'", train_match.group(1)) if train_match else []
    test_sessions = re.findall(r"'([^']+)'", test_match.group(1)) if test_match else []

    metrics: dict[str, float] = {}
    for key in ["mse", "rmse", "mae", "r2"]:
        metric_match = re.search(
            rf"test_{key}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text
        )
        if metric_match:
            metrics[f"test_{key}"] = float(metric_match.group(1))

    return {
        "outer_fold": fold,
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        **metrics,
    }



def compute_regression_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }



def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {ROOT_DIR.resolve()}")

    outer_dirs = sorted(
        [p for p in ROOT_DIR.glob("outer_fold_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not outer_dirs:
        raise FileNotFoundError(f"No outer_fold_* directories found under {ROOT_DIR}")

    all_predictions: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print(f"Found {len(outer_dirs)} outer folds under {ROOT_DIR}\n")

    for outer_dir in outer_dirs:
        fold_from_dir = int(outer_dir.name.split("_")[-1])
        pred_path = outer_dir / "outer_fold_predictions.csv"
        summary_path = outer_dir / "run_summary.txt"

        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing run summary file: {summary_path}")

        pred_df = pd.read_csv(pred_path)
        required_cols = {"trial_id", "session_id", "y_true", "y_pred"}
        missing = required_cols - set(pred_df.columns)
        if missing:
            raise ValueError(f"{pred_path} is missing required columns: {sorted(missing)}")

        pred_df = pred_df.copy()
        pred_df["outer_fold"] = fold_from_dir
        pred_df["source_dir"] = outer_dir.name
        pred_df["session_trial_key"] = (
            pred_df["session_id"].astype(str) + "__" + pred_df["trial_id"].astype(str)
        )
        pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
        all_predictions.append(pred_df)

        summary_info = parse_run_summary(summary_path)
        summary_rows.append(summary_info)

        computed_metrics = compute_regression_metrics(pred_df)
        reported_metrics = {
            k: summary_info.get(f"test_{k}") for k in ["mse", "rmse", "mae", "r2"]
        }

        fold_reports.append(
            {
                "outer_fold": fold_from_dir,
                "n_rows": len(pred_df),
                "n_unique_trials": pred_df["trial_id"].nunique(),
                "n_unique_sessions": pred_df["session_id"].nunique(),
                **{f"computed_{k}": v for k, v in computed_metrics.items()},
                **{f"reported_{k}": v for k, v in reported_metrics.items()},
            }
        )

    combined = pd.concat(all_predictions, ignore_index=True)
    combined = combined.sort_values(["outer_fold", "session_id", "trial_id"]).reset_index(drop=True)

    # =========================
    # Leakage checks
    # =========================
    leakage_messages: list[str] = []

    # 1) Same trial predicted in more than one outer fold.
    dup_trial = (
        combined.groupby("trial_id")["outer_fold"].nunique().reset_index(name="n_outer_folds")
    )
    dup_trial = dup_trial[dup_trial["n_outer_folds"] > 1]
    if len(dup_trial) == 0:
        leakage_messages.append("OK: no trial_id appears in more than one outer fold.")
    else:
        leakage_messages.append(
            f"WARNING: {len(dup_trial)} trial_id values appear in more than one outer fold."
        )

    # 2) Same session-trial pair predicted in more than one outer fold.
    dup_session_trial = (
        combined.groupby("session_trial_key")["outer_fold"]
        .nunique()
        .reset_index(name="n_outer_folds")
    )
    dup_session_trial = dup_session_trial[dup_session_trial["n_outer_folds"] > 1]
    if len(dup_session_trial) == 0:
        leakage_messages.append(
            "OK: no (session_id, trial_id) pair appears in more than one outer fold."
        )
    else:
        leakage_messages.append(
            f"WARNING: {len(dup_session_trial)} (session_id, trial_id) pairs appear in more than one outer fold."
        )

    # 3) Same session used as outer-test in more than one fold.
    session_fold_counts = (
        combined.groupby("session_id")["outer_fold"].nunique().reset_index(name="n_outer_folds")
    )
    repeated_test_sessions = session_fold_counts[session_fold_counts["n_outer_folds"] > 1]
    if len(repeated_test_sessions) == 0:
        leakage_messages.append(
            "OK: every session_id appears in the outer-test predictions of only one fold."
        )
    else:
        leakage_messages.append(
            f"WARNING: {len(repeated_test_sessions)} session_id values appear in the outer-test predictions of multiple folds."
        )

    # 4) Train/test overlap within each fold from run_summary.
    for row in summary_rows:
        fold = row["outer_fold"]
        train_sessions = set(row.get("train_sessions", []))
        test_sessions = set(row.get("test_sessions", []))
        overlap = sorted(train_sessions & test_sessions)
        if overlap:
            leakage_messages.append(
                f"WARNING: fold {fold} has train/test session overlap: {overlap}"
            )
        else:
            leakage_messages.append(f"OK: fold {fold} has no train/test session overlap.")

    # 5) Test-session overlap across folds from run_summary.
    summary_df = pd.DataFrame(summary_rows).sort_values("outer_fold")
    for (_, row_a), (_, row_b) in itertools.combinations(summary_df.iterrows(), 2):
        fold_a = int(row_a["outer_fold"])
        fold_b = int(row_b["outer_fold"])
        overlap = sorted(set(row_a["test_sessions"]) & set(row_b["test_sessions"]))
        if overlap:
            leakage_messages.append(
                f"WARNING: folds {fold_a} and {fold_b} share outer-test sessions: {overlap}"
            )

    # 6) Check that sessions in predictions match sessions reported as test sessions.
    for fold, group in combined.groupby("outer_fold"):
        predicted_sessions = set(group["session_id"].astype(str).unique())
        summary_match = next((r for r in summary_rows if r["outer_fold"] == fold), None)
        reported_test_sessions = set(summary_match["test_sessions"]) if summary_match else set()
        if predicted_sessions == reported_test_sessions:
            leakage_messages.append(
                f"OK: fold {fold} predicted sessions exactly match run_summary outer-test sessions."
            )
        else:
            leakage_messages.append(
                f"WARNING: fold {fold} predicted sessions do not match run_summary test sessions. "
                f"Only in predictions={sorted(predicted_sessions - reported_test_sessions)}, "
                f"only in summary={sorted(reported_test_sessions - predicted_sessions)}"
            )

    # =========================
    # Metrics
    # =========================
    per_fold_metrics = []
    for fold, group in combined.groupby("outer_fold"):
        metrics = compute_regression_metrics(group)
        per_fold_metrics.append(
            {
                "outer_fold": fold,
                "n_trials": len(group),
                "n_sessions": group["session_id"].nunique(),
                **metrics,
            }
        )

    per_fold_metrics_df = pd.DataFrame(per_fold_metrics).sort_values("outer_fold")
    metric_cols = ["mse", "rmse", "mae", "r2"]

    mean_metrics = per_fold_metrics_df[metric_cols].mean().rename("mean")
    std_metrics = per_fold_metrics_df[metric_cols].std(ddof=1).rename("std")

    pooled_metrics = pd.Series(compute_regression_metrics(combined), name="pooled_out_of_fold")

    # =========================
    # Save outputs
    # =========================
    combined.to_csv(ROOT_DIR / "combined_outer_fold_predictions.csv", index=False)
    per_fold_metrics_df.to_csv(ROOT_DIR / "aggregated_outer_fold_metrics.csv", index=False)
    pd.DataFrame(fold_reports).sort_values("outer_fold").to_csv(
        ROOT_DIR / "fold_metric_comparison.csv", index=False
    )

    with open(ROOT_DIR / "leakage_report.txt", "w", encoding="utf-8") as f:
        f.write("LEAKAGE CHECKS\n")
        f.write("=" * 80 + "\n")
        for line in leakage_messages:
            f.write(line + "\n")
        f.write("\nPER-FOLD METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(per_fold_metrics_df.to_string(index=False))
        f.write("\n\nMEAN OUTER-FOLD METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(mean_metrics.to_string())
        f.write("\n\nSTD OUTER-FOLD METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(std_metrics.to_string())
        f.write("\n\nPOOLED OUT-OF-FOLD METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(pooled_metrics.to_string())
        f.write("\n")

    # =========================
    # Print summary to terminal
    # =========================
    print("COMBINED PREDICTIONS SAVED TO:")
    print(f"  {ROOT_DIR / 'combined_outer_fold_predictions.csv'}\n")

    print("LEAKAGE CHECKS")
    print("=" * 80)
    for line in leakage_messages:
        print(line)

    print("\nPER-FOLD METRICS")
    print("=" * 80)
    print(per_fold_metrics_df.to_string(index=False))

    print("\nMEAN OUTER-FOLD METRICS")
    print("=" * 80)
    print(mean_metrics.to_string())

    print("\nSTD OUTER-FOLD METRICS")
    print("=" * 80)
    print(std_metrics.to_string())

    print("\nPOOLED OUT-OF-FOLD METRICS")
    print("=" * 80)
    print(pooled_metrics.to_string())

    print("\nOTHER SAVED FILES")
    print("=" * 80)
    print(f"  {ROOT_DIR / 'aggregated_outer_fold_metrics.csv'}")
    print(f"  {ROOT_DIR / 'fold_metric_comparison.csv'}")
    print(f"  {ROOT_DIR / 'leakage_report.txt'}")


if __name__ == "__main__":
    main()
