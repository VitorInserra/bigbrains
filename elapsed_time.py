#!/usr/bin/env python3

from pathlib import Path
import sys
import pandas as pd


INPUT_PATH = Path("feature_table.csv")
OUTPUT_DIR = Path("REPORT_CAMREAD")


def require_columns(df, required):
    missing = set(required) - set(df.columns)
    if missing:
        print(f"Error: missing required columns: {sorted(missing)}")
        print("\nAvailable columns:")
        print(df.columns.tolist())
        sys.exit(1)


def main():
    if not INPUT_PATH.exists():
        print(f"Error: could not find {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)

    require_columns(
        df,
        [
            "session_id",
            "round_id",
            "start_time",
            "end_time",
        ],
    )

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    df = df.dropna(subset=["session_id", "start_time", "end_time"]).copy()

    if df.empty:
        print("Error: no valid rows after parsing start_time and end_time.")
        sys.exit(1)

    # Trial-level elapsed time
    df["trial_elapsed_seconds"] = (
        df["end_time"] - df["start_time"]
    ).dt.total_seconds()

    # Session-level timing
    session_summary = (
        df.groupby("session_id")
        .agg(
            session_start=("start_time", "min"),
            session_end=("end_time", "max"),
            n_trials=("round_id", "count"),
            first_round=("round_id", "min"),
            last_round=("round_id", "max"),
            active_trial_time_seconds=("trial_elapsed_seconds", "sum"),
            mean_trial_time_seconds=("trial_elapsed_seconds", "mean"),
            min_trial_time_seconds=("trial_elapsed_seconds", "min"),
            max_trial_time_seconds=("trial_elapsed_seconds", "max"),
        )
        .reset_index()
    )

    # Wall-clock duration: from first trial start to last trial end
    session_summary["wall_clock_elapsed_seconds"] = (
        session_summary["session_end"] - session_summary["session_start"]
    ).dt.total_seconds()

    session_summary["wall_clock_elapsed_minutes"] = (
        session_summary["wall_clock_elapsed_seconds"] / 60.0
    )

    session_summary["active_trial_time_minutes"] = (
        session_summary["active_trial_time_seconds"] / 60.0
    )

    session_summary["round_span"] = (
        session_summary["last_round"] - session_summary["first_round"] + 1
    )

    session_summary["missing_rounds_estimate"] = (
        session_summary["round_span"] - session_summary["n_trials"]
    )

    session_summary = session_summary.sort_values(
        "wall_clock_elapsed_seconds",
        ascending=True,
    )

    # Longest and shortest sessions by wall-clock time
    shortest_idx = session_summary["wall_clock_elapsed_seconds"].idxmin()
    longest_idx = session_summary["wall_clock_elapsed_seconds"].idxmax()

    longest_shortest_wall_clock = pd.DataFrame(
        [
            {
                "metric": "shortest_session_by_wall_clock",
                **session_summary.loc[shortest_idx].to_dict(),
            },
            {
                "metric": "longest_session_by_wall_clock",
                **session_summary.loc[longest_idx].to_dict(),
            },
        ]
    )

    # Longest and shortest sessions by active task time
    shortest_active_idx = session_summary["active_trial_time_seconds"].idxmin()
    longest_active_idx = session_summary["active_trial_time_seconds"].idxmax()

    longest_shortest_active = pd.DataFrame(
        [
            {
                "metric": "shortest_session_by_active_trial_time",
                **session_summary.loc[shortest_active_idx].to_dict(),
            },
            {
                "metric": "longest_session_by_active_trial_time",
                **session_summary.loc[longest_active_idx].to_dict(),
            },
        ]
    )

    # Performance metric min/max if present
    performance_summary = None

    if "performance_metric" in df.columns:
        df["performance_metric"] = pd.to_numeric(
            df["performance_metric"],
            errors="coerce",
        )

        valid_perf = df.dropna(subset=["performance_metric"]).copy()

        if not valid_perf.empty:
            min_perf_idx = valid_perf["performance_metric"].idxmin()
            max_perf_idx = valid_perf["performance_metric"].idxmax()

            performance_summary = pd.DataFrame(
                [
                    {
                        "metric": "lowest_performance_metric",
                        "performance_metric": valid_perf.loc[min_perf_idx, "performance_metric"],
                        "session_id": valid_perf.loc[min_perf_idx, "session_id"],
                        "round_id": valid_perf.loc[min_perf_idx, "round_id"],
                        "start_time": valid_perf.loc[min_perf_idx, "start_time"],
                        "end_time": valid_perf.loc[min_perf_idx, "end_time"],
                        "trial_elapsed_seconds": valid_perf.loc[min_perf_idx, "trial_elapsed_seconds"],
                        "score": valid_perf.loc[min_perf_idx, "score"] if "score" in valid_perf.columns else None,
                        "test_version": valid_perf.loc[min_perf_idx, "test_version"] if "test_version" in valid_perf.columns else None,
                    },
                    {
                        "metric": "highest_performance_metric",
                        "performance_metric": valid_perf.loc[max_perf_idx, "performance_metric"],
                        "session_id": valid_perf.loc[max_perf_idx, "session_id"],
                        "round_id": valid_perf.loc[max_perf_idx, "round_id"],
                        "start_time": valid_perf.loc[max_perf_idx, "start_time"],
                        "end_time": valid_perf.loc[max_perf_idx, "end_time"],
                        "trial_elapsed_seconds": valid_perf.loc[max_perf_idx, "trial_elapsed_seconds"],
                        "score": valid_perf.loc[max_perf_idx, "score"] if "score" in valid_perf.columns else None,
                        "test_version": valid_perf.loc[max_perf_idx, "test_version"] if "test_version" in valid_perf.columns else None,
                    },
                ]
            )

    # Raw score min/max if present
    raw_score_summary = None

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        valid_score = df.dropna(subset=["score"]).copy()

        if not valid_score.empty:
            min_score_idx = valid_score["score"].idxmin()
            max_score_idx = valid_score["score"].idxmax()

            raw_score_summary = pd.DataFrame(
                [
                    {
                        "metric": "lowest_raw_score",
                        "score": valid_score.loc[min_score_idx, "score"],
                        "session_id": valid_score.loc[min_score_idx, "session_id"],
                        "round_id": valid_score.loc[min_score_idx, "round_id"],
                        "start_time": valid_score.loc[min_score_idx, "start_time"],
                        "end_time": valid_score.loc[min_score_idx, "end_time"],
                        "trial_elapsed_seconds": valid_score.loc[min_score_idx, "trial_elapsed_seconds"],
                        "performance_metric": valid_score.loc[min_score_idx, "performance_metric"]
                        if "performance_metric" in valid_score.columns
                        else None,
                        "test_version": valid_score.loc[min_score_idx, "test_version"]
                        if "test_version" in valid_score.columns
                        else None,
                    },
                    {
                        "metric": "highest_raw_score",
                        "score": valid_score.loc[max_score_idx, "score"],
                        "session_id": valid_score.loc[max_score_idx, "session_id"],
                        "round_id": valid_score.loc[max_score_idx, "round_id"],
                        "start_time": valid_score.loc[max_score_idx, "start_time"],
                        "end_time": valid_score.loc[max_score_idx, "end_time"],
                        "trial_elapsed_seconds": valid_score.loc[max_score_idx, "trial_elapsed_seconds"],
                        "performance_metric": valid_score.loc[max_score_idx, "performance_metric"]
                        if "performance_metric" in valid_score.columns
                        else None,
                        "test_version": valid_score.loc[max_score_idx, "test_version"]
                        if "test_version" in valid_score.columns
                        else None,
                    },
                ]
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session_summary_path = OUTPUT_DIR / "feature_table_session_time_summary.csv"
    longest_shortest_wall_clock_path = OUTPUT_DIR / "feature_table_longest_shortest_wall_clock_sessions.csv"
    longest_shortest_active_path = OUTPUT_DIR / "feature_table_longest_shortest_active_time_sessions.csv"
    trial_elapsed_path = OUTPUT_DIR / "feature_table_trial_elapsed_seconds.csv"

    session_summary.to_csv(session_summary_path, index=False)
    longest_shortest_wall_clock.to_csv(longest_shortest_wall_clock_path, index=False)
    longest_shortest_active.to_csv(longest_shortest_active_path, index=False)

    df[
        [
            "session_id",
            "round_id",
            "start_time",
            "end_time",
            "trial_elapsed_seconds",
        ]
    ].to_csv(trial_elapsed_path, index=False)

    if performance_summary is not None:
        performance_summary_path = OUTPUT_DIR / "feature_table_performance_metric_min_max.csv"
        performance_summary.to_csv(performance_summary_path, index=False)

    if raw_score_summary is not None:
        raw_score_summary_path = OUTPUT_DIR / "feature_table_raw_score_min_max.csv"
        raw_score_summary.to_csv(raw_score_summary_path, index=False)

    print(f"\nLoaded: {INPUT_PATH}")

    print("\n=== Session time summary ===")
    print(session_summary.to_string(index=False))

    print("\n=== Longest and shortest sessions by wall-clock time ===")
    print(longest_shortest_wall_clock.to_string(index=False))

    print("\n=== Longest and shortest sessions by active trial time ===")
    print(longest_shortest_active.to_string(index=False))

    if performance_summary is not None:
        print("\n=== Highest and lowest performance_metric ===")
        print(performance_summary.to_string(index=False))

    if raw_score_summary is not None:
        print("\n=== Highest and lowest raw score ===")
        print(raw_score_summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {session_summary_path}")
    print(f"  {longest_shortest_wall_clock_path}")
    print(f"  {longest_shortest_active_path}")
    print(f"  {trial_elapsed_path}")

    if performance_summary is not None:
        print(f"  {performance_summary_path}")

    if raw_score_summary is not None:
        print(f"  {raw_score_summary_path}")

    print("\nDefinitions:")
    print("  wall_clock_elapsed_seconds = max(end_time) - min(start_time) for each session")
    print("  active_trial_time_seconds  = sum(end_time - start_time) across trials in each session")
    print("  trial_elapsed_seconds      = end_time - start_time for each individual round")


if __name__ == "__main__":
    main()