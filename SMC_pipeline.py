import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from db import get_db
from models.VRDataModel import VRDataModel
from models.EpocXDataModel import EpocXDataModel


# =========================================================
# Configuration
# =========================================================
CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

BANDS = ["theta", "alpha", "beta_l", "beta_h", "gamma"]

EEG_COLUMNS = [
    f"{channel.lower()}_{band.lower()}"
    for channel in CHANNELS
    for band in BANDS
]

START_TOLERANCE_SEC = 4.0
END_TOLERANCE_SEC = 4.0

SAVE_MATCHED_ROW_PAIRS_CSV = "matched_vr_eeg_row_pairs.csv"
SAVE_UNMATCHED_VR_CSV = "unmatched_vr_rows.csv"
SAVE_UNMATCHED_EEG_CSV = "unmatched_eeg_rows.csv"
SAVE_SESSION_SUMMARY_CSV = "session_alignment_summary.csv"
SAVE_FINAL_TIMESTEP_TABLE_CSV = "merged_vr_eeg_timestep_table.csv"


# =========================================================
# Parsing helpers
# =========================================================
def parse_pg_array(value):
    """
    Convert a PostgreSQL array / Python list into a Python list of floats.
    """
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()

    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            if x is None:
                out.append(np.nan)
            else:
                out.append(float(x))
        return out

    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]

        if s == "":
            return []

        out = []
        for item in s.split(","):
            item = item.strip()
            if item in {"", "NULL", "None", "nan", "NaN"}:
                out.append(np.nan)
            else:
                out.append(float(item))
        return out

    return [float(value)]


def safe_float(value):
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# =========================================================
# Load data
# =========================================================
def load_vr_data_from_db(db_session: Session) -> pd.DataFrame:
    """
    Load VR rows and explicitly ignore eye_interactables.
    """
    vr_rows = db_session.query(VRDataModel).all()

    vr_data = []
    for row in vr_rows:
        vr_data.append({
            "vr_id": row.id,
            "session_id": row.session_id,
            "start_stamp": row.start_stamp,
            "end_stamp": row.end_stamp,
            "eye_id": row.eye_id,
            "score": row.score,  # kept only as reference
            "test_version": row.test_version,
            "end_timer": row.end_timer,
            "initial_timer": row.initial_timer,
            "rotation_speed": row.rotation_speed,
            "obj_rotation": row.obj_rotation,
            "description": row.description,
            "expected_rotation": row.expected_rotation,
            "obj_size": row.obj_size,
        })

    return pd.DataFrame(vr_data)


def load_eeg_data_from_db(db_session: Session) -> pd.DataFrame:
    """
    Load EEG rows and keep each EEG column as a parsed Python list.
    """
    eeg_rows = db_session.query(EpocXDataModel).all()

    eeg_data = []
    for row in eeg_rows:
        row_dict = {
            "eeg_id": row.id,
            "session_id": row.session_id,
            "start_stamp": row.start_stamp,
            "end_stamp": row.end_stamp,
        }

        for col in EEG_COLUMNS:
            row_dict[col] = parse_pg_array(getattr(row, col, None))

        eeg_data.append(row_dict)

    return pd.DataFrame(eeg_data)


# =========================================================
# Cleanup
# =========================================================
def standardize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_stamp"] = pd.to_datetime(df["start_stamp"], errors="coerce").dt.tz_localize(None)
    df["end_stamp"] = pd.to_datetime(df["end_stamp"], errors="coerce").dt.tz_localize(None)
    return df


def filter_basic_rows(vr_df: pd.DataFrame, eeg_df: pd.DataFrame):
    vr_df = vr_df.copy()
    eeg_df = eeg_df.copy()

    vr_df = vr_df[
        vr_df["session_id"].notna()
        & vr_df["start_stamp"].notna()
        & vr_df["end_stamp"].notna()
    ].copy()

    eeg_df = eeg_df[
        eeg_df["session_id"].notna()
        & eeg_df["start_stamp"].notna()
        & eeg_df["end_stamp"].notna()
    ].copy()

    return vr_df, eeg_df


def apply_vr_filters(vr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the same basic VR filtering idea from your older pipeline.
    """
    vr_df = vr_df.copy()
    vr_df = vr_df[vr_df["obj_size"].notna()].copy()
    vr_df = vr_df[vr_df["expected_rotation"].notna()].copy()
    vr_df = vr_df[vr_df["test_version"].isin([1, 2])].copy()
    return vr_df


# =========================================================
# Performance label
# =========================================================
def compute_elapsed_time_sec(vr_row: pd.Series) -> float:
    """
    Elapsed time Te for one trial.

    Preferred:
      Te = end_stamp - start_stamp

    Fallback:
      Te = initial_timer - end_timer
    """
    start_stamp = vr_row.get("start_stamp")
    end_stamp = vr_row.get("end_stamp")

    if pd.notna(start_stamp) and pd.notna(end_stamp):
        elapsed = (end_stamp - start_stamp).total_seconds()
        if np.isfinite(elapsed) and elapsed >= 0:
            return float(elapsed)

    initial_timer = safe_float(vr_row.get("initial_timer"))
    end_timer = safe_float(vr_row.get("end_timer"))

    if np.isfinite(initial_timer) and np.isfinite(end_timer):
        elapsed = initial_timer - end_timer
        if np.isfinite(elapsed) and elapsed >= 0:
            return float(elapsed)

    return np.nan


def add_performance_columns(vr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - elapsed_time_sec
      - expected_rotation_norm
      - obj_size_norm
      - difficulty_denominator
      - performance

    Formula implemented from the screenshot:

      perf(i) = Te / [ (ERot_i / ERot_max) + (NBlocks_i / NBlocks_max) ]
    """
    vr_df = vr_df.copy()

    vr_df["elapsed_time_sec"] = vr_df.apply(compute_elapsed_time_sec, axis=1)
    vr_df["expected_rotation"] = pd.to_numeric(vr_df["expected_rotation"], errors="coerce")
    vr_df["obj_size"] = pd.to_numeric(vr_df["obj_size"], errors="coerce")

    erot_max = vr_df["expected_rotation"].max(skipna=True)
    nblocks_max = vr_df["obj_size"].max(skipna=True)

    if pd.isna(erot_max) or erot_max <= 0:
        raise ValueError("Could not compute ERotmax from expected_rotation.")
    if pd.isna(nblocks_max) or nblocks_max <= 0:
        raise ValueError("Could not compute NBlocksmax from obj_size.")

    vr_df["expected_rotation_norm"] = vr_df["expected_rotation"] / float(erot_max)
    vr_df["obj_size_norm"] = vr_df["obj_size"] / float(nblocks_max)

    vr_df["difficulty_denominator"] = (
        vr_df["expected_rotation_norm"] + vr_df["obj_size_norm"]
    )

    vr_df["difficulty_denominator"] = vr_df["difficulty_denominator"].replace(0, np.nan)

    vr_df["performance"] = (
        vr_df["elapsed_time_sec"] / vr_df["difficulty_denominator"]
    )

    return vr_df


# =========================================================
# EEG validation
# =========================================================
def get_eeg_row_sample_length(eeg_row: pd.Series) -> int:
    """
    Return the common array length across EEG columns for one row.
    Raises if lengths are inconsistent.
    """
    lengths = []
    for col in EEG_COLUMNS:
        arr = parse_pg_array(eeg_row[col])
        if len(arr) > 0:
            lengths.append(len(arr))

    if not lengths:
        return 0

    unique_lengths = set(lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Inconsistent EEG array lengths for eeg_id={eeg_row['eeg_id']}: {sorted(unique_lengths)}"
        )

    return lengths[0]


def validate_eeg_rows(eeg_df: pd.DataFrame):
    """
    Make sure all EEG rows have consistent array lengths across columns.
    """
    bad_rows = []

    for _, row in eeg_df.iterrows():
        try:
            _ = get_eeg_row_sample_length(row)
        except ValueError as e:
            bad_rows.append(str(e))

    if bad_rows:
        print("\nBad EEG rows detected:")
        for msg in bad_rows[:20]:
            print(msg)
        raise ValueError(f"Found {len(bad_rows)} EEG rows with inconsistent array lengths.")


# =========================================================
# Alignment helpers
# =========================================================
def compute_time_gaps(vr_row: pd.Series, eeg_row: pd.Series):
    start_gap_signed = (vr_row["start_stamp"] - eeg_row["start_stamp"]).total_seconds()
    end_gap_signed = (vr_row["end_stamp"] - eeg_row["end_stamp"]).total_seconds()

    return (
        abs(start_gap_signed),
        abs(end_gap_signed),
        start_gap_signed,
        end_gap_signed,
    )


def align_session_rows_sequential(
    vr_session: pd.DataFrame,
    eeg_session: pd.DataFrame,
    start_tolerance_sec: float,
    end_tolerance_sec: float,
):
    """
    Sequential greedy alignment within one session.
    """
    vr_session = vr_session.sort_values("start_stamp").reset_index(drop=True)
    eeg_session = eeg_session.sort_values("start_stamp").reset_index(drop=True)

    matches = []
    unmatched_vr = []
    unmatched_eeg = []

    i = 0
    j = 0

    while i < len(vr_session) and j < len(eeg_session):
        vr_row = vr_session.iloc[i]
        eeg_row = eeg_session.iloc[j]

        start_gap_abs, end_gap_abs, start_gap_signed, end_gap_signed = compute_time_gaps(vr_row, eeg_row)

        if start_gap_abs <= start_tolerance_sec and end_gap_abs <= end_tolerance_sec:
            assert vr_row["session_id"] == eeg_row["session_id"]

            merged = {
                "session_id": vr_row["session_id"],

                # VR identifiers and labels
                "vr_id": vr_row["vr_id"],
                "vr_start_stamp": vr_row["start_stamp"],
                "vr_end_stamp": vr_row["end_stamp"],
                "eye_id": vr_row["eye_id"],

                # old reference label
                "score": vr_row["score"],

                # new label + components
                "elapsed_time_sec": vr_row["elapsed_time_sec"],
                "expected_rotation_norm": vr_row["expected_rotation_norm"],
                "obj_size_norm": vr_row["obj_size_norm"],
                "difficulty_denominator": vr_row["difficulty_denominator"],
                "performance": vr_row["performance"],

                "test_version": vr_row["test_version"],
                "end_timer": vr_row["end_timer"],
                "initial_timer": vr_row["initial_timer"],
                "rotation_speed": vr_row["rotation_speed"],
                "obj_rotation": vr_row["obj_rotation"],
                "description": vr_row["description"],
                "expected_rotation": vr_row["expected_rotation"],
                "obj_size": vr_row["obj_size"],

                # EEG identifiers and row timing
                "eeg_id": eeg_row["eeg_id"],
                "eeg_start_stamp": eeg_row["start_stamp"],
                "eeg_end_stamp": eeg_row["end_stamp"],

                # diagnostics
                "match_start_gap_sec": start_gap_abs,
                "match_end_gap_sec": end_gap_abs,
                "match_start_gap_signed_sec": start_gap_signed,
                "match_end_gap_signed_sec": end_gap_signed,
            }

            for col in EEG_COLUMNS:
                merged[col] = eeg_row[col]

            matches.append(merged)
            i += 1
            j += 1
            continue

        if vr_row["start_stamp"] < eeg_row["start_stamp"]:
            unmatched_vr.append({
                "session_id": vr_row["session_id"],
                "vr_id": vr_row["vr_id"],
                "vr_start_stamp": vr_row["start_stamp"],
                "vr_end_stamp": vr_row["end_stamp"],
                "candidate_eeg_id": eeg_row["eeg_id"],
                "candidate_eeg_start_stamp": eeg_row["start_stamp"],
                "candidate_eeg_end_stamp": eeg_row["end_stamp"],
                "start_gap_abs_sec": start_gap_abs,
                "end_gap_abs_sec": end_gap_abs,
                "start_gap_signed_sec": start_gap_signed,
                "end_gap_signed_sec": end_gap_signed,
                "reason": "VR row earlier than current EEG row; no close sequential match",
            })
            i += 1
        else:
            unmatched_eeg.append({
                "session_id": eeg_row["session_id"],
                "eeg_id": eeg_row["eeg_id"],
                "eeg_start_stamp": eeg_row["start_stamp"],
                "eeg_end_stamp": eeg_row["end_stamp"],
                "candidate_vr_id": vr_row["vr_id"],
                "candidate_vr_start_stamp": vr_row["start_stamp"],
                "candidate_vr_end_stamp": vr_row["end_stamp"],
                "start_gap_abs_sec": start_gap_abs,
                "end_gap_abs_sec": end_gap_abs,
                "start_gap_signed_sec": start_gap_signed,
                "end_gap_signed_sec": end_gap_signed,
                "reason": "EEG row earlier than current VR row; no close sequential match",
            })
            j += 1

    while i < len(vr_session):
        vr_row = vr_session.iloc[i]
        unmatched_vr.append({
            "session_id": vr_row["session_id"],
            "vr_id": vr_row["vr_id"],
            "vr_start_stamp": vr_row["start_stamp"],
            "vr_end_stamp": vr_row["end_stamp"],
            "reason": "Leftover VR row at end of session",
        })
        i += 1

    while j < len(eeg_session):
        eeg_row = eeg_session.iloc[j]
        unmatched_eeg.append({
            "session_id": eeg_row["session_id"],
            "eeg_id": eeg_row["eeg_id"],
            "eeg_start_stamp": eeg_row["start_stamp"],
            "eeg_end_stamp": eeg_row["end_stamp"],
            "reason": "Leftover EEG row at end of session",
        })
        j += 1

    matches_df = pd.DataFrame(matches)
    unmatched_vr_df = pd.DataFrame(unmatched_vr)
    unmatched_eeg_df = pd.DataFrame(unmatched_eeg)

    return matches_df, unmatched_vr_df, unmatched_eeg_df


def align_all_sessions(
    vr_df: pd.DataFrame,
    eeg_df: pd.DataFrame,
    start_tolerance_sec: float,
    end_tolerance_sec: float,
):
    vr_sessions = set(vr_df["session_id"])
    eeg_sessions = set(eeg_df["session_id"])

    common_sessions = sorted(vr_sessions.intersection(eeg_sessions))
    vr_only_sessions = sorted(vr_sessions - eeg_sessions)
    eeg_only_sessions = sorted(eeg_sessions - vr_sessions)

    all_matches = []
    all_unmatched_vr = []
    all_unmatched_eeg = []
    session_summary = []

    for session_id in common_sessions:
        vr_s = vr_df[vr_df["session_id"] == session_id].copy()
        eeg_s = eeg_df[eeg_df["session_id"] == session_id].copy()

        matches_df, unmatched_vr_df, unmatched_eeg_df = align_session_rows_sequential(
            vr_s,
            eeg_s,
            start_tolerance_sec=start_tolerance_sec,
            end_tolerance_sec=end_tolerance_sec,
        )

        if not matches_df.empty:
            all_matches.append(matches_df)
        if not unmatched_vr_df.empty:
            all_unmatched_vr.append(unmatched_vr_df)
        if not unmatched_eeg_df.empty:
            all_unmatched_eeg.append(unmatched_eeg_df)

        session_summary.append({
            "session_id": session_id,
            "vr_rows": len(vr_s),
            "eeg_rows": len(eeg_s),
            "matched_rows": len(matches_df),
            "unmatched_vr_rows": len(unmatched_vr_df),
            "unmatched_eeg_rows": len(unmatched_eeg_df),
        })

    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    unmatched_vr_df = pd.concat(all_unmatched_vr, ignore_index=True) if all_unmatched_vr else pd.DataFrame()
    unmatched_eeg_df = pd.concat(all_unmatched_eeg, ignore_index=True) if all_unmatched_eeg else pd.DataFrame()
    session_summary_df = pd.DataFrame(session_summary)

    return (
        matches_df,
        unmatched_vr_df,
        unmatched_eeg_df,
        session_summary_df,
        vr_only_sessions,
        eeg_only_sessions,
    )


# =========================================================
# Reporting
# =========================================================
def print_alignment_report(
    session_summary_df: pd.DataFrame,
    unmatched_vr_df: pd.DataFrame,
    unmatched_eeg_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    vr_only_sessions,
    eeg_only_sessions,
):
    print("\n================ ALIGNMENT REPORT ================\n")

    if vr_only_sessions:
        print("Sessions only in VR:")
        for s in vr_only_sessions:
            print(f"  {s}")
        print()

    if eeg_only_sessions:
        print("Sessions only in EEG:")
        for s in eeg_only_sessions:
            print(f"  {s}")
        print()

    if session_summary_df.empty:
        print("No overlapping sessions found between VR and EEG.")
        print("\n==================================================\n")
        return

    print("Per-session summary:")
    print(session_summary_df.to_string(index=False))

    total_matched = len(matches_df)
    total_unmatched_vr = len(unmatched_vr_df)
    total_unmatched_eeg = len(unmatched_eeg_df)
    total_mismatches = total_unmatched_vr + total_unmatched_eeg

    print(f"\nTotal matched row-pairs: {total_matched}")
    print(f"Total unmatched VR rows: {total_unmatched_vr}")
    print(f"Total unmatched EEG rows: {total_unmatched_eeg}")
    print(f"Total mismatches: {total_mismatches}")

    if not matches_df.empty:
        print("\nMatched row timing gaps (first 20):")
        print(
            matches_df[
                [
                    "session_id",
                    "vr_id",
                    "eeg_id",
                    "match_start_gap_sec",
                    "match_end_gap_sec",
                    "match_start_gap_signed_sec",
                    "match_end_gap_signed_sec",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        print("\nMatched row timing gap summary:")
        print(
            matches_df[
                [
                    "match_start_gap_sec",
                    "match_end_gap_sec",
                    "match_start_gap_signed_sec",
                    "match_end_gap_signed_sec",
                ]
            ]
            .describe()
            .to_string()
        )

        print("\nPerformance summary on matched trials:")
        print(
            matches_df[
                [
                    "elapsed_time_sec",
                    "expected_rotation_norm",
                    "obj_size_norm",
                    "difficulty_denominator",
                    "performance",
                ]
            ]
            .describe()
            .to_string()
        )

    if not unmatched_vr_df.empty:
        print("\nUnmatched VR rows (first 20):")
        print(unmatched_vr_df.head(20).to_string(index=False))

    if not unmatched_eeg_df.empty:
        print("\nUnmatched EEG rows (first 20):")
        print(unmatched_eeg_df.head(20).to_string(index=False))

    print("\n==================================================\n")


# =========================================================
# Expansion to EEG timesteps
# =========================================================
def expand_matched_rows_to_eeg_timesteps(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each matched row-pair into one row per EEG timestep and replicate
    the VR labels across every EEG timestep.
    """
    expanded_rows = []

    for _, row in matches_df.iterrows():
        sample_len = get_eeg_row_sample_length(row)
        if sample_len == 0:
            continue

        eeg_start = pd.to_datetime(row["eeg_start_stamp"])
        eeg_end = pd.to_datetime(row["eeg_end_stamp"])
        duration_sec = (eeg_end - eeg_start).total_seconds()

        for sample_idx in range(sample_len):
            if sample_len == 1:
                rel_sec = 0.0
            else:
                rel_sec = sample_idx * duration_sec / (sample_len - 1)

            out = {
                "session_id": row["session_id"],
                "vr_id": row["vr_id"],
                "eeg_id": row["eeg_id"],

                "vr_start_stamp": row["vr_start_stamp"],
                "vr_end_stamp": row["vr_end_stamp"],
                "eeg_start_stamp": row["eeg_start_stamp"],
                "eeg_end_stamp": row["eeg_end_stamp"],

                "sample_idx": sample_idx,
                "sample_time_sec": rel_sec,
                "sample_stamp": eeg_start + pd.to_timedelta(rel_sec, unit="s"),

                "eye_id": row["eye_id"],

                "score": row["score"],

                "elapsed_time_sec": row["elapsed_time_sec"],
                "expected_rotation_norm": row["expected_rotation_norm"],
                "obj_size_norm": row["obj_size_norm"],
                "difficulty_denominator": row["difficulty_denominator"],
                "performance": row["performance"],

                "test_version": row["test_version"],
                "end_timer": row["end_timer"],
                "initial_timer": row["initial_timer"],
                "rotation_speed": row["rotation_speed"],
                "obj_rotation": row["obj_rotation"],
                "description": row["description"],
                "expected_rotation": row["expected_rotation"],
                "obj_size": row["obj_size"],

                "match_start_gap_sec": row["match_start_gap_sec"],
                "match_end_gap_sec": row["match_end_gap_sec"],
                "match_start_gap_signed_sec": row["match_start_gap_signed_sec"],
                "match_end_gap_signed_sec": row["match_end_gap_signed_sec"],
            }

            for col in EEG_COLUMNS:
                arr = parse_pg_array(row[col])
                out[col] = arr[sample_idx] if sample_idx < len(arr) else np.nan

            expanded_rows.append(out)

    return pd.DataFrame(expanded_rows)


# =========================================================
# Main pipeline
# =========================================================
def build_merged_eeg_vr_timestep_table(
    db_session: Session,
    start_tolerance_sec: float = START_TOLERANCE_SEC,
    end_tolerance_sec: float = END_TOLERANCE_SEC,
):
    vr_df = load_vr_data_from_db(db_session)
    eeg_df = load_eeg_data_from_db(db_session)

    vr_df = standardize_datetimes(vr_df)
    eeg_df = standardize_datetimes(eeg_df)

    vr_df, eeg_df = filter_basic_rows(vr_df, eeg_df)
    vr_df = apply_vr_filters(vr_df)
    vr_df = add_performance_columns(vr_df)

    validate_eeg_rows(eeg_df)

    (
        matches_df,
        unmatched_vr_df,
        unmatched_eeg_df,
        session_summary_df,
        vr_only_sessions,
        eeg_only_sessions,
    ) = align_all_sessions(
        vr_df=vr_df,
        eeg_df=eeg_df,
        start_tolerance_sec=start_tolerance_sec,
        end_tolerance_sec=end_tolerance_sec,
    )

    print_alignment_report(
        session_summary_df=session_summary_df,
        unmatched_vr_df=unmatched_vr_df,
        unmatched_eeg_df=unmatched_eeg_df,
        matches_df=matches_df,
        vr_only_sessions=vr_only_sessions,
        eeg_only_sessions=eeg_only_sessions,
    )

    final_timestep_table = expand_matched_rows_to_eeg_timesteps(matches_df)

    return {
        "matched_row_pairs": matches_df,
        "unmatched_vr_rows": unmatched_vr_df,
        "unmatched_eeg_rows": unmatched_eeg_df,
        "session_summary": session_summary_df,
        "final_timestep_table": final_timestep_table,
    }


def main():
    db = next(get_db())
    try:
        result = build_merged_eeg_vr_timestep_table(
            db_session=db,
            start_tolerance_sec=START_TOLERANCE_SEC,
            end_tolerance_sec=END_TOLERANCE_SEC,
        )

        matches_df = result["matched_row_pairs"]
        unmatched_vr_df = result["unmatched_vr_rows"]
        unmatched_eeg_df = result["unmatched_eeg_rows"]
        session_summary_df = result["session_summary"]
        final_timestep_table = result["final_timestep_table"]

        print(f"Matched row-pairs kept: {len(matches_df)}")
        print(f"Removed unmatched VR rows: {len(unmatched_vr_df)}")
        print(f"Removed unmatched EEG rows: {len(unmatched_eeg_df)}")
        print(f"Final EEG timestep rows: {len(final_timestep_table)}")

        if not final_timestep_table.empty:
            print("\nFinal merged timestep table preview:")
            print(final_timestep_table.head().to_string(index=False))

        matches_df.to_csv(SAVE_MATCHED_ROW_PAIRS_CSV, index=False)
        unmatched_vr_df.to_csv(SAVE_UNMATCHED_VR_CSV, index=False)
        unmatched_eeg_df.to_csv(SAVE_UNMATCHED_EEG_CSV, index=False)
        session_summary_df.to_csv(SAVE_SESSION_SUMMARY_CSV, index=False)
        final_timestep_table.to_csv(SAVE_FINAL_TIMESTEP_TABLE_CSV, index=False)

        print("\nSaved files:")
        print(f"  {SAVE_MATCHED_ROW_PAIRS_CSV}")
        print(f"  {SAVE_UNMATCHED_VR_CSV}")
        print(f"  {SAVE_UNMATCHED_EEG_CSV}")
        print(f"  {SAVE_SESSION_SUMMARY_CSV}")
        print(f"  {SAVE_FINAL_TIMESTEP_TABLE_CSV}")

    finally:
        db.close()


if __name__ == "__main__":
    main()