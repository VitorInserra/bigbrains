# ============================================================
# build_single_feature_table.py
#
# Purpose:
#   1) Read the matched EEG/VR pairs CSV
#   2) Build ONE single long feature table:
#        - one row = one timestep/window inside one trial
#        - includes PSD band features, Shannon entropy features,
#          and asymmetry features
#   3) Provide an in-memory helper to turn that table into
#      variable-length trial sequences for a future BiLSTM
#
# Output:
#   - ONE CSV only: single_feature_table.csv
# ============================================================

import ast
import csv
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
INPUT_CSV = "matched_vr_eeg_row_pairs.csv"
OUTPUT_CSV = "single_feature_table.csv"

WINDOW_SEC = 1.0
OVERLAP = 0.5
DEFAULT_FS = 50.0

ENTROPY_BINS = 10
EPS = 1e-8

CHANNELS = [
    "af3", "f7", "f3", "fc5", "t7", "p7", "o1",
    "o2", "p8", "t8", "fc6", "f4", "f8", "af4"
]
BANDS = ["theta", "alpha", "beta_l", "beta_h", "gamma"]

ASYM_PAIRS = [
    ("af3", "af4"),
    ("f7", "f8"),
    ("f3", "f4"),
    ("fc5", "fc6"),
    ("t7", "t8"),
    ("p7", "p8"),
    ("o1", "o2"),
]


# =========================
# HELPERS
# =========================
def safe_field_size_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def parse_timestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_int(value, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_array_cell(cell: str) -> Optional[np.ndarray]:
    if cell is None:
        return None

    text = str(cell).strip()
    if text == "" or text.lower() == "nan":
        return None

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, (list, tuple, np.ndarray)):
        return None

    try:
        arr = np.asarray(parsed, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if arr.ndim != 1 or arr.size == 0:
        return None

    return arr


def infer_fs(row: Dict[str, str], n_samples: int) -> float:
    eeg_start = parse_timestamp(row.get("eeg_start_stamp"))
    eeg_end = parse_timestamp(row.get("eeg_end_stamp"))

    if pd.notna(eeg_start) and pd.notna(eeg_end):
        duration = (eeg_end - eeg_start).total_seconds()
        if duration and duration > 0:
            return float(n_samples / duration)

    vr_start = parse_timestamp(row.get("vr_start_stamp"))
    vr_end = parse_timestamp(row.get("vr_end_stamp"))

    if pd.notna(vr_start) and pd.notna(vr_end):
        duration = (vr_end - vr_start).total_seconds()
        if duration and duration > 0:
            return float(n_samples / duration)

    return float(DEFAULT_FS)


def shannon_entropy_1d(x: np.ndarray, bins: int = ENTROPY_BINS) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size <= 1:
        return 0.0

    if np.allclose(x, x[0]):
        return 0.0

    n_bins = min(bins, max(2, int(np.sqrt(x.size))))
    hist, _ = np.histogram(x, bins=n_bins, density=False)

    hist = hist.astype(np.float64)
    hist = hist[hist > 0]

    if hist.size == 0:
        return 0.0

    p = hist / hist.sum()
    return float(-(p * np.log2(p)).sum())


def build_trial_band_df(row: Dict[str, str]) -> pd.DataFrame:
    lower_key_lookup = {k.lower(): k for k in row.keys()}

    data = {}
    lengths = []

    for ch in CHANNELS:
        for band in BANDS:
            col_lower = f"{ch}_{band}"
            if col_lower not in lower_key_lookup:
                continue

            actual_col = lower_key_lookup[col_lower]
            arr = parse_array_cell(row.get(actual_col))

            if arr is None:
                continue

            data[col_lower] = arr
            lengths.append(len(arr))

    if not data:
        return pd.DataFrame()

    min_len = min(lengths)
    for col in data:
        data[col] = data[col][:min_len]

    return pd.DataFrame(data)


def make_trial_id(row: Dict[str, str], row_index: int) -> str:
    session_id = str(row.get("session_id", ""))
    vr_id = safe_int(row.get("vr_id"))
    eeg_id = safe_int(row.get("eeg_id"))
    return f"{session_id}__vr{vr_id}__eeg{eeg_id}__row{row_index}"


def get_window_bounds(n_samples: int, fs: float, window_sec: float, overlap: float) -> List[Tuple[int, int]]:
    if n_samples <= 0:
        return []

    window_samples = max(2, int(round(window_sec * fs)))
    if window_samples >= n_samples:
        return [(0, n_samples)]

    step = max(1, int(round(window_samples * (1.0 - overlap))))
    bounds = []

    start = 0
    while start + window_samples <= n_samples:
        bounds.append((start, start + window_samples))
        start += step

    if bounds and bounds[-1][1] < n_samples:
        bounds.append((n_samples - window_samples, n_samples))

    bounds = list(dict.fromkeys(bounds))
    return bounds


def extract_trial_timestep_rows(row: Dict[str, str], row_index: int) -> List[Dict]:
    band_df = build_trial_band_df(row)
    if band_df.empty:
        return []

    n_samples = len(band_df)
    fs = infer_fs(row, n_samples)
    windows = get_window_bounds(n_samples, fs, WINDOW_SEC, OVERLAP)
    if not windows:
        return []

    trial_id = make_trial_id(row, row_index)

    eeg_start = parse_timestamp(row.get("eeg_start_stamp"))
    eeg_end = parse_timestamp(row.get("eeg_end_stamp"))
    vr_start = parse_timestamp(row.get("vr_start_stamp"))
    vr_end = parse_timestamp(row.get("vr_end_stamp"))

    out_rows = []

    for timestep_idx, (start, end) in enumerate(windows):
        feat = {
            "trial_id": trial_id,
            "session_id": row.get("session_id", ""),
            "row_index": row_index,
            "timestep_idx": timestep_idx,

            # label
            "performance": safe_float(row.get("performance")),

            # trial metadata
            "test_version": safe_int(row.get("test_version")),
            "vr_id": safe_int(row.get("vr_id")),
            "eeg_id": safe_int(row.get("eeg_id")),
            "eye_id": row.get("eye_id", ""),

            "elapsed_time_sec": safe_float(row.get("elapsed_time_sec")),
            "expected_rotation": safe_float(row.get("expected_rotation")),
            "expected_rotation_norm": safe_float(row.get("expected_rotation_norm")),
            "obj_size": safe_float(row.get("obj_size")),
            "obj_size_norm": safe_float(row.get("obj_size_norm")),
            "difficulty_denominator": safe_float(row.get("difficulty_denominator")),

            "vr_start_stamp": vr_start,
            "vr_end_stamp": vr_end,
            "eeg_start_stamp": eeg_start,
            "eeg_end_stamp": eeg_end,

            "window_start_sample": start,
            "window_end_sample": end,
            "window_num_samples": end - start,
            "fs_inferred_hz": fs,
            "window_start_sec_from_trial": start / fs,
            "window_end_sec_from_trial": end / fs,
        }

        psd_means = {}

        for col in band_df.columns:
            segment = band_df[col].iloc[start:end].to_numpy(dtype=np.float64)
            psd_mean = float(np.nanmean(segment))
            psd_means[col] = psd_mean
            feat[f"{col}_psd"] = psd_mean

        for col in band_df.columns:
            segment = band_df[col].iloc[start:end].to_numpy(dtype=np.float64)
            feat[f"{col}_shannon"] = shannon_entropy_1d(segment, bins=ENTROPY_BINS)

        for left, right in ASYM_PAIRS:
            for band in BANDS:
                left_col = f"{left}_{band}"
                right_col = f"{right}_{band}"

                if left_col not in psd_means or right_col not in psd_means:
                    continue

                left_val = max(psd_means[left_col], 0.0)
                right_val = max(psd_means[right_col], 0.0)

                feat[f"{left}_{right}_{band}_asym_log"] = (
                    np.log(left_val + EPS) - np.log(right_val + EPS)
                )
                feat[f"{left}_{right}_{band}_asym_diff"] = left_val - right_val

        out_rows.append(feat)

    return out_rows


def build_single_feature_table(input_csv: str) -> pd.DataFrame:
    safe_field_size_limit()

    rows_out = []

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):
            timestep_rows = extract_trial_timestep_rows(row, row_index)
            rows_out.extend(timestep_rows)

    if not rows_out:
        raise RuntimeError("No rows were produced. Check your input CSV path and column names.")

    df = pd.DataFrame(rows_out)

    df = df.sort_values(
        by=["session_id", "trial_id", "timestep_idx"],
        kind="stable"
    ).reset_index(drop=True)

    return df


def get_bilstm_inputs_from_feature_table(
    feature_table: pd.DataFrame,
    label_col: str = "performance",
) -> Tuple[List[np.ndarray], np.ndarray, List[str], np.ndarray]:
    meta_cols = {
        "trial_id", "session_id", "row_index", "timestep_idx",
        "performance",
        "test_version", "vr_id", "eeg_id", "eye_id",
        "elapsed_time_sec", "expected_rotation", "expected_rotation_norm",
        "obj_size", "obj_size_norm", "difficulty_denominator",
        "vr_start_stamp", "vr_end_stamp", "eeg_start_stamp", "eeg_end_stamp",
        "window_start_sample", "window_end_sample", "window_num_samples",
        "window_start_sec_from_trial", "window_end_sec_from_trial",
        "fs_inferred_hz"
    }

    feature_cols = [c for c in feature_table.columns if c not in meta_cols]

    X_list = []
    y_list = []
    groups = []

    grouped = feature_table.sort_values(
        ["trial_id", "timestep_idx"], kind="stable"
    ).groupby("trial_id", sort=False)

    for _, g in grouped:
        x = (
            g[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        y = g[label_col].iloc[0]
        group = g["session_id"].iloc[0]

        X_list.append(x)
        y_list.append(y)
        groups.append(group)

    y_array = np.asarray(y_list, dtype=np.float32)
    groups_array = np.asarray(groups)

    return X_list, y_array, feature_cols, groups_array


if __name__ == "__main__":
    feature_table = build_single_feature_table(INPUT_CSV)
    feature_table.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved single feature table: {OUTPUT_CSV}")
    print(f"Rows: {len(feature_table)}")
    print(f"Trials: {feature_table['trial_id'].nunique()}")
    print(f"Sessions: {feature_table['session_id'].nunique()}")

    X_list, y, feature_cols, groups = get_bilstm_inputs_from_feature_table(feature_table)

    lengths = [x.shape[0] for x in X_list]
    print(f"Number of trial sequences: {len(X_list)}")
    print(f"Feature dimension: {len(feature_cols)}")
    print(f"Min timesteps per trial: {min(lengths)}")
    print(f"Max timesteps per trial: {max(lengths)}")
    print(f"Mean timesteps per trial: {np.mean(lengths):.2f}")