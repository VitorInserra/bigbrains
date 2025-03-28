import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models.VRDataModel import VRDataModel
from models.EpocXDataModel import EpocXDataModel
from db import get_db
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.stats import pearsonr
import itertools
from MLPipe import load_data_from_db


def compute_eeg_averages_by_row(
    eeg_df, sensor_keys, impossible_threshold=150.0, z_thresh=1.0, max_iters=5
):
    """
    For each row in eeg_df, take the 1D array in each sensor_key column,
    clean it iteratively, then compute a single scalar average.
    The result is placed in a new column: sensor_key + "_avg".
    """
    # Make sure we don't modify original data
    eeg_df = eeg_df.copy()

    for sensor_key in sensor_keys:
        avg_col = sensor_key + "_avg"
        avg_values = []

        for _, row in eeg_df.iterrows():
            # row[sensor_key] should be an array-like of raw values
            sensor_data = np.array(row[sensor_key], dtype=float)

            # 1) Basic outlier-cleaning (iterative z-score)
            s = pd.Series(sensor_data).interpolate(method="linear")
            for _ in range(max_iters):
                mean_val = s.mean()
                std_val = s.std()
                if std_val == 0 or s.empty:
                    break
                z_scores = (s - mean_val).abs() / std_val
                outliers = z_scores > z_thresh
                if not outliers.any():
                    break
                s[outliers] = np.nan
                s = s.interpolate(method="linear")

            # 2) Discard impossible values
            s = s[s < impossible_threshold]

            if len(s) == 0:
                avg_values.append(np.nan)
            else:
                avg_values.append(s.mean())

        # Attach the new average column
        eeg_df[avg_col] = avg_values

    return eeg_df




def find_and_plot_correlation(
    df, sensor_keys_avg, output_path="stats_imgs/correlation_matrix.png"
):
    """
    Given df with columns like ["af3_alpha_avg", "timer_diff", "rot_diff", ...],
    compute pairwise correlations and their p-values, then plot a heatmap.
    """
    # subset of columns we want to correlate
    cols_of_interest = sensor_keys_avg + ["timer_diff", "rot_diff", "rand_obj_gaze"]

    # 1) Print correlation + p-values for each pair
    for col1, col2 in itertools.combinations(cols_of_interest, 2):
        valid_data = df[[col1, col2]].dropna()
        if len(valid_data) < 2:
            print(f"Not enough data to correlate {col1} and {col2}.")
            continue

        r_value, p_value = pearsonr(valid_data[col1], valid_data[col2])
        print(
            f"Correlation between {col1} and {col2}: r = {r_value:.4f}, p = {p_value:.4g}"
        )

    # 2) Compute correlation matrix
    corr_matrix = df[cols_of_interest].corr()

    # 3) Plot correlation matrix as heatmap
    plt.figure()
    plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest")
    plt.title("Correlation Matrix")
    plt.xticks(range(len(cols_of_interest)), cols_of_interest, rotation=45)
    plt.yticks(range(len(cols_of_interest)), cols_of_interest)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def extract_obj_gaze_time(eye_array):
    """
    eye_array: e.g. [[0,1,0,0], [0,1,0,0], [0,0,1,0], ...]
    Each row is a 'sample' in your VR logging.

    We'll compute fraction of samples that are 1 for each column.
    """
    if not eye_array:
        return {"frac_timer": 0, "frac_gameobj": 0, "frac_outline": 0, "frac_score": 0}

    df = pd.DataFrame(eye_array, columns=["timer", "gameobj", "outline", "score"])
    frac_gameobj = df["gameobj"].mean()

    return frac_gameobj


def main_feature_extraction():
    db = next(get_db())
    try:
        vr_df, eeg_df = load_data_from_db(db)
        eeg_df["start_stamp"] = pd.to_datetime(eeg_df["start_stamp"]).dt.tz_localize(
            None
        )
        eeg_df["end_stamp"] = pd.to_datetime(eeg_df["end_stamp"]).dt.tz_localize(None)

        first_eeg_row = eeg_df.iloc[350].to_dict()
        filtered_eeg = eeg_df[eeg_df["session_id"] == first_eeg_row["session_id"]]
        filtered_vr = vr_df[vr_df["session_id"] == first_eeg_row["session_id"]]

        # Sort VR by start_stamp
        filtered_vr.sort_values(by="start_stamp", inplace=True)
        filtered_vr = filtered_vr.iloc[1:].reset_index(drop=True)
        if len(filtered_vr) >= 2:
            last_two = filtered_vr.iloc[-2:]
            if (last_two["end_timer"] == 0).all():
                filtered_vr = filtered_vr.iloc[:-1].reset_index(drop=True)

        # Compute VR columns
        filtered_vr["timer_diff"] = (
            filtered_vr["initial_timer"] - filtered_vr["end_timer"]
        )
        filtered_vr["rot_diff"] = (
            filtered_vr["obj_rotation"] - filtered_vr["expected_rotation"]
        )

        rand_obj_gaze = []
        for _, vr_row in filtered_vr.iterrows():
            rand_obj_gaze.append(extract_obj_gaze_time(vr_row["eye_interactables"]))

        filtered_vr["rand_obj_gaze"] = rand_obj_gaze

        filtered_vr = filtered_vr.sort_values("start_stamp")

        plt.figure(figsize=(12, 6))
        plt.plot(filtered_vr["start_stamp"], filtered_vr["rand_obj_gaze"], marker='o')
        plt.title("Random Object Gaze Time Over Time")
        plt.xlabel("Start Time")
        plt.ylabel("Rand Obj Gaze (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("stats_imgs/rand_obj_gaze")
        plt.show()


        filtered_eeg.sort_values(by="start_stamp", inplace=True)
        filtered_eeg.reset_index(drop=True, inplace=True)

        # EXAMPLE: Let's average a handful of sensors, row by row
        sensors_to_avg = []
        channels = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
        ]
        for channel in channels:
            for band in ["alpha"]:
                sensors_to_avg.append(channel.lower() + "_" + band)
        filtered_eeg = compute_eeg_averages_by_row(
            filtered_eeg,
            sensors_to_avg,
            impossible_threshold=20.0,
            z_thresh=1.0,
            max_iters=5,
        )

        combined_df = filtered_eeg.join(filtered_vr[["timer_diff", "rot_diff", "rand_obj_gaze"]])
        # sensor_avg_cols = []
        # for channel in channels:
        #     sensor_avg_cols.append(channel.lower() + "_alpha")
        #     combined_df[channel.lower() + "_theta_beta_ratio"] = combined_df[channel.lower() + "_theta_avg"] / (
        #         combined_df[channel.lower() + "_beta_l_avg"] + combined_df[channel.lower() + "_beta_h_avg"]
        #     )


        # Build list of new sensor avg columns (e.g. "af3_alpha_avg")
        sensor_avg_cols = [f"{s}_avg" for s in sensors_to_avg]

        # Finally, compute correlation & plot a correlation matrix
        find_and_plot_correlation(
            combined_df,
            sensor_avg_cols,
            output_path="stats_imgs/correlation_matrix.png",
        )

        # Optionally, also plot your VR performance or sensor traces individually...
        # plot_performance(filtered_vr)  # etc.

    finally:
        db.close()


if __name__ == "__main__":
    main_feature_extraction()
