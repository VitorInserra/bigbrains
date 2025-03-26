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


def load_data_from_db(db_session: Session):
    """
    Loads VR data and EEG data from the database.

    Returns:
      - vr_df: A DataFrame with VR session details (one row per VR interaction).
      - eeg_df: A DataFrame with EEG power band values, session IDs, and timestamps.
    """

    vr_rows = db_session.query(VRDataModel).all()
    vr_data = [
        {
            "id": row.id,
            "session_id": row.session_id,
            "start_stamp": row.start_stamp,
            "end_stamp": row.end_stamp,
            "score": row.score,
            "test_version": row.test_version,
            "end_timer": row.end_timer,
            "initial_timer": row.initial_timer,
            "rotation_speed": row.rotation_speed,
            "obj_rotation": row.obj_rotation,
            "expected_rotation": row.expected_rotation,
            "obj_size": row.obj_size,
            "description": row.description,
            "eye_interactables": row.eye_interactables,
        }
        for row in vr_rows
    ]
    vr_df = pd.DataFrame(vr_data)

    eeg_rows = db_session.query(EpocXDataModel).all()

    eeg_data = []
    for row in eeg_rows:
        row_dict = {
            "session_id": row.session_id,
            "start_stamp": row.start_stamp,
            "end_stamp": row.end_stamp,
        }

        for channel in [
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
        ]:
            for band in ["theta", "alpha", "beta_l", "beta_h", "gamma"]:
                column_name = f"{channel.lower()}_{band.lower()}"
                row_dict[column_name] = getattr(row, column_name, None)

        eeg_data.append(row_dict)

    eeg_df = pd.DataFrame(eeg_data)

    return vr_df, eeg_df


def extract_eeg_features_for_round(eeg_row, round_start, round_end):
    """
    Example: If you saved 'af3' as an array of band-power samples that exactly
    matches round_start→round_end, you can compute summary stats here.
    """
    features = {}

    for channel_name in [
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
    ]:
        for band in ["theta", "alpha", "beta_l", "beta_h", "gamma"]:
            column_name = f"{channel_name.lower()}_{band.lower()}"
            channel_data = eeg_row[column_name]
            if not channel_data:
                features[f"{column_name}_mean"] = 0
                features[f"{column_name}_std"] = 0
            else:
                series = pd.Series(channel_data)
                features[f"{column_name}_mean"] = series.mean()
                features[f"{column_name}_std"] = series.std()

    return features


def summarize_eye_gaze(eye_array):
    """
    eye_array: e.g. [[0,1,0,0], [0,1,0,0], [0,0,1,0], ...]
    Each row is a 'sample' in your VR logging.

    We'll compute fraction of samples that are 1 for each column.
    """
    if not eye_array:
        return {"frac_timer": 0, "frac_gameobj": 0, "frac_outline": 0, "frac_score": 0}

    df = pd.DataFrame(eye_array, columns=["timer", "gameobj", "outline", "score"])
    frac_timer = df["timer"].mean()
    frac_gameobj = df["gameobj"].mean()
    frac_outline = df["outline"].mean()
    frac_score = df["score"].mean()

    return {
        "frac_timer": frac_timer,
        "frac_gameobj": frac_gameobj,
        "frac_outline": frac_outline,
        "frac_score": frac_score,
    }


def build_feature_table(vr_df, eeg_df):
    """
    Creates a final DataFrame where each row = 1 VR round + summarized EEG features.
    """
    feature_rows = []

    for _, vr_row in vr_df.iterrows():
        round_session_id = vr_row["session_id"]
        round_start = vr_row["start_stamp"]
        round_end = vr_row["end_stamp"]

        matching_eeg = eeg_df[
            (eeg_df["session_id"] == round_session_id)
            & (eeg_df["start_stamp"] >= round_start)
            & (eeg_df["end_stamp"] <= round_end)
        ]

        if matching_eeg.empty:

            continue

        eeg_row = matching_eeg.iloc[0].to_dict()

        eeg_features = extract_eeg_features_for_round(eeg_row, round_start, round_end)

        eye_gaze_features = summarize_eye_gaze(vr_row["eye_interactables"])

        leftover_time = vr_row["end_timer"]

        performance_metric = compute_performance_metric(
            score=vr_row["score"],
            rotation_speed=vr_row["rotation_speed"],
            obj_rotation=vr_row["obj_rotation"],
            leftover_time=leftover_time,
        )

        row_features = {
            "session_id": round_session_id,
            "round_id": vr_row["id"],
            "start_time": round_start,
            "end_time": round_end,
            "score": vr_row["score"],
            "test_version": vr_row["test_version"],
            "rotation_speed": vr_row["rotation_speed"],
            "obj_rotation": vr_row["obj_rotation"],
            "leftover_time": leftover_time,
            "performance_metric": performance_metric,
        }
        row_features.update(eeg_features)
        row_features.update(eye_gaze_features)

        feature_rows.append(row_features)

    return pd.DataFrame(feature_rows)


def compute_performance_metric(score, rotation_speed, obj_rotation, leftover_time):
    """
    Your custom formula. As an example, let's do:
        perf = score + leftover_time - (obj_rotation / 10)  (some arbitrary formula)
    """
    return float(score) + float(leftover_time) - 0.1 * float(obj_rotation)


def plot_eye_gaze_percentages(vr_df, row_range=None):
    """
    Plots a stacked bar graph showing the percentage of time each object
    (timer, gameobj, outline, score) was looked at per round.

    Parameters:
    - vr_df: DataFrame that includes a 'start_stamp' and 'eye_interactables' column
    - row_range: tuple (start_idx, end_idx) to slice vr_df rows; if None, uses all rows
    """
    if row_range:
        vr_df = vr_df.iloc[row_range[0] : row_range[1]]

    gaze_data = []
    timestamps = []

    for _, row in vr_df.iterrows():
        eye_array = row["eye_interactables"]
        if not eye_array:
            continue
        df = pd.DataFrame(eye_array, columns=["timer", "gameobj", "outline", "score"])
        percentages = df.mean() * 100
        gaze_data.append(percentages)
        timestamps.append(row["start_stamp"])

    gaze_df = pd.DataFrame(gaze_data)
    gaze_df["start_stamp"] = timestamps
    gaze_df.set_index("start_stamp", inplace=True)

    ax = gaze_df.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title("Percentage of Gaze on Objects per Round")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Start Time")
    ax.legend(title="Object")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def clean_sensor_data(sensor_data, max_iters=5, z_thresh=3):

    s = pd.Series(sensor_data).interpolate(method="linear")
    for _ in range(max_iters):
        mean_val = s.mean()
        std_val = s.std()
        z_scores = (s - mean_val).abs() / std_val  # z = (data_point - avg)/std

        outliers = z_scores > z_thresh
        if not outliers.any():
            break

        s[outliers] = np.nan
        s = s.interpolate(method="linear")

    return sensor_data


def plot_multiple_sensors_avg(
    eeg_df,
    sensor_keys,
    chunk_size=2.0,
    impossible_threshold=150.0,
    z_thresh=1,
    max_iters=5,
):
    """
    Plots multiple sensors (e.g. alpha, beta_l, beta_h, etc.) from the same dataset,
    each in its own subplot, stacked vertically in one figure.

    Parameters:
      - eeg_df: DataFrame of EEG data (multiple rows)
      - sensor_keys: list of column names, e.g. ["af3_alpha", "af3_beta_h", ...]
      - chunk_size: size of each averaging chunk in seconds
      - impossible_threshold: values above this are discarded
      - z_thresh, max_iters: parameters for the iterative z-score cleaning
    """

    # Create a figure with len(sensor_keys) subplots, sharing the x-axis
    fig, axs = plt.subplots(
        nrows=len(sensor_keys),
        ncols=1,
        figsize=(8, 2.5 * len(sensor_keys)),
        sharex=True,
    )

    # If there's only one sensor_key, axs will be just one Axes object rather than a list
    if len(sensor_keys) == 1:
        axs = [axs]

    # We'll track the same "time_counted" across the entire dataset OR you can reset per sensor
    # For EEG that is in multiple rows, let's keep it sensor-by-sensor but across all rows

    for i, sensor_key in enumerate(sensor_keys):
        time_counted = 0.0
        time_points = []
        avg_values = []

        for _, row in eeg_df.iterrows():
            eeg_row = row.to_dict()
            sensor_data = np.array(eeg_row[sensor_key])

            # Clean the data
            s = pd.Series(sensor_data).interpolate(method="linear")
            for _ in range(max_iters):
                mean_val = s.mean()
                std_val = s.std()
                z_scores = (s - mean_val).abs() / std_val
                outliers = z_scores > z_thresh
                if not outliers.any():
                    break
                s[outliers] = np.nan
                s = s.interpolate(method="linear")

            sensor_data = s.to_numpy()

            # Timestamps
            start_time = eeg_row["start_stamp"]
            end_time = eeg_row["end_stamp"]
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)

            total_seconds = (end_time - start_time).total_seconds()
            if total_seconds <= 0:
                continue  # skip invalid

            num_samples = len(sensor_data)
            sample_rate = num_samples / total_seconds

            # Chunking loop
            t = 0.0
            while t < total_seconds:
                t_end = min(t + chunk_size, total_seconds)
                start_idx = int(t * sample_rate)
                end_idx = int(t_end * sample_rate)
                if end_idx > start_idx:
                    avg_val = sensor_data[start_idx:end_idx].mean()
                else:
                    avg_val = np.nan

                # Discard values above impossible threshold
                if avg_val < impossible_threshold:
                    time_points.append(time_counted + t + (t_end - t) / 2.0)
                    avg_values.append(avg_val)

                t += chunk_size

            time_counted += total_seconds

        # Now plot for this sensor
        axs[i].plot(time_points, avg_values, marker="", linestyle="-", label=sensor_key)
        axs[i].set_ylabel(f"{sensor_key}")
        axs[i].grid(True)
        axs[i].legend(loc="upper right")

    # Label only the bottom subplot’s x-axis
    axs[-1].set_xlabel("Time (seconds from start)")
    fig.suptitle(f"Sensors Average Every {chunk_size} Seconds", y=1.02, fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_performance(df: pd.DataFrame):
    df["timer_diff"] = df["initial_timer"] - df["end_timer"]
    # )/(15.1 - df['initial_timer']) * df["obj_size"] * 0.02 + df["end_timer"] * 0.1

    # grouped = df.groupby('session_id')['performance'].mean().reset_index()

    plt.figure()
    plt.plot(df["start_stamp"], df["timer_diff"], marker="o")
    plt.title("Timer difference")
    plt.xlabel("Start Timestamp")
    plt.ylabel("Timer Difference")
    plt.xticks(rotation=45)  # rotate x-axis labels if needed
    plt.tight_layout()  # fix layout issues
    plt.savefig("stats_imgs/timer_diff")

    df["rot_diff"] = df["obj_rotation"] - df["expected_rotation"]

    plt.figure()
    plt.plot(df["start_stamp"], df["rot_diff"], marker="o")
    plt.title("Rotation")
    plt.xlabel("Start Timestamp")
    plt.ylabel("Rotation Difference")
    plt.xticks(rotation=45)  # rotate x-axis labels if needed
    plt.tight_layout()  # fix layout issues
    plt.savefig("stats_imgs/rotation_diff")

    plt.figure()
    plt.plot(df["start_stamp"], df["obj_size"], marker="o")
    plt.title("Object_size")
    plt.xlabel("Start Timestamp")
    plt.ylabel("Performance")
    plt.xticks(rotation=45)  # rotate x-axis labels if needed
    plt.tight_layout()  # fix layout issues
    plt.savefig("stats_imgs/obj_size")

    df["timer_diff"] = df["initial_timer"] - df["end_timer"]
    df["rot_diff"] = df["obj_rotation"] - df["expected_rotation"]
    # df["obj_size"] is already present

    cols_of_interest = ["timer_diff", "rot_diff"]
    corr_matrix = df[cols_of_interest].corr()

    # Pairwise Pearson correlation with significance testing
    for col1, col2 in itertools.combinations(cols_of_interest, 2):
        # Drop any rows with NaNs in these two columns to avoid errors
        valid_data = df[[col1, col2]].dropna()
        if len(valid_data) < 2:
            # Not enough data for correlation
            print(f"Not enough data to correlate {col1} and {col2}.")
            continue

    # Compute correlation coefficient and p-value
    r_value, p_value = pearsonr(valid_data[col1], valid_data[col2])

    print(f"Correlation between {col1} and {col2}:")
    print(f"  r = {r_value:.4f}, p = {p_value:.4g}\n")
    plt.figure()
    plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest")
    plt.title("Correlation Matrix")

    # Show column labels
    plt.xticks(range(len(cols_of_interest)), cols_of_interest, rotation=45)
    plt.yticks(range(len(cols_of_interest)), cols_of_interest)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig("stats_imgs/correlation_matrix.png")


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

        filtered_vr.sort_values(by="start_stamp", inplace=True)
        filtered_vr = filtered_vr.iloc[1:].reset_index(drop=True)
        if len(filtered_vr) >= 2:
            last_two = filtered_vr.iloc[-2:]
            if (last_two["end_timer"] == 0).all():
                filtered_vr = filtered_vr.iloc[:-1].reset_index(drop=True)

        sensors_to_plot = [
            "af3_alpha",
            "af3_beta_h",
            "af3_beta_l",
            "af3_theta",
            "af3_gamma",
        ]

        # # Plot all five in one figure with stacked subplots
        # plot_multiple_sensors_avg(
        #     filtered_eeg,
        #     sensors_to_plot,
        #     chunk_size=0.3,
        #     impossible_threshold=20.0,
        #     z_thresh=1,
        #     max_iters=5,
        # )
        # sensors_to_plot = [
        #     "af3_alpha",
        #     "f7_alpha",
        #     "t7_alpha",
        #     "p7_alpha",
        #     "o1_alpha",
        #     "fc6_alpha",
        # ]

        # # Plot all five in one figure with stacked subplots
        # plot_multiple_sensors_avg(
        #     filtered_eeg,
        #     sensors_to_plot,
        #     chunk_size=0.3,
        #     impossible_threshold=20.0,
        #     z_thresh=1,
        #     max_iters=5,
        # )

        plot_performance(filtered_vr)

    finally:
        db.close()


if __name__ == "__main__":
    main_feature_extraction()
