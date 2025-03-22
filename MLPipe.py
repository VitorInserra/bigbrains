import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models.VRDataModel import VRDataModel
from models.EpocXDataModel import EpocXDataModel
from db import get_db
import matplotlib.pyplot as plt
import numpy as np


def load_data_from_db(db_session: Session):
    """
    Loads VR data and EEG data from the database.

    Returns:
      - vr_df: A DataFrame with VR session details (one row per VR interaction).
      - eeg_df: A DataFrame with EEG power band values, session IDs, and timestamps.
    """

    # Load VR data
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
            "description": row.description,
            "eye_interactables": row.eye_interactables,
        }
        for row in vr_rows
    ]
    vr_df = pd.DataFrame(vr_data)

    # Load EEG data
    eeg_rows = db_session.query(EpocXDataModel).all()

    eeg_data = []
    for row in eeg_rows:
        row_dict = {
            "session_id": row.session_id,
            "start_stamp": row.start_stamp,
            "end_stamp": row.end_stamp,
        }

        # Dynamically extract all EEG channels & band values
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
                row_dict[column_name] = getattr(
                    row, column_name, None
                )  # Get attribute dynamically

        eeg_data.append(row_dict)

    eeg_df = pd.DataFrame(eeg_data)

    return vr_df, eeg_df


def extract_eeg_features_for_round(eeg_row, round_start, round_end):
    """
    Example: If you saved 'af3' as an array of band-power samples that exactly
    matches round_start→round_end, you can compute summary stats here.
    """
    features = {}

    # For each channel, compute stats
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
            channel_data = eeg_row[column_name]  # This is a list of floats
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
    # fraction of '1' for each column
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

        # 1) Find the matching EEG row(s) for that session & time.
        #    This example assumes exactly one row in eeg_df for each round (by session_id).
        #    If you have multiple or a single big chunk, you'd need a more careful approach.
        matching_eeg = eeg_df[
            (eeg_df["session_id"] == round_session_id)
            & (eeg_df["start_stamp"] >= round_start)
            & (eeg_df["end_stamp"] <= round_end)
        ]

        if matching_eeg.empty:
            # If no matching EEG, skip or handle gracefully
            continue

        # We'll assume one row, or just take the first if multiple
        eeg_row = matching_eeg.iloc[0].to_dict()

        # 2) Extract EEG features
        eeg_features = extract_eeg_features_for_round(eeg_row, round_start, round_end)

        # 3) Summarize eye gaze
        eye_gaze_features = summarize_eye_gaze(vr_row["eye_interactables"])

        # 4) Compute your performance label or metrics
        #    For example, let's define "leftover_time" or any composite
        leftover_time = vr_row[
            "end_timer"
        ]  # or (vr_row["end_timer"] / vr_row["initial_timer"])
        # or define your own formula combining rotation_speed, obj_rotation, etc.
        # Example:
        performance_metric = compute_performance_metric(
            score=vr_row["score"],
            rotation_speed=vr_row["rotation_speed"],
            obj_rotation=vr_row["obj_rotation"],
            leftover_time=leftover_time,
        )

        # 5) Combine everything into one dictionary
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
        percentages = df.mean() * 100  # Convert to percentage
        gaze_data.append(percentages)
        timestamps.append(row["start_stamp"])

    # Create DataFrame for plotting
    gaze_df = pd.DataFrame(gaze_data)
    gaze_df["start_stamp"] = timestamps
    gaze_df.set_index("start_stamp", inplace=True)

    # Plot
    ax = gaze_df.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title("Percentage of Gaze on Objects per Round")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Start Time")
    ax.legend(title="Object")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def plot_sensor_avg(eeg_df, sensor_key, chunk_size=2.0):
    """
    For a single EEG row (dictionary) that contains a sensor's data as a list of floats,
    this function divides the total time span (from start_stamp to end_stamp) into 2-second chunks,
    computes the average reading of the sensor in each chunk, and plots the time series.

    Parameters:
      - eeg_row: dict with keys "start_stamp", "end_stamp", and sensor data under sensor_key.
      - sensor_key: string key for the sensor (e.g. "AF3_alpha")
      - chunk_size: time window in seconds for averaging (default=2.0 seconds)
    """

    time_counted = 0.0
    time_points = []  # X-axis: time (in seconds from start)
    avg_values = []  # Y-axis: average sensor reading for the chunk

    for _, row in eeg_df.iterrows():
        eeg_row = row.to_dict()
        # Get sensor data as numpy array.
        sensor_data = np.array(eeg_row[sensor_key])

        # Ensure start_stamp and end_stamp are datetime objects.
        start_time = eeg_row["start_stamp"]
        end_time = eeg_row["end_stamp"]
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)

        total_seconds = (end_time - start_time).total_seconds()
        if total_seconds <= 0:
            print("Invalid timestamps; total duration must be positive.")
            return

        num_samples = len(sensor_data)
        sample_rate = num_samples / total_seconds  # samples per second

        t = 0.0
        while t < total_seconds:
            t_end = min(t + chunk_size, total_seconds)
            start_idx = int(t * sample_rate)
            end_idx = int(t_end * sample_rate)
            if end_idx > start_idx:
                avg_val = sensor_data[start_idx:end_idx].mean()
            else:
                avg_val = np.nan
            # Using midpoint of chunk for plotting.
            time_points.append(time_counted + t + (t_end - t) / 2.0)
            avg_values.append(avg_val)
            t += chunk_size

        time_counted += total_seconds

    plt.figure(figsize=(8, 4))
    plt.plot(time_points, avg_values, marker="o", linestyle="-")
    plt.xlabel("Time (seconds from start)")
    plt.ylabel(f"{sensor_key} Average Reading in uV^2 / Hz")
    plt.title(f"{sensor_key} Average Every {chunk_size} Seconds")
    plt.grid(True)
    plt.show()


# Example call (for a single EEG row):
# Assuming 'eeg_df' is your DataFrame and you want to plot "AF3_alpha" from the first row:
def main_feature_extraction():
    db = next(get_db())
    try:
        _, eeg_df = load_data_from_db(db)
        eeg_df["start_stamp"] = pd.to_datetime(eeg_df["start_stamp"]).dt.tz_localize(
            None
        )
        eeg_df["end_stamp"] = pd.to_datetime(eeg_df["end_stamp"]).dt.tz_localize(None)

        first_eeg_row = eeg_df.iloc[96].to_dict()
        filtered_df = eeg_df[eeg_df["session_id"] == first_eeg_row["session_id"]]
        plot_sensor_avg(filtered_df, "af3_alpha", chunk_size=0.3)
        plot_sensor_avg(filtered_df, "af3_beta_h", chunk_size=0.3)
        plot_sensor_avg(filtered_df, "af3_beta_l", chunk_size=0.3)

    finally:
        db.close()


    # feature_df = build_feature_table(vr_df, eeg_df)
    # plot_eye_gaze_percentages(vr_df, row_range=(0, 200))

    # print("Final Feature DF:\n", feature_df.head())
    # feature_df.to_csv("feature_table.csv", index=False)
if __name__ == "__main__":
    main_feature_extraction()
