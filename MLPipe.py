import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models.VRDataModel import VRDataModel
from models.EpocXDataModel import EpocXDataModel
from db import get_db
import numpy as np
from compute_stats.Plotting import plot_all, compute_performance
from compute_stats.ComputeEyeGaze import plot_gaze
import datetime


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
            "id": row.id,
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


def clean_sensor_data(sensor_data, max_iters=5, z_thresh=1):
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
    return s.to_numpy()


def extract_eeg_features_for_round(
    eeg_row, round_start, round_end, max_iters=5, z_thresh=3
):
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
                # Clean the sensor data before extracting features
                cleaned_data = clean_sensor_data(channel_data, max_iters, z_thresh)
                series = pd.Series(cleaned_data)
                features[f"{column_name}_mean"] = series.mean()
                features[f"{column_name}_std"] = series.std()
    return features


def extract_full_eeg_features(eeg_row, round_start, round_end, max_iters=5, z_thresh=3):
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
            if channel_data:
                # Replace raw data with cleaned data
                features[f"{column_name}"] = clean_sensor_data(
                    channel_data, max_iters, z_thresh
                )
            else:
                features[f"{column_name}"] = channel_data
    return features


def build_feature_table(vr_df, eeg_df):
    """
    Creates a final DataFrame where each row = 1 VR round + summarized EEG features.
    """
    feature_rows = []
    print(len(vr_df), len(eeg_df))
    assert len(vr_df) == len(eeg_df)

    vr_df = compute_performance(vr_df)

    for i, (vr_index, vr_row) in enumerate(vr_df.iterrows()):
        round_session_id = vr_row["session_id"]
        round_start = vr_row["start_stamp"]
        round_end = vr_row["end_stamp"]

        matching_eeg = eeg_df.iloc[i]
        if matching_eeg.empty:

            continue

        eeg_features = extract_eeg_features_for_round(
            matching_eeg, round_start, round_end
        )

        # eye_gaze_features = summarize_eye_gaze(vr_row["eye_interactables"])

        row_features = {
            "session_id": round_session_id,
            "round_id": vr_row["id"],
            "start_time": round_start,
            "end_time": round_end,
            "score": vr_row["score"],
            "test_version": vr_row["test_version"],
            "performance_metric": vr_row["performance"],
        }
        row_features.update(eeg_features)
        # row_features.update(eye_gaze_features)

        feature_rows.append(row_features)

    return pd.DataFrame(feature_rows)


def build_full_feature_table(vr_df, eeg_df):
    """
    Creates a final DataFrame where each row = 1 VR round + summarized EEG features.
    """
    feature_rows = []
    print(len(vr_df), len(eeg_df))
    assert len(vr_df) == len(eeg_df)

    vr_df = compute_performance(vr_df)

    for i, (vr_index, vr_row) in enumerate(vr_df.iterrows()):
        round_session_id = vr_row["session_id"]
        round_start = vr_row["start_stamp"]
        round_end = vr_row["end_stamp"]

        matching_eeg = eeg_df.iloc[i]
        if matching_eeg.empty:

            continue

        eeg_features = extract_full_eeg_features(matching_eeg, round_start, round_end)

        # eye_gaze_features = summarize_eye_gaze(vr_row["eye_interactables"])

        row_features = {
            "session_id": round_session_id,
            "round_id": vr_row["id"],
            "start_time": round_start,
            "end_time": round_end,
            "score": vr_row["score"],
            "test_version": vr_row["test_version"],
            "performance_metric": vr_row["performance"],
        }
        row_features.update(eeg_features)
        # row_features.update(eye_gaze_features)

        feature_rows.append(row_features)

    return pd.DataFrame(feature_rows)


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


def main_feature_extraction():
    db = next(get_db())
    try:
        vr_df, eeg_df = load_data_from_db(db)

        vr_df["start_stamp"] = pd.to_datetime(vr_df["start_stamp"]).dt.tz_localize(None)
        vr_df["end_stamp"] = pd.to_datetime(vr_df["end_stamp"]).dt.tz_localize(None)
        eeg_df["start_stamp"] = pd.to_datetime(eeg_df["start_stamp"]).dt.tz_localize(
            None
        )
        eeg_df["end_stamp"] = pd.to_datetime(eeg_df["end_stamp"]).dt.tz_localize(None)

        # Convert columns to datetime if necessary
        vr_df["start_stamp"] = pd.to_datetime(vr_df["start_stamp"])
        vr_df["end_stamp"] = pd.to_datetime(vr_df["end_stamp"])
        eeg_df["start_stamp"] = pd.to_datetime(eeg_df["start_stamp"])
        eeg_df["end_stamp"] = pd.to_datetime(eeg_df["end_stamp"])

        # Optional filtering for VR data
        filtered_vr = vr_df[vr_df["obj_size"].notnull()].copy()
        filtered_vr = filtered_vr[filtered_vr["test_version"] == 1].copy()

        # Only keep rows whose session_id is in VR data
        sessions = filtered_vr["session_id"].unique()
        filtered_eeg = eeg_df[eeg_df["session_id"].isin(sessions)].copy()

        # Sort by start_stamp
        filtered_vr = filtered_vr.sort_values(by="start_stamp").reset_index(drop=True)
        filtered_eeg = filtered_eeg.sort_values(by="start_stamp").reset_index(drop=True)

        # If we need to remove the last row(s) if the last two have end_timer == 0
        if len(filtered_vr) >= 2 and len(filtered_eeg) >= 2:
            last_two = filtered_vr.iloc[-2:]
            if (last_two["end_timer"] == 0).all():
                filtered_vr = filtered_vr.iloc[:-1].reset_index(drop=True)
                filtered_eeg = filtered_eeg.iloc[:-1].reset_index(drop=True)

        # --- Filter only today's rows ---
        today = datetime.datetime.today() - datetime.timedelta(days=2)
        filtered_vr = filtered_vr[filtered_vr["start_stamp"] >= today].copy()
        filtered_eeg = filtered_eeg[filtered_eeg["start_stamp"] >= today].copy()
        # --------------------------------

        # plot_all(filtered_vr, filtered_eeg)
        # plot_gaze(filtered_vr)
        table = build_feature_table(filtered_vr, filtered_eeg)
        table.to_csv("feature_table.csv")
        table = build_full_feature_table(filtered_vr, filtered_eeg)
        table.to_csv("full_feature_table.csv")

    finally:
        db.close()


if __name__ == "__main__":
    main_feature_extraction()
