import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models.VRDataModel import VRDataModel
from models.EpocXDataModel import EpocXDataModel
from db import get_db
import numpy as np


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



def build_feature_table(vr_df, eeg_df):
    """
    Creates a final DataFrame where each row = 1 VR round + summarized EEG features.
    """
    feature_rows = []
    print(len(vr_df), len(eeg_df))
    assert len(vr_df) == len(eeg_df)


    vr_df = compute_performance(vr_df)

    for _, vr_row in vr_df.iterrows():
        round_session_id = vr_row["session_id"]
        round_start = vr_row["start_stamp"]
        round_end = vr_row["end_stamp"]

        print(eeg_df.iloc[_])
        matching_eeg = eeg_df.iloc[_]
        if matching_eeg.empty:

            continue

        #TODO: add performance


        eeg_features = extract_eeg_features_for_round(matching_eeg, round_start, round_end)

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
    print(vr_df["performance"])

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


def compute_performance(df: pd.DataFrame):
    df["rot_ratio"] = df["expected_rotation"] / df["obj_rotation"]

    # 20% tolerance on rotation gives us 252/360
    df["rot_ratio"] = np.where(df["rot_ratio"] > (360 / 252), 0, df["rot_ratio"])
    avg = np.mean(df["rot_ratio"])
    df["rot_ratio"] = np.where(df["rot_ratio"] == 0, avg, df["rot_ratio"])

    df["performance"] = (
        df["rot_ratio"]
        * df["obj_size"]
        * (df["initial_timer"] - df["end_timer"])
        / df["initial_timer"]
    )

    return df



def main_feature_extraction():
    db = next(get_db())
    try:
        vr_df, eeg_df = load_data_from_db(db)

        filtered_vr = vr_df[vr_df["obj_size"].notnull()]
        filtered_vr = filtered_vr[filtered_vr["test_version"] == 1]


        sessions = filtered_vr['session_id'].unique()
        filtered_eeg = eeg_df[eeg_df["session_id"].isin(sessions)]
        filtered_eeg.sort_values(by="start_stamp", inplace=True)
        filtered_vr.sort_values(by="start_stamp", inplace=True)

        
        if len(filtered_vr) >= 2 and len(filtered_eeg) >= 2:
            last_two = filtered_vr.iloc[-2:]
            if (last_two["end_timer"] == 0).all():
                filtered_vr = filtered_vr.iloc[:-1].reset_index(drop=True)
                filtered_eeg = filtered_eeg.iloc[:-1].reset_index(drop=True)
        
        filtered_vr = filtered_vr.reset_index(drop=True)
        filtered_eeg = filtered_eeg.reset_index(drop=True)

        build_feature_table(filtered_vr, filtered_eeg).to_csv('feature_table.csv', index=False)

    finally:
        db.close()


if __name__ == "__main__":
    main_feature_extraction()