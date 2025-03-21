import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models.VRDataModel import VRDataModel
from models.EpocXDataModel import EpocXDataModel
from db import get_db


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
            for band in ["theta", "alpha", "betaL", "betaH", "gamma"]:
                column_name = f"{channel}_{band}"
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
         for band in ["theta", "alpha", "betaL", "betaH", "gamma"]:
                column_name = f"{channel_name}_{band}"
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

        feature_df = build_feature_table(vr_df, eeg_df)

        print("Final Feature DF:\n", feature_df.head())
        feature_df.to_csv("feature_table.csv", index=False)
    finally:
        db.close()


if __name__ == "__main__":
    main_feature_extraction()
