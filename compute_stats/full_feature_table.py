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

        eye_gaze_features = summarize_eye_gaze(vr_row["eye_interactables"])

        if eye_gaze_features["frac_gameobj"] == 0:
            continue

        eeg_features = extract_full_eeg_features(matching_eeg, round_start, round_end)

        row_features = {
            "session_id": round_session_id,
            "round_id": vr_row["id"],
            "start_time": round_start,
            "end_time": round_end,
            "score": vr_row["score"],
            "test_version": vr_row["test_version"],
            "obj_rotation": vr_row["obj_rotation"],
            "expected_rotation": vr_row["expected_rotation"],
            "obj_size": vr_row["obj_size"],
            "eye_timer": eye_gaze_features["frac_timer"],
            "eye_gameobj": eye_gaze_features["frac_gameobj"],
            "eye_outline": eye_gaze_features["frac_outline"],
            "eye_score": eye_gaze_features["frac_score"],
            "performance_metric": vr_row["performance"],
        }
        row_features.update(eeg_features)

        feature_rows.append(row_features)

    return pd.DataFrame(feature_rows)



def extract_full_eeg_features(eeg_row, round_start, round_end, max_iters=5, z_thresh=1):
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
