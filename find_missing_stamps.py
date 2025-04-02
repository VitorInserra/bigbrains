import pandas as pd
from db import get_db
from MLPipe import load_data_from_db
import datetime


def check_vr_epoc_gaps(vr_data, epoc_data, gap_threshold=2):
    """
    Iterates over vr_data and epoc_data so that vr_data[i] is compared
    to epoc_data[j], where j starts as i+1. If the gap between
    vr_data[i].end_stamp and epoc_data[j].start_stamp > gap_threshold (sec),
    prints the row info and shifts j by an extra increment.
    """
    vr_data["start_stamp"] = pd.to_datetime(vr_data["start_stamp"]).dt.tz_localize(None)
    vr_data["end_stamp"] = pd.to_datetime(vr_data["end_stamp"]).dt.tz_localize(None)
    epoc_data["start_stamp"] = pd.to_datetime(epoc_data["start_stamp"]).dt.tz_localize(None)
    epoc_data["end_stamp"] = pd.to_datetime(epoc_data["end_stamp"]).dt.tz_localize(None)


    # Make sure these are datetime
    vr_data["start_stamp"] = pd.to_datetime(vr_data["start_stamp"])
    vr_data["end_stamp"] = pd.to_datetime(vr_data["end_stamp"])
    epoc_data["start_stamp"] = pd.to_datetime(epoc_data["start_stamp"])
    epoc_data["end_stamp"] = pd.to_datetime(epoc_data["end_stamp"])

    # Sort by start_stamp so they're in chronological order
    vr_data = vr_data.sort_values("start_stamp").reset_index(drop=True)
    epoc_data = epoc_data.sort_values("start_stamp").reset_index(drop=True)

    i = 0  # index through VR
    j = 0  # index offset for EPOC
    while i < len(vr_data) and j < len(epoc_data):
        # Compute the gap: EPOC[j].start_stamp - VR[i].end_stamp
        gap_timedelta = vr_data.loc[i, "start_stamp"] - epoc_data.loc[j, "start_stamp"]
        gap_seconds = gap_timedelta.total_seconds()

        if vr_data.loc[i, "session_id"] == epoc_data.loc[j, "session_id"] and vr_data.loc[i, "score"] != 0:
            if gap_seconds > gap_threshold:
                # Print the relevant information
                print(f"Found gap over {gap_threshold}s:")
                print(
                    f"  VR row {i}, vr_id={vr_data.loc[i, 'id']}, "
                    f"start={vr_data.loc[i, 'start_stamp']}, end={vr_data.loc[i, 'end_stamp']}"
                )
                print(
                    f"  EPOC row {j}, epoc_id={epoc_data.loc[j, 'id']}, "
                    f"start={epoc_data.loc[j, 'start_stamp']}, end={epoc_data.loc[j, 'end_stamp']}"
                )
                print(f"  Gap = {gap_seconds:.1f} seconds\n")

                # Because we found a gap, we shift EPOC index by +2 instead of +1
                j += 2
            else:
                # No big gap, proceed normally
                j += 1
        else:
            j += 1

        i += 1  # Move to the next VR row


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


        for s in sessions.tolist():
            print(s)
            t = filtered_eeg[filtered_eeg['session_id'] == s]
            d = filtered_vr[filtered_vr['session_id'] == s]
            print(len(t), len(d))

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
        # Now check for time gaps among today's records only
        check_vr_epoc_gaps(filtered_vr, filtered_eeg, gap_threshold=3)

    finally:
        db.close()


if __name__ == "__main__":
    main_feature_extraction()
