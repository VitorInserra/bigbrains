import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from models.VRDataModel import VRDataModel
from models.EpocXDataModel import EpocXDataModel
from db import get_db
from MLPipe import load_data_from_db
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.stats import pearsonr
import itertools



def summarize_eye_gaze(eye_array):
    """
    eye_array: e.g. [[0,1,0,0], [0,1,0,0], [0,0,1,0], ...]
    Each row is a 'sample' in your VR logging.

    We'll compute fraction of samples that are 1 for each column.
    """


    df = pd.DataFrame(eye_array, columns=["timer", "gameobj", "outline", "score"])
    frac_timer = df["timer"].sum()
    frac_gameobj = df["gameobj"].sum()
    frac_outline = df["outline"].sum()
    frac_score = df["score"].sum()

    print(frac_score)
    print(len(df))
    
    return [
        frac_timer/len(df),
        frac_gameobj/len(df),
        frac_outline/len(df),
        frac_score/len(df),
    ]

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
        df = pd.DataFrame(eye_array, columns=["timer", "gameobj", "outline", "score"])
        percentages = summarize_eye_gaze(df)
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
    plt.savefig("stats_imgs/eye_gaze_ratios.png")
    plt.show()


def main_feature_extraction():
    db = next(get_db())
    try:
        vr_df, eeg_df = load_data_from_db(db)
        eeg_df["start_stamp"] = pd.to_datetime(eeg_df["start_stamp"]).dt.tz_localize(
            None
        )
        eeg_df["end_stamp"] = pd.to_datetime(eeg_df["end_stamp"]).dt.tz_localize(None)

        first_eeg_row = eeg_df.iloc[380].to_dict()
        session_id = first_eeg_row["session_id"]
        print(session_id)
        # print(session_id)
        session_id = "90884ec0-ea3f-4efb-a7cb-054b6741e8d4"
        filtered_eeg = eeg_df[eeg_df["session_id"] == session_id]
        filtered_vr = vr_df[vr_df["session_id"] == session_id]

        filtered_vr.sort_values(by="start_stamp", inplace=True)
        # filtered_vr = filtered_vr.iloc[1 : len(filtered_vr) - 1].reset_index(drop=True)
        # if len(filtered_vr) >= 2:
        #     last_two = filtered_vr.iloc[-2:]
        #     if (last_two["end_timer"] == 0).all():
        #         filtered_vr = filtered_vr.iloc[:-1].reset_index(drop=True)

        plot_eye_gaze_percentages(filtered_vr)

    finally:
        db.close()


if __name__ == "__main__":
    main_feature_extraction()