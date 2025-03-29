import pandas as pd
from scipy.signal import medfilt
from scipy.stats import pearsonr
import itertools
import matplotlib.pyplot as plt


def plot_performance(df: pd.DataFrame):
    df["timer_diff"] = df["initial_timer"] - df["end_timer"]

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
    # df["performance"] = df["rot_ratio"]*df["obj_size"] + 0.8*(df["initial_timer"] - df["end_timer"])

    plt.figure()
    plt.plot(df["start_stamp"], df["performance"], marker="o")
    plt.title("Performance")
    plt.xlabel("Start Timestamp")
    plt.ylabel("Performance")
    plt.xticks(rotation=45)  # rotate x-axis labels if needed
    plt.tight_layout()  # fix layout issues
    plt.savefig("stats_imgs/perf")
    plt.show()

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



    db = next(get_db())
    try:
        vr_df, eeg_df = load_data_from_db(db)
        eeg_df["start_stamp"] = pd.to_datetime(eeg_df["start_stamp"]).dt.tz_localize(
            None
        )
        eeg_df["end_stamp"] = pd.to_datetime(eeg_df["end_stamp"]).dt.tz_localize(None)

        first_eeg_row = eeg_df.iloc[350].to_dict()
        session_id = first_eeg_row["session_id"]
        # print(session_id)
        # session_id = "91ab78a3-7e9e-49d2-95c9-d53e0e202c83"
        filtered_eeg = eeg_df[eeg_df["session_id"] == session_id]
        filtered_vr = vr_df[vr_df["session_id"] == session_id]

        filtered_vr.sort_values(by="start_stamp", inplace=True)
        filtered_vr = filtered_vr.iloc[1 : len(filtered_vr) - 1].reset_index(drop=True)
        if len(filtered_vr) >= 2:
            last_two = filtered_vr.iloc[-2:]
            if (last_two["end_timer"] == 0).all():
                filtered_vr = filtered_vr.iloc[:-1].reset_index(drop=True)


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

        # plot_performance(filtered_vr)