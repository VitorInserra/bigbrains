#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_session_features(df):
    # Define the target session id
    target_session = "c488b456-7651-4424-8cc9-8f81ed92e0e2"

    # Filter rows for the given session id
    session_df = df[df["session_id"] == target_session]

    if session_df.empty:
        print(f"No data found for session_id {target_session}")
        sys.exit(1)

    # Sort by start_stamp (assuming this is your time reference)
    session_df = session_df.sort_values(by="start_stamp")

    # Create a figure with two subplots: one for initial_timer and another for obj_size.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # Plot initial_timer against start_stamp.
    axes[0].plot(
        session_df["start_stamp"],
        session_df["initial_timer"],
        marker="o",
        linestyle="-",
    )
    axes[0].set_title("Initial Timer for Version B")
    axes[0].set_ylabel("Initial Timer")
    # Set y-axis limit for the timer plot from 0 to 18.
    axes[0].set_ylim(0, 18)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot obj_size against start_stamp.
    if "obj_size" in session_df.columns:
        axes[1].plot(
            session_df["start_stamp"],
            session_df["obj_size"],
            marker="o",
            linestyle="-",
            color="orange",
        )
        axes[1].set_title("Object Size")
        axes[1].set_xlabel("Start Stamp")
        axes[1].set_ylabel("Object Size")
        # Set y-axis limit for the object size plot from 0 to 50.
        axes[1].set_ylim(0, 50)
        axes[1].grid(True, linestyle="--", alpha=0.6)
    else:
        axes[1].text(0.5, 0.5, "Column 'obj_size' not found", ha="center", va="center")

    # Improve layout.
    plt.tight_layout()

    # Save the figure as a file and display it.
    plt.savefig("session_plots.png")
    print("Plots saved as 'session_plots.png'")
    plt.show()

