import numpy as np
import pandas as pd

df = pd.read_csv("feature_table.csv")

target_col = "performance_metric"

# 2. Identify relevant EEG features
exclude_cols = [
    "session_id",
    "round_id",
    "start_time",
    "end_time",
    "score",
    "test_version",
]
relevant_sensors = [
    "f3",
    "f4",
    "fc3",
    "fc4",
    "af3",
    "af4",
    "o1",
    "o2",
]
relevant_bands = ["theta", "alpha", "beta"]

def is_relevant_column(col):
    lc = col.lower()
    sensor_match = any(sensor in lc for sensor in relevant_sensors)
    band_match = any(band in lc for band in relevant_bands)
    return sensor_match and band_match


# 3. Select columns
keep_cols = []
keep_cols = [col for col in df.columns if is_relevant_column(col)]
keep_cols += ["obj_rotation"] #, "eye_gameobj"]
keep_cols.append(target_col)
df_relevant = df[keep_cols].dropna()

Q1 = df_relevant[target_col].quantile(0.1)
Q3 = df_relevant[target_col].quantile(0.9)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

# Remove rows where the target is outside the bounds
df_relevant = df_relevant[(df_relevant[target_col] >= lower_bound) & (df_relevant[target_col] <= upper_bound)]

print("Original shape:", df_relevant.shape)
print("Shape after removing outliers:", df_relevant.shape)