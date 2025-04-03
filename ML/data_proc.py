import numpy as np
import pandas as pd

df = pd.read_csv("feature_table.csv")

target_col = "performance_metric"

# exclude_cols = [
#     "session_id",
#     "round_id",
#     "start_time",
#     "end_time",
#     "score",
#     "test_version",
# ]
relevant_sensors = [
    "f3",
    "f4",
    "fc6",
    "fc3",
    "fc4",
    # "fc5",
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
# keep_cols += ["start_time", "session_id"]
keep_cols.append(target_col)
# keep_cols.append("test_version")
# keep_cols.append("obj_rotation")
keep_cols.append("start_time")
keep_cols.append("session_id")

df_relevant = df[keep_cols].dropna()
# df_relevant["start_time"] = pd.to_datetime(df_relevant["start_time"])

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