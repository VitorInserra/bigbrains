import numpy as np
import pandas as pd
import itertools
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Example: read the data
df = pd.read_csv("feature_table.csv")

target_col = "performance_metric"

all_sensors = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    # "T7",
    # "P7",
    # "O1",
    "O2",
    # "P8",
    # "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
all_bands = ["theta", "alpha", "beta_l", "beta_h", "delta"]


# ------------------------------------------------------------------
# Helper to generate all non-empty subsets of a given list
def all_nonempty_subsets(items):
    """Returns a generator over all non-empty subsets of `items`."""
    for r in range(3, 5):
        for combo in itertools.combinations(items, r):
            yield list(combo)


# ------------------------------------------------------------------
def columns_for_subset(df, sensors_subset, bands_subset):
    """
    Return columns from df that contain *any* of the sensor names in `sensors_subset`
    AND also contain *any* of the band names in `bands_subset`.
    """
    sensors_subset_lower = [s.lower() for s in sensors_subset]
    bands_subset_lower = [b.lower() for b in bands_subset]

    def matches_any_sensor_band(col_name):
        lower_col = col_name.lower()
        return (
            any(sensor in lower_col for sensor in sensors_subset_lower)
            and any(band in lower_col for band in bands_subset_lower)
        )

    subset_cols = [c for c in df.columns if matches_any_sensor_band(c)]
    return subset_cols


# ------------------------------------------------------------------
def build_relevant_dataframe(df, sensors_subset, bands_subset, target_col):
    """
    1) Pick columns relevant to the chosen sensors and bands
    2) Drop NaNs
    3) Remove outliers (based on target_col)
    4) Return the filtered DataFrame
    """
    chosen_cols = columns_for_subset(df, sensors_subset, bands_subset)

    # Always include target col
    if target_col not in chosen_cols:
        chosen_cols.append(target_col)

    df_temp = df[chosen_cols].dropna().copy()

    # Outlier filter
    Q1 = df_temp[target_col].quantile(0.1)
    Q3 = df_temp[target_col].quantile(0.9)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_temp = df_temp[
        (df_temp[target_col] >= lower_bound) & (df_temp[target_col] <= upper_bound)
    ]

    return df_temp


# ------------------------------------------------------------------
def label_performance_values(y):
    """
    Convert the performance metric to "good"/"bad" based on the 20th and 50th percentiles.
    """
    q1, q2 = np.percentile(y, [20, 50])

    def label(val):
        if val <= q1:
            return "good"
        elif val <= q2:
            return "good"
        else:
            return "bad"

    return np.array([label(val) for val in y])


# ------------------------------------------------------------------
# Create lists of subsets
all_sensor_subsets = list(all_nonempty_subsets(all_sensors))
all_band_subsets = list(all_nonempty_subsets(all_bands))

depths = list(range(6, 15))             # 6..11
estimators = list(range(100, 141, 20))   # 80,100,120,140

# Calculate total number of runs
num_sensor_subsets = len(all_sensor_subsets)   # (2^14 - 1) = 16383
num_band_subsets = len(all_band_subsets)       # (2^5  - 1) = 31
total_runs = num_sensor_subsets * num_band_subsets * len(depths) * len(estimators)

progress_count = 0

# Initialize the best trackers
best_accuracy = -1
best_sensors_combo = None
best_bands_combo = None
best_depth = None
best_n_est = None

# Where we store any newly found best result
BEST_RESULT_CSV = "best_result.csv"

for sensor_subset in all_sensor_subsets:
    for band_subset in all_band_subsets:
        df_temp = build_relevant_dataframe(df, sensor_subset, band_subset, target_col)

        for depth in depths:
            for n_est in estimators:
                progress_count += 1
                loaded_percentage = (progress_count / total_runs) * 100.0
                print(f"Loaded: {loaded_percentage:.4f}%")

                # Check feasibility
                if df_temp.shape[1] <= 1:
                    # Only target col or none
                    continue
                if df_temp.shape[0] < 10:
                    # Not enough data
                    continue

                # Prepare X, y
                y = df_temp[target_col].values
                X = df_temp.drop(columns=[target_col]).values

                # Convert to "good"/"bad"
                y_class = label_performance_values(y)

                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_class, test_size=0.2, random_state=42
                )

                # Train the RandomForest
                rf_clf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=42,
                    n_jobs=-1,
                )
                rf_clf.fit(X_train, y_train)

                # Evaluate
                y_pred = rf_clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # If new best, update and APPEND to CSV
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_sensors_combo = sensor_subset
                    best_bands_combo = band_subset
                    best_depth = depth
                    best_n_est = n_est

                    print(f"New best accuracy: {best_accuracy:.4f}")

                    best_result = {
                        "best_accuracy": [best_accuracy],
                        "best_sensors": [best_sensors_combo],
                        "best_bands": [best_bands_combo],
                        "best_depth": [best_depth],
                        "best_n_estimators": [best_n_est],
                    }
                    best_df = pd.DataFrame(best_result)

                    # If "best_result.csv" already exists, append
                    if os.path.exists(BEST_RESULT_CSV):
                        existing = pd.read_csv(BEST_RESULT_CSV)
                        updated = pd.concat([existing, best_df], ignore_index=True)
                        updated.to_csv(BEST_RESULT_CSV, index=False)
                    else:
                        # Otherwise, create it
                        best_df.to_csv(BEST_RESULT_CSV, index=False)
