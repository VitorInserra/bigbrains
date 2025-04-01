import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Load CSV Data and Drop Rows with NaN in Target
# -----------------------------------------------------
df = pd.read_csv("full_feature_table.csv", index_col=0)
df = df.dropna(subset=["performance_metric"])

# -----------------------------------------------------
# 2. (Regression Target) Use performance_metric as continuous target
# -----------------------------------------------------
# We no longer create categorical labels.

# -----------------------------------------------------
# 3. Identify EEG Array Columns
# -----------------------------------------------------
# We assume EEG columns contain one of these keywords (case insensitive)
eeg_keywords = ["theta", "alpha", "beta", "delta"]
eeg_cols = [col for col in df.columns if any(kw in col.lower() for kw in eeg_keywords)]
if "performance_metric" in eeg_cols:
    eeg_cols.remove("performance_metric")
print("Number of EEG feature columns:", len(eeg_cols))


# -----------------------------------------------------
# 4. Helper Functions for Parsing and Fixing Sequence Length
# -----------------------------------------------------
def parse_eeg_array(x):
    """
    Parse a string representation of a list into a numpy array.
    For example: "[2.618, 2.275, ...]" -> np.array([...])
    """
    try:
        return np.array(ast.literal_eval(x))
    except Exception as e:
        print("Error parsing:", x, "\nException:", e)
        return np.array([])


def fix_length(arr, target_length):
    """
    Truncate arr if it is longer than target_length.
    If shorter, pad with zeros at the end.
    """
    if len(arr) > target_length:
        return arr[:target_length]
    elif len(arr) < target_length:
        pad_width = target_length - len(arr)
        return np.pad(arr, (0, pad_width), mode="constant", constant_values=0)
    else:
        return arr


# -----------------------------------------------------
# 5. Optionally Filter Out Rows with Too Short EEG Arrays
# -----------------------------------------------------
min_length_threshold = 30  # set minimum time steps (adjust as needed)


def has_min_length(x):
    return len(parse_eeg_array(x)) >= min_length_threshold


# Check each EEG column; keep rows where all EEG arrays meet the minimum length
valid_mask = df[eeg_cols].applymap(has_min_length).all(axis=1)
df = df[valid_mask]

# -----------------------------------------------------
# 6. Determine Global Target Length (Time Steps)
# -----------------------------------------------------
all_lengths = []
for col in eeg_cols:
    lengths = df[col].apply(lambda x: len(parse_eeg_array(x)))
    all_lengths.extend(lengths.values)
global_target_length = int(min(all_lengths))
print("Global target length (time steps):", global_target_length)
time_steps = global_target_length
num_features = len(eeg_cols)
print("Number of EEG features (channels/bands):", num_features)

# -----------------------------------------------------
# 7. Parse EEG Data and Stack into a 3D Array
# -----------------------------------------------------
num_samples = df.shape[0]
# Preallocate an array of shape (num_samples, time_steps, num_features)
X = np.zeros((num_samples, time_steps, num_features))
for i, col in enumerate(eeg_cols):
    fixed_list = [fix_length(parse_eeg_array(x), global_target_length) for x in df[col]]
    try:
        X[:, :, i] = np.stack(fixed_list)
    except Exception as e:
        print(f"Error stacking column {col}: {e}")

# -----------------------------------------------------
# 8. Extract the Target (Continuous Performance Metric)
# -----------------------------------------------------
y = df["performance_metric"].values.astype(np.float32)
from sklearn.preprocessing import StandardScaler

# Reshape y to be a 2D array as required by StandardScaler
y = y.reshape(-1, 1)

# Create and fit the scaler on y
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
# Optionally, you could normalize y if needed (e.g., divide by 10 if scale is 0-10)

# -----------------------------------------------------
# 9. Scale EEG Data Per Feature (Across All Time Steps)
# -----------------------------------------------------
# Reshape X to 2D: (num_samples * time_steps, num_features)
X_reshaped = X.reshape(-1, num_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
# Reshape back to 3D
X = X_scaled.reshape(num_samples, time_steps, num_features)

if np.any(np.isnan(X)):
    print("Warning: X contains NaNs")

# -----------------------------------------------------
# 10. Train/Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# 11. Build a Regression Model (Using Flattened Input)
# -----------------------------------------------------
model = keras.Sequential(
    [
        layers.LSTM(16, return_sequences=True, input_shape=(time_steps, num_features)),
        layers.GlobalAveragePooling1D(),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1),  # Linear output for regression
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# -----------------------------------------------------
# 12. Evaluate the Model
# -----------------------------------------------------
loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss (MSE):", loss)
print("Test MAE:", mae)

# -----------------------------------------------------
# 13. Plot Predictions vs True Values
# -----------------------------------------------------
y_test = np.array(y_test).flatten()
y_pred = model.predict(X_test).flatten()
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
plt.xlabel("True Performance Metric")
plt.ylabel("Predicted Performance Metric")
plt.title("Regression Predictions vs True Values")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
plt.legend()
plt.show()
