import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast

# ------------------------------
# 1. Load CSV Data
# ------------------------------
# Use index_col=0 if the first column is an extra index.
df = pd.read_csv("full_feature_table.csv", index_col=0)

# ------------------------------
# 2. Identify EEG Array Columns
# ------------------------------
# We assume EEG columns include one of these keywords (case insensitive).
eeg_keywords = ["theta", "alpha", "beta", "gamma", "delta"]
eeg_cols = [col for col in df.columns if any(kw in col.lower() for kw in eeg_keywords)]
# Remove target column if accidentally included
if "performance_metric" in eeg_cols:
    eeg_cols.remove("performance_metric")

print("Number of EEG feature columns:", len(eeg_cols))


# ------------------------------
# 3. Define Helper Functions for Parsing and Fixing Sequence Length
# ------------------------------
def parse_eeg_array(x):
    """
    Safely parse a string representation of a list into a numpy array.
    For example, "[2.618, 2.275, ...]" -> np.array([...])
    """
    try:
        return np.array(ast.literal_eval(x))
    except Exception as e:
        print("Error parsing:", x, "\nException:", e)
        return np.array([])


def fix_length(arr, target_length):
    """
    If arr is longer than target_length, truncate it.
    If it is shorter, pad with zeros at the end.
    """
    if len(arr) > target_length:
        return arr[:target_length]
    elif len(arr) < target_length:
        pad_width = target_length - len(arr)
        return np.pad(arr, (0, pad_width), mode="constant", constant_values=0)
    else:
        return arr


# ------------------------------
# 4. Determine Global Target Length Across All EEG Arrays
# ------------------------------
all_lengths = []
for col in eeg_cols:
    # Compute the length of the parsed array for each row in this column
    lengths = df[col].apply(lambda x: len(parse_eeg_array(x)))
    all_lengths.extend(lengths.values)
global_target_length = int(min(all_lengths))
print("Global target length (time steps):", global_target_length)

time_steps = global_target_length
num_features = len(eeg_cols)
print("Number of EEG features (channels/bands):", num_features)

# ------------------------------
# 5. Parse EEG Data and Stack into a 3D Array
# ------------------------------
num_samples = df.shape[0]
# Preallocate an array of shape (num_samples, time_steps, num_features)
X = np.zeros((num_samples, time_steps, num_features))


for i, col in enumerate(eeg_cols):
    fixed_list = [fix_length(parse_eeg_array(x), global_target_length) for x in df[col]]
    try:
        X[:, :, i] = np.stack(fixed_list)
    except Exception as e:
        print(f"Error stacking column {col}: {e}")

# ------------------------------
# 6. Extract and Normalize Target
# ------------------------------
y = df["performance_metric"].values.astype(np.float32)
# Normalize performance_metric assuming it is on a scale from 0 to 10.
y = y / 1.2

# Check for NaNs in the target
if np.any(np.isnan(y)):
    print("Warning: y contains NaNs")

# ------------------------------
# 7. Scale EEG Data Per Feature
# ------------------------------
# Reshape X to 2D (combine samples and time steps) for scaling.
X_reshaped = X.reshape(
    -1, num_features
)  # shape: (num_samples * time_steps, num_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
# Reshape back to 3D
X = X_scaled.reshape(num_samples, time_steps, num_features)

# Check for NaNs in the input
if np.any(np.isnan(X)):
    print("Warning: X contains NaNs")

# ------------------------------
# 8. Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 9. Build the LSTM Model
# ------------------------------
model = keras.Sequential(
    [
        layers.Input(shape=(time_steps, num_features)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),  # Linear output for regression
    ]
)

# Use a lower learning rate and add gradient clipping to help with stability.
opt = keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
model.compile(optimizer=opt, loss="mse", metrics=["mae", "mse"])
model.summary()

# ------------------------------
# 10. Train the Model
# ------------------------------
history = model.fit(
    X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1
)

# ------------------------------
# 11. Evaluate the Model
# ------------------------------
loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print("Test MSE:", mse)
print("Test MAE:", mae)
