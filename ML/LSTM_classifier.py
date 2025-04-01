import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------------------------------
# 1. Load CSV Data and Drop Rows with NaN in Target
# -----------------------------------------------------
df = pd.read_csv("full_feature_table.csv", index_col=0)
# Drop rows where the performance metric is NaN
df = df.dropna(subset=["performance_metric"])

# -----------------------------------------------------
# 2. Create Categorical Labels from Performance Metric
# -----------------------------------------------------
quantiles = df["performance_metric"].quantile([0.5, 0.8]).values
df["performance_label"] = pd.cut(
    df["performance_metric"],
    bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
    labels=["bad", "average", "good"],
)

# -----------------------------------------------------
# 3. Identify EEG Array Columns
# -----------------------------------------------------
# We assume EEG columns contain one of these keywords (case insensitive)
eeg_keywords = ["theta", "alpha", "beta", "delta"]
eeg_cols = [col for col in df.columns if any(kw in col.lower() for kw in eeg_keywords)]
# Ensure target column is not included
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
# 8. Extract and Encode the Target
# -----------------------------------------------------
# IMPORTANT: Use the performance_label instead of performance_metric
y = df["performance_label"].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Now y_encoded is in {0, 1, 2}
print("Unique encoded labels:", np.unique(y_encoded))

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
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# 11. Build a Classifier Model (Using Flattened Input)
# -----------------------------------------------------
model = keras.Sequential([
    layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax"),  # 3 classes: bad, average, good
])

model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# -----------------------------------------------------
# 12. Evaluate the Model
# -----------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Optionally, print classification metrics
y_pred_classes = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))


# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_classes, alpha=0.5, label="Predictions")
# plt.xlabel("True Performance Metric (Normalized)")
# plt.ylabel("Predicted Performance Metric (Normalized)")
# plt.title(f"Predictions vs True Values (R² = {r2:.2f})")
# # Plot the identity line
# min_val = min(y_test.min(), y_pred_classes.min())
# max_val = max(y_test.max(), y_pred_classes.max())
# plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
# plt.legend()
# plt.show()
