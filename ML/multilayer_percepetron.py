import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. Load dataset
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
    "fz",
    "f3",
    "f4",
    "f7",
    "f8",
    "fc3",
    "fc4",
    "af3",
    "af4",
    "p3",
    "p4",
    "p7",
    "p8",
    "pz",
    "o1",
    "o2",
    "oz",
]
relevant_bands = ["theta", "alpha", "beta", "delta"]

def is_relevant_column(col):
    lc = col.lower()
    sensor_match = any(sensor in lc for sensor in relevant_sensors)
    band_match = any(band in lc for band in relevant_bands)
    return sensor_match and band_match


# 3. Select columns
keep_cols = [col for col in df.columns if is_relevant_column(col)]
keep_cols.append(target_col)
df_relevant = df[keep_cols].dropna()

Q1 = df_relevant[target_col].quantile(0.2)
Q3 = df_relevant[target_col].quantile(0.8)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

# Remove rows where the target is outside the bounds
df_relevant = df_relevant[(df_relevant[target_col] >= lower_bound) & (df_relevant[target_col] <= upper_bound)]

print("Original shape:", df_relevant.shape)
print("Shape after removing outliers:", df_relevant.shape)



# 4. Split data
scaler = StandardScaler()
X = scaler.fit_transform(df_relevant.drop(columns=[target_col]).values)
y = df_relevant[target_col].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def mlp():
    # 5. Build MLP for regression
    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),
            layers.Dense(8, activation="relu"),
            layers.Dense(4, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )



    # 6. Compile with regression loss & metrics
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss="mse", metrics=["mae", "mse"])

    callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,    # or so
        restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[callback],
        verbose=1
    )
    return model


model = mlp()

# 8. Evaluate
loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print("Test MSE:", mse)
print("Test MAE:", mae)


import matplotlib.pyplot as plt

y_pred = model.predict(X_test).flatten()
y_true = y_test
r2 = r2_score(y_true, y_pred)
print("R^2:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
plt.xlabel("True Performance Metric (Normalized)")
plt.ylabel("Predicted Performance Metric (Normalized)")
plt.title(f"Predictions vs True Values (R² = {r2:.2f})")
# Plot the identity line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
plt.legend()
plt.show()
