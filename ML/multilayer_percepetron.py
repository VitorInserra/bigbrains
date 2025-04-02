import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from data_proc import df_relevant

target_col = "performance_metric"

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
    opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss="mse", metrics=["mae", "mse"])

    # callback = keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=20,    # or so
    #     restore_best_weights=True
    # )
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=300,
        batch_size=32,
        # callbacks=[callback],
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
