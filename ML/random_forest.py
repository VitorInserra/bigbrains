import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_proc import df_relevant

target_col = "performance_metric"

X = df_relevant.drop(columns=[target_col]).values
y = df_relevant[target_col].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Performance:")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")


import matplotlib.pyplot as plt

feature_importances = rf.feature_importances_
feature_names = df_relevant.drop(columns=[target_col]).columns

sorted_idx = np.argsort(feature_importances)[::-1][:15]  # top 15 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top EEG Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


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
