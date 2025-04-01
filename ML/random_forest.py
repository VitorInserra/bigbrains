import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load and prepare the dataset
df = pd.read_csv("feature_table.csv")
target_col = "performance_metric"

# EEG filtering
exclude_cols = ["session_id", "round_id", "start_time", "end_time", "score", "test_version"]
relevant_sensors = ["fz", "f3", "f4", "f7", "f8", "fc3", "fc4", "af3", "af4", "p3", "p4", "p7", "p8", "pz", "o1", "o2", "oz"]
relevant_bands = ["theta", "alpha", "beta", "delta"]

def is_relevant_column(col):
    lc = col.lower()
    return any(sensor in lc for sensor in relevant_sensors) and any(band in lc for band in relevant_bands)

keep_cols = [col for col in df.columns if is_relevant_column(col)]
keep_cols.append(target_col)
df_relevant = df[keep_cols].dropna()

# 2. Prepare features and target
X = df_relevant.drop(columns=[target_col]).values
y = df_relevant[target_col].values

# Optional: scale features (not required for RF, but helpful for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 5. Predict and evaluate
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
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top EEG Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
