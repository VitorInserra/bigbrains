# random_forest_classifier.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_proc import df_relevant
import matplotlib.pyplot as plt

target_col = "performance_metric"

# ------------------------------------------------
# 1) Split by session_id
# ------------------------------------------------
TARGET_SESSION = "3e4a23a8-c0e7-4ce7-ae53-eeaf9f4c0fd5"

train_df = df_relevant[df_relevant["session_id"] != TARGET_SESSION]
test_df = df_relevant[df_relevant["session_id"] == TARGET_SESSION]

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ------------------------------------------------
# 2) Separate features, target, and time in each set
# ------------------------------------------------
# We'll keep 'start_time' separate so we can plot by time later.
X_train = train_df.drop(columns=[target_col, "session_id", "start_time"]).values
y_train = train_df[target_col].values

X_test = test_df.drop(columns=[target_col, "session_id", "start_time"]).values
y_test = test_df[target_col].values

# Store time for plotting
time_test = test_df["start_time"].values

# ------------------------------------------------
# 3) Turn continuous target into 'good'/'bad' labels
#    (Quantiles computed from training set)
# ------------------------------------------------
q1, q2 = np.percentile(y_train, [20, 50])


def label_performance(val):
    # Lower = "good", higher = "bad"
    if val <= q1:
        return "good"
    elif val <= q2:
        return "good"
    else:
        return "bad"


y_train_class = np.array([label_performance(val) for val in y_train])
y_test_class = np.array([label_performance(val) for val in y_test])

# ------------------------------------------------
# 4) Scale features
# ------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# 5) Train the Random Forest on train_df only
# ------------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
)
rf_clf.fit(X_train_scaled, y_train_class)

# ------------------------------------------------
# 6) Predict on the held-out session
# ------------------------------------------------
y_pred = rf_clf.predict(X_test_scaled)

acc = accuracy_score(y_test_class, y_pred)
report = classification_report(y_test_class, y_pred)
cm = confusion_matrix(y_test_class, y_pred)

print("Random Forest Classification Performance (held-out session):")
print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

# ------------------------------------------------
# 7) Plot Feature Importances
# ------------------------------------------------
feature_importances = rf_clf.feature_importances_

# We dropped [target_col, session_id, start_time]
# So let's build feature_names from that subset:
feature_cols = train_df.drop(columns=[target_col, "session_id", "start_time"]).columns
sorted_idx = np.argsort(feature_importances)[::-1][:15]  # top 15

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top EEG Feature Importances (Random Forest Classifier)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 8) Plot Confusion Matrix
# ------------------------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Held-Out Session)")
plt.colorbar()
tick_marks = np.arange(len(rf_clf.classes_))
plt.xticks(tick_marks, rf_clf.classes_, rotation=45)
plt.yticks(tick_marks, rf_clf.classes_)

# Add numbers inside each square of the confusion matrix
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(
        j,
        i,
        format(cm[i, j], "d"),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 9) Plot Actual vs. Predicted over Time
# ------------------------------------------------
results_df = pd.DataFrame(
    {"time": time_test, "y_actual": y_test_class, "y_pred": y_pred}
)

results_df.sort_values("time", inplace=True)

plt.figure(figsize=(10, 6))
# For classification, scatter is more typical:
plt.scatter(results_df["time"], results_df["y_actual"], label="Actual")
plt.scatter(results_df["time"], results_df["y_pred"], label="Predicted", marker="x")

plt.xlabel("Time")
plt.ylabel("Class Label")
plt.title("Actual vs. Predicted Performance Class over Time")
plt.legend()
plt.tight_layout()
plt.show()
