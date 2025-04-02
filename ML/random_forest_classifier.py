import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_proc import df_relevant

target_col = "performance_metric"

# 1. Prepare features and target
X = df_relevant.drop(columns=[target_col]).values
y = df_relevant[target_col].values

# 2. Transform continuous target into categorical labels
# Define thresholds using quantiles (33.33% and 66.66% percentiles)
q1, q2 = np.percentile(y, [20, 50])

def label_performance(val):
    if val <= q1:
        return "good"
    elif val <= q2:
        return "good"
    else:
        return "bad"      # Lower values are considered good

y_class = np.array([label_performance(val) for val in y])

# 3. Optional: scale features (RF doesn't require scaling but it can help for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

# 5. Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# 6. Predict and evaluate the classifier
y_pred = rf_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Random Forest Classification Performance:")
print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

import matplotlib.pyplot as plt

# 7. Plot Feature Importances
feature_importances = rf_clf.feature_importances_
feature_names = df_relevant.drop(columns=[target_col]).columns

sorted_idx = np.argsort(feature_importances)[::-1][:15]  # top 15 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top EEG Feature Importances (Random Forest Classifier)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 8. Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(rf_clf.classes_))
plt.xticks(tick_marks, rf_clf.classes_, rotation=45)
plt.yticks(tick_marks, rf_clf.classes_)

# Add numbers inside each square of the confusion matrix
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
