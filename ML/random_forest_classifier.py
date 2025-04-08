import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_proc import df_relevant
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt

target_col = "performance_metric"

# time_values = df_relevant["start_time"].values

X = df_relevant.drop(columns=[target_col]).values
y = df_relevant[target_col].values


q1, q2 = np.percentile(y, [20, 50])


def label_performance(val):
    if val <= q1:
        return "good"
    elif val <= q2:
        return "good"
    else:
        return "bad" 


y_class = np.array([label_performance(val) for val in y])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

# depths = list(range(4, 24))
# estimators = list(range(40, 240, 20))
# acc_matrix = np.zeros((len(depths), len(estimators)))

# for i_idx, depth in enumerate(depths):
#     for j_idx, n_est in enumerate(estimators):
#         rf_clf = RandomForestClassifier(
#             n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
#         )
#         rf_clf.fit(X_train, y_train)

#         y_pred = rf_clf.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)

#         acc_matrix[i_idx, j_idx] = acc

# plt.figure(figsize=(8, 6))

# plt.imshow(
#     acc_matrix,
#     origin="lower",
#     aspect="auto",
#     extent=[min(estimators), max(estimators), min(depths), max(depths)]
# )

# plt.colorbar(label="Accuracy")
# plt.xlabel("Number of Estimators (n_estimators)")
# plt.ylabel("Max Depth")
# plt.title("Random Forest Accuracy Heatmap")

# plt.show()

rf_clf = RandomForestClassifier(
        n_estimators=120, max_depth=14, random_state=42, n_jobs=-1
    )
rf_clf.fit(X_train, y_train)


final_tree = rf_clf.estimators_[-1]

dot_data = export_graphviz(
    final_tree,
    out_file=None,
    feature_names=df_relevant.drop(columns=[target_col]).columns,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("final_tree")  


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

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(rf_clf.classes_))
plt.xticks(tick_marks, rf_clf.classes_, rotation=45)
plt.yticks(tick_marks, rf_clf.classes_)

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

# results_df = pd.DataFrame({"time": time_test, "y_actual": y_test, "y_pred": y_pred})

# results_df.sort_values("time", inplace=True)

# plt.figure(figsize=(10, 6))

# # Because it's classification ("good"/"bad"), one way is a scatter plot:
# plt.scatter(results_df["time"], results_df["y_actual"], label="Actual")
# plt.scatter(results_df["time"], results_df["y_pred"], label="Predicted", marker="x")

# plt.xlabel("Time")
# plt.ylabel("Class Label")
# plt.title("Actual vs. Predicted Performance Class over Time")
# plt.legend()
# plt.tight_layout()
# plt.show()
