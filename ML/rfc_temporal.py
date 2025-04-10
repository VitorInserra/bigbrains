# random_forest_classifier.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_proc import df_relevant
import matplotlib.pyplot as plt

target_col = "performance_metric"


def rfc(target_session):
    train_df = df_relevant[df_relevant["session_id"] != target_session]
    test_df = df_relevant[df_relevant["session_id"] == target_session]

    # print("Train shape:", train_df.shape)
    # print("Test shape:", test_df.shape)
    X_train = train_df.drop(
        columns=[target_col, "session_id", "start_time", "test_version"]
    ).values
    y_train = train_df[target_col].values

    X_test = test_df.drop(
        columns=[target_col, "session_id", "start_time", "test_version"]
    ).values
    y_test = test_df[target_col].values

    time_test = test_df["start_time"].values


    q1, q2 = np.percentile(y_train, [20, 50])

    def label_performance(val):
        if val <= q1:
            return "good"
        elif val <= q2:
            return "good"
        else:
            return "bad"

    y_train_class = np.array([label_performance(val) for val in y_train])
    y_test_class = np.array([label_performance(val) for val in y_test])



    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    rf_clf = RandomForestClassifier(
        n_estimators=120, max_depth=14, random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train_class)



    y_pred = rf_clf.predict(X_test_scaled)

    acc = accuracy_score(y_test_class, y_pred)
    report = classification_report(y_test_class, y_pred)
    cm = confusion_matrix(y_test_class, y_pred)

    # print("Random Forest Classification Performance (held-out session):")
    # print(f"Test Accuracy: {acc:.4f}")
    # print("Classification Report:")
    # print(report)
    # print("Confusion Matrix:")
    # print(cm)



    # feature_importances = rf_clf.feature_importances_

    # We dropped [target_col, session_id, start_time]
    # So let's build feature_names from that subset:
    # feature_cols = train_df.drop(columns=[target_col, "session_id", "start_time"]).columns
    # sorted_idx = np.argsort(feature_importances)[::-1][:15]  # top 15

    # plt.figure(figsize=(10, 6))
    # plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
    # plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
    # plt.xlabel("Feature Importance")
    # plt.title("Top EEG Feature Importances (Random Forest Classifier)")
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # plt.show()


    # # 8) Plot Confusion Matrix

    # plt.figure(figsize=(8, 6))
    # plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix (Held-Out Session)")
    # plt.colorbar()
    # tick_marks = np.arange(len(rf_clf.classes_))
    # plt.xticks(tick_marks, rf_clf.classes_, rotation=45)
    # plt.yticks(tick_marks, rf_clf.classes_)

    # # Add numbers inside each square of the confusion matrix
    # thresh = cm.max() / 2.0
    # for i, j in np.ndindex(cm.shape):
    #     plt.text(
    #         j,
    #         i,
    #         format(cm[i, j], "d"),
    #         horizontalalignment="center",
    #         color="white" if cm[i, j] > thresh else "black",
    #     )

    # plt.ylabel("True label")
    # plt.xlabel("Predicted label")
    # plt.tight_layout()
    # plt.show()


    version = test_df.iloc[0]["test_version"]
    results_df = pd.DataFrame(
        {"time": time_test, "y_actual": y_test_class, "y_pred": y_pred}
    )

    results_df.sort_values("time", inplace=True)

    # plt.figure(figsize=(10, 6))
    # # For classification, scatter is more typical:
    # plt.plot(
    #     results_df["time"],
    #     results_df["y_actual"],
    #     label="Actual",
    #     linestyle="-",
    #     marker="o",
    # )
    # plt.scatter(results_df["time"], results_df["y_pred"], label="Predicted", marker="x")

    # plt.xlabel("Time")
    # plt.ylabel("Class Label")
    # plt.title("Actual vs. Predicted Performance Class over Time")
    # plt.legend()

    # Add count and percentage labels for actual and predicted

    from collections import Counter

    # def get_label_stats(labels):
    #     counts = Counter(labels)
    #     total = sum(counts.values())
    #     good = counts.get("good", 0)
    #     bad = counts.get("bad", 0)
    #     return {"good": f"{good} ({good/total:.1%})", "bad": f"{bad} ({bad/total:.1%})"}

    # actual_stats = get_label_stats(results_df["y_actual"])
    # pred_stats = get_label_stats(results_df["y_pred"])

    # # Position for the text box
    # text_x = results_df["time"].min()
    # text_y = 1.2  # Above 'bad' line (1)

    # summary_text = (
    #     f"Actual:\n"
    #     f"  good: {actual_stats['good']}\n"
    #     f"  bad:  {actual_stats['bad']}\n\n"
    #     f"Predicted:\n"
    #     f"  good: {pred_stats['good']}\n"
    #     f"  bad:  {pred_stats['bad']}"
    # )

    # plt.text(
    #     text_x,
    #     text_y,
    #     summary_text,
    #     fontsize=10,
    #     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    # )

    def get_label_stats(labels):
        counts = Counter(labels)
        total = sum(counts.values())
        good = counts.get("good", 0)
        bad = counts.get("bad", 0)
        return {"good": good / total, "bad": bad / total, "total": total}

    actual_stats = get_label_stats(results_df["y_actual"])
    pred_stats = get_label_stats(results_df["y_pred"])

    return {
        "test_version": int(version),
        "pred": pred_stats,
        "actual": actual_stats,
        "acc": acc,
    }


t1_pred = {"good": 0, "bad": 0}
t1_actual = {"good": 0, "bad": 0}
t2_pred = {"good": 0, "bad": 0}
t2_actual = {"good": 0, "bad": 0}
acc1 = 0
acc2 = 0

t1_count = 0
t2_count = 0

t1_total = 0
t2_total = 0

# for session_id in df_relevant["session_id"].unique():

#     summary = rfc(session_id)

#     if summary["test_version"] == 1:
#         t1_count += 1
#         t1_pred["good"] += summary["pred"]["good"]
#         t1_pred["bad"] += summary["pred"]["bad"]
#         t1_actual["good"] += summary["actual"]["good"]
#         t1_actual["bad"] += summary["actual"]["bad"]
#         acc1 += summary["acc"]
#         t1_total += summary["pred"]["total"]

#     elif summary["test_version"] == 2:
#         t2_count += 1
#         t2_pred["good"] += summary["pred"]["good"]
#         t2_pred["bad"] += summary["pred"]["bad"]
#         t2_actual["good"] += summary["actual"]["good"]
#         t2_actual["bad"] += summary["actual"]["bad"]
#         acc2 += summary["acc"]
#         t2_total += summary["actual"]["total"]

for session_id in df_relevant["session_id"].unique():

    summary = rfc(session_id)

    if summary["test_version"] == 1:
        t1_count += 1
        t1_pred["good"] += summary["pred"]["good"]*summary["pred"]["total"]
        t1_pred["bad"] += summary["pred"]["bad"]*summary["pred"]["total"]
        t1_actual["good"] += summary["actual"]["good"]*summary["pred"]["total"]
        t1_actual["bad"] += summary["actual"]["bad"]*summary["pred"]["total"]
        acc1 += summary["acc"]*summary["pred"]["total"] 
        t1_total += summary["pred"]["total"]

    elif summary["test_version"] == 2:
        t2_count += 1
        t2_pred["good"] += summary["pred"]["good"]*summary["pred"]["total"]
        t2_pred["bad"] += summary["pred"]["bad"]*summary["pred"]["total"]
        t2_actual["good"] += summary["actual"]["good"]*summary["pred"]["total"]
        t2_actual["bad"] += summary["actual"]["bad"]*summary["pred"]["total"]
        acc2 += summary["acc"]*summary["pred"]["total"]
        t2_total += summary["actual"]["total"]


t1_pred["good"] /= t1_total
t1_pred["bad"] /= t1_total
t1_actual["good"] /= t1_total
t1_actual["bad"] /= t1_total
acc1 /= t1_total
print(t1_pred, t1_actual, acc1, t2_total)

t2_pred["good"] /= t2_total
t2_pred["bad"] /= t2_total
t2_actual["good"] /= t2_total
t2_actual["bad"] /= t2_total
acc2 /= t2_total
print(t2_pred, t2_actual, acc2, t2_total)


labels = ["Version A", "Version B"]
actual_good = [t1_actual["good"], t2_actual["good"]]
pred_good = [t1_pred["good"], t2_pred["good"]]

x = range(len(labels))
width = 0.4

plt.figure(figsize=(12, 8))
plt.bar([i + width for i in x], actual_good, width, label="Actual Good")
plt.bar(x, pred_good, width, label="Predicted Good")

plt.xticks([i + width/2 for i in x], labels)
plt.ylabel("Proportion of Good Results")
plt.title("Good Results Comparison")

legend_info = (
    f"A) N trials: {t1_total}, Cross-validation Prediction Accuracy: {acc1*100:.3f}%\n"
    f"B) N trials {t2_total},  Cross-validation Prediction Accuracy {acc2*100:.3f}%"
)

plt.legend(title=legend_info)

plt.show()

labels = ["Version A", "Version B"]
pred_bad = [t1_pred["bad"], t2_pred["bad"]]
actual_bad = [t1_actual["bad"], t2_actual["bad"]]

x = range(len(labels))
width = 0.4

plt.figure(figsize=(12, 8))
plt.bar([i + width for i in x], actual_bad, width, label="Actual Bad")
plt.bar(x, pred_bad, width, label="Predicted Bad")

plt.xticks([i + width/2 for i in x], labels)
plt.ylabel("Proportion of Bad Results")
plt.title("Bad Results Comparison")

legend_info = (
    f"A) N trials: {t1_total}, Cross-validation Prediction Accuracy: {acc1*100:.3f}%\n"
    f"B) N trials {t2_total},  Cross-validation Prediction Accuracy {acc2*100:.3f}%"
)

plt.legend(title=legend_info)

plt.show()




t1_pred = {"good": 0, "bad": 0}
t1_actual = {"good": 0, "bad": 0}
t2_pred = {"good": 0, "bad": 0}
t2_actual = {"good": 0, "bad": 0}
acc1 = 0
acc2 = 0

t1_count = 0
t2_count = 0

t1_total = 0
t2_total = 0

for session_id in df_relevant["session_id"].unique():

    summary = rfc(session_id)

    if summary["test_version"] == 1:
        t1_count += 1
        t1_pred["good"] += summary["pred"]["good"]*summary["pred"]["total"]
        t1_pred["bad"] += summary["pred"]["bad"]*summary["pred"]["total"]
        t1_actual["good"] += summary["actual"]["good"]*summary["pred"]["total"]
        t1_actual["bad"] += summary["actual"]["bad"]*summary["pred"]["total"]
        acc1 += summary["acc"]
        t1_total += summary["pred"]["total"]

    elif summary["test_version"] == 2:
        t2_count += 1
        t2_pred["good"] += summary["pred"]["good"]*summary["pred"]["total"]
        t2_pred["bad"] += summary["pred"]["bad"]*summary["pred"]["total"]
        t2_actual["good"] += summary["actual"]["good"]*summary["pred"]["total"]
        t2_actual["bad"] += summary["actual"]["bad"]*summary["pred"]["total"]
        acc2 += summary["acc"]
        t2_total += summary["actual"]["total"]

from scipy.stats import chi2_contingency

print(t1_total, t2_total)

good_pred_t1_count = t1_pred["good"] #* t1_total
bad_pred_t1_count  = t1_pred["bad"]   #* t1_total
good_pred_t2_count = t2_pred["good"] #* t2_total
bad_pred_t2_count  = t2_pred["bad"]  #* t2_total


good_actual_t1_count = t1_actual["good"] #* t1_total
bad_actual_t1_count  = t1_actual["bad"]   #* t1_total
good_actual_t2_count = t2_actual["good"] #* t2_count #* t2_total
bad_actual_t2_count  = t2_actual["bad"]  #* t2_count #* t2_total


observed_pred = [
    [good_pred_t1_count, bad_pred_t1_count],
    [good_pred_t2_count, bad_pred_t2_count]
]

observed_actual = [
    [good_actual_t1_count, bad_actual_t1_count],
    [good_actual_t2_count, bad_actual_t2_count]
]

chi2_pred, p_pred, dof_pred, expected_pred = chi2_contingency(observed_pred)
chi2_actual, p_actual, dof_actual, expected_actual = chi2_contingency(observed_actual)

print("=== Chi-square Test (Predictions) ===")
print(f"chi2 = {chi2_pred:.3f}, p = {p_pred}, dof = {dof_pred}")
print("Expected frequencies:", expected_pred)
print("------------------------------------\n")

print("=== Chi-square Test (Actuals) ===")
print(f"chi2 = {chi2_actual:.3f}, p = {p_actual}, dof = {dof_actual}")
print("Expected frequencies:", expected_actual)
print("------------------------------------\n")


alpha = 0.05
if p_pred <= alpha:
    print("Pred metric: There is a significant difference between T1 and T2.")
else:
    print("Pred metric: No significant difference found.")

if p_actual <= alpha:
    print("Actual metric: There is a significant difference between T1 and T2.")
else:
    print("Actual metric: No significant difference found.")
