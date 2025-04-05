import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from data_proc import df_relevant

target_col = "performance_metric"

X = df_relevant.drop(columns=[target_col]).values
y = df_relevant[target_col].values

q1, q2 = np.percentile(y, [33.33, 50])
def label_performance(val):
    if val <= q1:
        return 0
    elif val <= q2:
        return 0
    else:
        return 1

y_class = np.array([label_performance(val) for val in y])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["good", "bad"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
