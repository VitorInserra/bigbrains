import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from data_proc import df_relevant

target_col = "performance_metric"

# 1. Prepare features and target
X = df_relevant.drop(columns=[target_col]).values
y = df_relevant[target_col].values

# 2. Transform continuous target into categorical labels using quantiles
# Compute the 33.33rd and 66.66th percentiles for thresholding
q1, q2 = np.percentile(y, [33.33, 50])
def label_performance(val):
    if val <= q1:
        return 0  # good (lower is better)
    elif val <= q2:
        return 0  # average
    else:
        return 1  # bad

y_class = np.array([label_performance(val) for val in y])

# 3. Optionally scale features (Decision Trees don't require scaling, but it can be used for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

# 5. Build and train the Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# 6. Evaluate the classifier
y_pred = dt_clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["good", "bad"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
