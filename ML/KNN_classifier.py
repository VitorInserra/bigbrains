import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
    X_scaled, y_class, test_size=0.15, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["good", "bad"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


import matplotlib.pyplot as plt
import numpy as np


class_names = ["good", "bad"]  # Use whichever names match your 0/1 labels

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)


thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(
        j, i, str(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black"
    )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()
