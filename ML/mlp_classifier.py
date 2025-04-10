import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from data_proc import df_relevant

target_col = "performance_metric"

X = df_relevant.drop(columns=[target_col]).values
y = df_relevant[target_col].values

q1, q2 = np.percentile(y, [20, 50])
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


def mlp_classifier():
    model = keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(2, activation="softmax"), 
    ])

    opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # callback = keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=20,    # adjust as needed
    #     restore_best_weights=True
    # )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=32,
        # callbacks=[callback],
        verbose=1
    )
    return model

model = mlp_classifier()

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
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
