import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppose your DataFrame is called df and it has a column named "performance_metric"
df = pd.read_csv("feature_table.csv")
performance = df["performance_metric"].dropna()

# 1. Plot the distribution
plt.figure(figsize=(8, 6))
sns.histplot(performance, kde=True, bins=30, color="skyblue", alpha=0.7)
plt.title("Performance Distribution")
plt.xlabel("Performance Metric")
plt.ylabel("Count")

# 2. Compute quantile-based thresholds
q1, q2 = performance.quantile([0.33, 0.66])
print(f"33rd percentile (Q1) = {q1:.4f}")
print(f"66th percentile (Q2) = {q2:.4f}")

# 3. Plot vertical lines for thresholds
plt.axvline(q1, color="red", linestyle="--", label=f"33rd percentile = {q1:.4f}")
plt.axvline(q2, color="green", linestyle="--", label=f"66th percentile = {q2:.4f}")
plt.legend()
plt.show()

# 4. Assign labels based on these thresholds
df["performance_label"] = pd.cut(
    df["performance_metric"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["bad", "average", "good"],
)

# Optional: Check the distribution of labels
label_counts = df["performance_label"].value_counts()
print("\nLabel Distribution:")
print(label_counts)
