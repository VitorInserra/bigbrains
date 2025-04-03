import pandas as pd
import numpy as np

# 1. scikit-learn imports
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV

# Suppose you’ve already loaded your data into a DataFrame `df`,
# with your target in the column 'performance_metric'.

df = pd.read_csv("feature_table.csv")  # or any DataFrame your data is in

# 2. Choose the target column
target_col = "performance_metric"

# 3. Identify and drop any non-feature columns
exclude_cols = [
    "session_id",
    "round_id",
    "start_time",
    "end_time",
    "score",
    "test_version",
    target_col,
]
exclude_cols = list(set(exclude_cols))


relevant_sensors = [
    "fz", "f3", "f4", "f7", "f8",     # some frontal leads 
    "fc3", "fc4", "af3", "af4",       # potential mid-frontal leads
    "p3", "p4", "p7", "p8", "pz",     # parietal
    "o1", "o2", "oz",                 # occipital
    # plus any others you consider relevant
]
relevant_bands = ["theta", "alpha", "beta", "delta:"]


def is_relevant_column(col):
    lc = col.lower()
    sensor_match = any(sensor in lc for sensor in relevant_sensors)
    band_match = any(band in lc for band in relevant_bands)
    return sensor_match and band_match

# Build the final list of columns to keep
keep_cols = [col for col in df.columns if is_relevant_column(col)]

# Also keep your target column, e.g. "performance_metric"
keep_cols.append("performance_metric")

# Now, create a new DataFrame with only relevant columns
df_relevant = df[keep_cols]

print("Kept columns:", df_relevant.columns)


def svr(df, target_col, exclude_cols):
    print("##### SVR #####")
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df[target_col].values
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))]
    )

    # 5. Define a cross-validation strategy
    #    For example, 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 6. Evaluate your model using cross_val_score
    #    'scoring' can be "neg_mean_squared_error", "r2", or another regression metric
    scores_mse = cross_val_score(
        pipeline, X, y, cv=kfold, scoring="neg_mean_squared_error"
    )
    scores_r2 = cross_val_score(pipeline, X, y, cv=kfold, scoring="r2")

    # 7. Print average results (over the folds)
    print("Cross-validated Mean Squared Error: ", -np.mean(scores_mse))
    print("Cross-validated R^2: ", np.mean(scores_r2))


def random_forest(df, target_col, exclude_cols):
    print("##### Random Forest #####")
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df[target_col].values

    # Build pipeline: (scaler) -> (RandomForestRegressor)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )

    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="neg_mean_squared_error")

    mse = -np.mean(scores)
    print("Cross-validated MSE:", mse)


def mlp(df, target_col, exclude_cols):
    print("##### MLP #####")
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df[target_col].values

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),  # single hidden layer of 100 neurons
                    activation="relu",  # activation function for hidden layer
                    solver="adam",  # weight optimization method
                    alpha=0.0001,  # L2 penalty (regularization term)
                    max_iter=2000,  # maximum number of iterations
                    random_state=42,  # reproducibility
                ),
            ),
        ]
    )
    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64, 32)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__max_iter": [2000, 5000]
    }
    grid_search = GridSearchCV(
        pipeline, param_grid=param_grid,
        cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, y)

    print("Best params:", grid_search.best_params_)
    print("Best score:", -grid_search.best_score_)  # since MSE is negative



    # We’ll do 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate with negative MSE (we take negative to revert sign for scikit scoring)
    scores_mse = cross_val_score(pipeline, X, y, cv=kfold, scoring="neg_mean_squared_error")
    # Evaluate R^2 as well
    scores_r2 = cross_val_score(pipeline, X, y, cv=kfold, scoring="r2")

    mse = -np.mean(scores_mse)
    r2 = np.mean(scores_r2)

    print("Cross-validated Mean Squared Error (MLP):", mse)
    print("Cross-validated R^2 (MLP):", r2)


# svr(df_relevant, target_col, exclude_cols)
random_forest(df_relevant, target_col, exclude_cols)
mlp(df, target_col, exclude_cols)
