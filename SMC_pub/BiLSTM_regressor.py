# ============================================================
# nested_bilstm_regression.py
#
# Purpose:
#   - Read the single feature table
#   - Build trial-level variable-length sequences
#   - Use the precomputed PERFORMANCE target from the table
#   - Remove 1% target outliers at the trial level
#   - Run nested grouped CV by session_id
#   - Train a basic TensorFlow BiLSTM for regression
#   - Save metrics, predictions, and plots to outputs/
#
# Important:
#   - The target column is "performance"
#   - LOWER performance values mean BETTER task performance
#   - This script does NOT recompute performance; it assumes
#     build_single_feature_table.py already created it correctly
#
# Input:
#   single_feature_table.csv
# ============================================================

import itertools
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # important for SLURM / headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================
INPUT_FEATURE_TABLE = "single_feature_table.csv"

SEQUENCE_ID_COLUMN = "trial_id"
GROUP_COLUMN = "session_id"
TIME_COLUMN = "timestep_idx"
TARGET_COLUMN = "performance"

# Nested CV
OUTER_N_SPLITS = 4
INNER_N_SPLITS = 3

# Training
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 45
SEED = 42
VERBOSE_FIT = 0

# Output management
OUTPUT_ROOT = Path("outputs")
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = OUTPUT_ROOT / f"run_{RUN_STAMP}"

# Outlier trimming
REMOVE_TARGET_OUTLIERS = True
TARGET_OUTLIER_TRIM_FRACTION = 0.01  # removes 1% total: 0.5% low + 0.5% high

# Hyperparameter grid
HYPERPARAM_GRID = {
    "lstm_units": [128, 256, 512],
    "dense_units": [16, 32, 64],
    "dropout": [0.2, 0.4],
    "learning_rate": [1e-4, 5e-5],
    "batch_size": [16, 64, 128],
}
# HYPERPARAM_GRID = {
#     "lstm_units": [32],
#     "dropout": [0.2],
#     "dense_units": [16],
#     "learning_rate": [1e-2],
#     "batch_size": [64],
# }

# Metadata columns to exclude from features
# Important: exclude all columns used to define the label
# to avoid leakage.
META_COLUMNS = {
    "trial_id",
    "session_id",
    "row_index",
    "timestep_idx",

    # labels / legacy labels
    "score",
    "performance",

    # trial metadata
    "test_version",
    "vr_id",
    "eeg_id",
    "eye_id",

    # performance-building metadata
    "elapsed_time_sec",
    "expected_rotation",
    "expected_rotation_norm",
    "obj_size",
    "obj_size_norm",
    "difficulty_denominator",

    # timestamps and window bookkeeping
    "vr_start_stamp",
    "vr_end_stamp",
    "eeg_start_stamp",
    "eeg_end_stamp",
    "window_start_sample",
    "window_end_sample",
    "window_num_samples",
    "window_start_sec_from_trial",
    "window_end_sec_from_trial",
    "fs_inferred_hz",
}


# =========================
# REPRODUCIBILITY
# =========================
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# OUTPUT HELPERS
# =========================
def prepare_output_dir() -> Path:
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_OUTPUT_DIR


def save_text_summary(text: str, output_dir: Path, filename: str = "run_summary.txt") -> None:
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_plot_sorted_predictions(y_true_all: np.ndarray, y_pred_all: np.ndarray, output_dir: Path) -> None:
    order = np.argsort(y_true_all)
    y_true_sorted = y_true_all[order]
    y_pred_sorted = y_pred_all[order]

    plt.figure(figsize=(8, 5))
    plt.plot(y_true_sorted, label="True")
    plt.plot(y_pred_sorted, label="Predicted")
    plt.xlabel("Sorted Trial Index")
    plt.ylabel("Performance (lower is better)")
    plt.title("Nested CV: Predicted vs True Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_true_sorted.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_plot_scatter(y_true_all: np.ndarray, y_pred_all: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_all, y_pred_all, alpha=0.7)

    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--")

    plt.xlabel("True Performance (lower is better)")
    plt.ylabel("Predicted Performance (lower is better)")
    plt.title("Nested CV: Predicted vs True Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_true_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# DATA HELPERS
# =========================
def load_feature_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {SEQUENCE_ID_COLUMN, GROUP_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(
        [GROUP_COLUMN, SEQUENCE_ID_COLUMN, TIME_COLUMN],
        kind="stable"
    ).reset_index(drop=True)

    return df


def build_trial_sequences(df: pd.DataFrame):
    """
    Returns:
        X_list         : list of arrays, each [timesteps, n_features]
        y              : regression target, one per trial
        groups         : session_id, one per trial
        trial_ids      : one per trial
        feature_cols   : list of feature column names
        global_max_len : max timesteps across all trials
    """
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding metadata columns.")

    X_list = []
    y_list = []
    groups_list = []
    trial_ids_list = []

    for trial_id, g in df.groupby(SEQUENCE_ID_COLUMN, sort=False):
        g = g.sort_values(TIME_COLUMN, kind="stable")

        x = (
            g[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        y = float(g[TARGET_COLUMN].iloc[0])
        group = g[GROUP_COLUMN].iloc[0]

        if not np.isfinite(y):
            raise ValueError(f"Non-finite target found for trial_id={trial_id}: {y}")

        X_list.append(x)
        y_list.append(y)
        groups_list.append(group)
        trial_ids_list.append(trial_id)

    y = np.asarray(y_list, dtype=np.float32)
    groups = np.asarray(groups_list)
    trial_ids = np.asarray(trial_ids_list)
    global_max_len = max(seq.shape[0] for seq in X_list)

    return X_list, y, groups, trial_ids, feature_cols, global_max_len


def trim_trial_level_target_outliers(
    X_list,
    y: np.ndarray,
    groups: np.ndarray,
    trial_ids: np.ndarray,
    trim_fraction: float = 0.01,
):
    """
    Symmetric trimming on the trial-level target.
    Example: trim_fraction=0.01 removes bottom 0.5% and top 0.5%.
    """
    if trim_fraction <= 0.0:
        return X_list, y, groups, trial_ids, {
            "n_removed": 0,
            "n_remaining": len(y),
            "lower_bound": None,
            "upper_bound": None,
        }

    if trim_fraction >= 1.0:
        raise ValueError("trim_fraction must be < 1.0")

    lower_q = trim_fraction / 2.0
    upper_q = 1.0 - (trim_fraction / 2.0)

    lower_bound = float(np.quantile(y, lower_q))
    upper_bound = float(np.quantile(y, upper_q))

    keep_mask = (y >= lower_bound) & (y <= upper_bound)

    X_list_trimmed = [x for x, keep in zip(X_list, keep_mask) if keep]
    y_trimmed = y[keep_mask]
    groups_trimmed = groups[keep_mask]
    trial_ids_trimmed = trial_ids[keep_mask]

    info = {
        "n_removed": int((~keep_mask).sum()),
        "n_remaining": int(keep_mask.sum()),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }

    return X_list_trimmed, y_trimmed, groups_trimmed, trial_ids_trimmed, info


def subset_list_by_indices(lst, indices):
    return [lst[i] for i in indices]


def fit_feature_scaler(seq_list: List[np.ndarray]) -> StandardScaler:
    """
    Fit on stacked training timesteps only.
    """
    stacked = np.vstack(seq_list)
    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


def transform_sequence_list(seq_list: List[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    return [scaler.transform(seq).astype(np.float32) for seq in seq_list]


def pad_sequence_list(seq_list: List[np.ndarray], max_len: int, n_features: int) -> np.ndarray:
    X = np.zeros((len(seq_list), max_len, n_features), dtype=np.float32)

    for i, seq in enumerate(seq_list):
        seq_len = min(seq.shape[0], max_len)
        X[i, :seq_len, :] = seq[:seq_len, :]

    return X


# =========================
# MODEL
# =========================
def build_bilstm_regressor(max_len: int, n_features: int, hp: dict) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(max_len, n_features), name="sequence_input")

    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hp["lstm_units"], return_sequences=False)
    )(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    x = tf.keras.layers.Dense(hp["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def hp_grid_to_list(grid: dict):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# =========================
# TRAIN / EVAL HELPERS
# =========================
def train_one_inner_fold(
    X_train_seq: List[np.ndarray],
    y_train: np.ndarray,
    X_val_seq: List[np.ndarray],
    y_val: np.ndarray,
    hp: dict,
    max_len: int,
    n_features: int,
):
    scaler = fit_feature_scaler(X_train_seq)

    X_train_scaled = transform_sequence_list(X_train_seq, scaler)
    X_val_scaled = transform_sequence_list(X_val_seq, scaler)

    X_train_pad = pad_sequence_list(X_train_scaled, max_len=max_len, n_features=n_features)
    X_val_pad = pad_sequence_list(X_val_scaled, max_len=max_len, n_features=n_features)

    model = build_bilstm_regressor(max_len=max_len, n_features=n_features, hp=hp)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=MAX_EPOCHS,
        batch_size=hp["batch_size"],
        verbose=VERBOSE_FIT,
        callbacks=callbacks,
    )

    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses) + 1)
    best_val_loss = float(np.min(val_losses))

    return best_val_loss, best_epoch


def train_final_outer_model(
    X_train_seq: List[np.ndarray],
    y_train: np.ndarray,
    X_test_seq: List[np.ndarray],
    y_test: np.ndarray,
    hp: dict,
    final_epochs: int,
    max_len: int,
    n_features: int,
):
    scaler = fit_feature_scaler(X_train_seq)

    X_train_scaled = transform_sequence_list(X_train_seq, scaler)
    X_test_scaled = transform_sequence_list(X_test_seq, scaler)

    X_train_pad = pad_sequence_list(X_train_scaled, max_len=max_len, n_features=n_features)
    X_test_pad = pad_sequence_list(X_test_scaled, max_len=max_len, n_features=n_features)

    model = build_bilstm_regressor(max_len=max_len, n_features=n_features, hp=hp)

    model.fit(
        X_train_pad,
        y_train,
        epochs=final_epochs,
        batch_size=hp["batch_size"],
        verbose=VERBOSE_FIT,
    )

    preds = model.predict(X_test_pad, verbose=0).reshape(-1)

    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)

    if len(np.unique(y_test)) > 1:
        r2 = r2_score(y_test, preds)
    else:
        r2 = np.nan

    return {
        "test_mse": float(mse),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_r2": float(r2) if not np.isnan(r2) else np.nan,
        "preds": preds,
    }


# =========================
# NESTED CV
# =========================
def run_nested_cv():
    output_dir = prepare_output_dir()
    set_all_seeds(SEED)

    df = load_feature_table(INPUT_FEATURE_TABLE)
    X_list, y, groups, trial_ids, feature_cols, global_max_len = build_trial_sequences(df)

    if REMOVE_TARGET_OUTLIERS:
        X_list, y, groups, trial_ids, trim_info = trim_trial_level_target_outliers(
            X_list=X_list,
            y=y,
            groups=groups,
            trial_ids=trial_ids,
            trim_fraction=TARGET_OUTLIER_TRIM_FRACTION,
        )
    else:
        trim_info = {
            "n_removed": 0,
            "n_remaining": len(y),
            "lower_bound": None,
            "upper_bound": None,
        }

    n_trials = len(X_list)
    n_features = len(feature_cols)

    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(str(msg))

    log(f"\nOutput directory: {output_dir.resolve()}")
    log(f"Trials after trimming: {n_trials}")
    log(f"Sessions after trimming: {len(np.unique(groups))}")
    log(f"Feature dimension: {n_features}")
    log(f"Global max sequence length: {global_max_len}")
    log(f"Target column: {TARGET_COLUMN}")
    log("Interpretation: LOWER target values = BETTER task performance")
    log(f"Removed target outliers: {trim_info['n_removed']}")
    log(f"Remaining trials: {trim_info['n_remaining']}")
    if trim_info["lower_bound"] is not None:
        log(f"Target keep range: [{trim_info['lower_bound']:.6f}, {trim_info['upper_bound']:.6f}]")
    log(f"Target mean: {y.mean():.6f}")
    log(f"Target std : {y.std():.6f}")
    log(f"Target min : {y.min():.6f}")
    log(f"Target max : {y.max():.6f}")

    outer_cv = GroupKFold(n_splits=OUTER_N_SPLITS)
    hp_configs = hp_grid_to_list(HYPERPARAM_GRID)

    outer_results = []
    all_y_true = []
    all_y_pred = []
    all_prediction_rows = []

    dummy_X = np.zeros((n_trials, 1), dtype=np.float32)

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(dummy_X, y, groups), start=1
    ):
        log("\n" + "=" * 70)
        log(f"OUTER FOLD {outer_fold}/{OUTER_N_SPLITS}")

        X_outer_train = subset_list_by_indices(X_list, outer_train_idx)
        y_outer_train = y[outer_train_idx]
        groups_outer_train = groups[outer_train_idx]
        trial_ids_outer_train = trial_ids[outer_train_idx]

        X_outer_test = subset_list_by_indices(X_list, outer_test_idx)
        y_outer_test = y[outer_test_idx]
        groups_outer_test = groups[outer_test_idx]
        trial_ids_outer_test = trial_ids[outer_test_idx]

        log(f"Outer train sessions: {np.unique(groups_outer_train)}")
        log(f"Outer test sessions : {np.unique(groups_outer_test)}")

        inner_cv = GroupKFold(n_splits=INNER_N_SPLITS)
        inner_dummy_X = np.zeros((len(X_outer_train), 1), dtype=np.float32)

        hp_search_rows = []

        for hp in hp_configs:
            inner_val_losses = []
            inner_best_epochs = []

            for inner_train_rel_idx, inner_val_rel_idx in inner_cv.split(
                inner_dummy_X, y_outer_train, groups_outer_train
            ):
                X_inner_train = subset_list_by_indices(X_outer_train, inner_train_rel_idx)
                y_inner_train = y_outer_train[inner_train_rel_idx]

                X_inner_val = subset_list_by_indices(X_outer_train, inner_val_rel_idx)
                y_inner_val = y_outer_train[inner_val_rel_idx]

                set_all_seeds(SEED)

                best_val_loss, best_epoch = train_one_inner_fold(
                    X_train_seq=X_inner_train,
                    y_train=y_inner_train,
                    X_val_seq=X_inner_val,
                    y_val=y_inner_val,
                    hp=hp,
                    max_len=global_max_len,
                    n_features=n_features,
                )

                inner_val_losses.append(best_val_loss)
                inner_best_epochs.append(best_epoch)

            hp_search_rows.append({
                **hp,
                "mean_inner_val_loss": float(np.mean(inner_val_losses)),
                "median_best_epoch": int(np.median(inner_best_epochs)),
            })

        hp_search_df = pd.DataFrame(hp_search_rows).sort_values(
            "mean_inner_val_loss", ascending=True, kind="stable"
        ).reset_index(drop=True)

        hp_search_df.to_csv(output_dir / f"inner_cv_results_outer_fold_{outer_fold}.csv", index=False)

        best_hp_row = hp_search_df.iloc[0].to_dict()
        best_hp = {
            "lstm_units": int(best_hp_row["lstm_units"]),
            "dense_units": int(best_hp_row["dense_units"]),
            "dropout": float(best_hp_row["dropout"]),
            "learning_rate": float(best_hp_row["learning_rate"]),
            "batch_size": int(best_hp_row["batch_size"]),
        }
        final_epochs = int(best_hp_row["median_best_epoch"])

        log("\nBest hyperparameters from inner CV:")
        log(str(best_hp))
        log(f"Mean inner validation loss: {best_hp_row['mean_inner_val_loss']:.6f}")
        log(f"Final epochs: {final_epochs}")

        set_all_seeds(SEED)

        fold_test_results = train_final_outer_model(
            X_train_seq=X_outer_train,
            y_train=y_outer_train,
            X_test_seq=X_outer_test,
            y_test=y_outer_test,
            hp=best_hp,
            final_epochs=final_epochs,
            max_len=global_max_len,
            n_features=n_features,
        )

        fold_preds = fold_test_results.pop("preds")

        all_y_true.append(y_outer_test.copy())
        all_y_pred.append(fold_preds.copy())

        for i in range(len(y_outer_test)):
            all_prediction_rows.append({
                "outer_fold": outer_fold,
                "trial_id": trial_ids_outer_test[i],
                "session_id": groups_outer_test[i],
                "y_true": float(y_outer_test[i]),
                "y_pred": float(fold_preds[i]),
                "abs_error": float(abs(y_outer_test[i] - fold_preds[i])),
            })

        fold_result = {
            "outer_fold": outer_fold,
            "n_outer_train_trials": len(outer_train_idx),
            "n_outer_test_trials": len(outer_test_idx),
            "n_outer_train_sessions": len(np.unique(groups_outer_train)),
            "n_outer_test_sessions": len(np.unique(groups_outer_test)),
            "best_lstm_units": best_hp["lstm_units"],
            "best_dense_units": best_hp["dense_units"],
            "best_dropout": best_hp["dropout"],
            "best_learning_rate": best_hp["learning_rate"],
            "best_batch_size": best_hp["batch_size"],
            "final_epochs": final_epochs,
            **fold_test_results,
        }

        outer_results.append(fold_result)

        log("\nOuter fold test results:")
        for k, v in fold_test_results.items():
            if isinstance(v, float) and not np.isnan(v):
                log(f"  {k}: {v:.6f}")
            else:
                log(f"  {k}: {v}")

    results_df = pd.DataFrame(outer_results)
    results_df.to_csv(output_dir / "outer_fold_results.csv", index=False)

    log("\n" + "=" * 70)
    log("FINAL NESTED CV RESULTS")
    log(results_df.to_string(index=False))

    numeric_results = results_df.select_dtypes(include=[np.number])

    summary_df = pd.DataFrame({
        "metric": numeric_results.columns,
        "mean": numeric_results.mean(numeric_only=True).values,
        "std": numeric_results.std(numeric_only=True).values,
        "min": numeric_results.min(numeric_only=True).values,
        "max": numeric_results.max(numeric_only=True).values,
    })
    summary_df.to_csv(output_dir / "outer_fold_summary_stats.csv", index=False)

    log("\nMean across outer folds:")
    log(numeric_results.mean(numeric_only=True).to_string())

    y_true_all = np.concatenate(all_y_true).reshape(-1)
    y_pred_all = np.concatenate(all_y_pred).reshape(-1)

    overall_mse = mean_squared_error(y_true_all, y_pred_all)
    overall_rmse = math.sqrt(overall_mse)
    overall_mae = mean_absolute_error(y_true_all, y_pred_all)
    overall_r2 = r2_score(y_true_all, y_pred_all) if len(np.unique(y_true_all)) > 1 else np.nan

    overall_metrics_df = pd.DataFrame([{
        "overall_mse": float(overall_mse),
        "overall_rmse": float(overall_rmse),
        "overall_mae": float(overall_mae),
        "overall_r2": float(overall_r2) if not np.isnan(overall_r2) else np.nan,
        "n_trials_total": int(len(y_true_all)),
    }])
    overall_metrics_df.to_csv(output_dir / "overall_prediction_metrics.csv", index=False)

    predictions_df = pd.DataFrame(all_prediction_rows)
    predictions_df.to_csv(output_dir / "all_outer_fold_predictions.csv", index=False)

    save_plot_sorted_predictions(y_true_all, y_pred_all, output_dir)
    save_plot_scatter(y_true_all, y_pred_all, output_dir)

    log("\nSaved files:")
    log(f"  - {output_dir / 'outer_fold_results.csv'}")
    log(f"  - {output_dir / 'outer_fold_summary_stats.csv'}")
    log(f"  - {output_dir / 'overall_prediction_metrics.csv'}")
    log(f"  - {output_dir / 'all_outer_fold_predictions.csv'}")
    log(f"  - {output_dir / 'predicted_vs_true_sorted.png'}")
    log(f"  - {output_dir / 'predicted_vs_true_scatter.png'}")
    for outer_fold in range(1, OUTER_N_SPLITS + 1):
        log(f"  - {output_dir / f'inner_cv_results_outer_fold_{outer_fold}.csv'}")

    save_text_summary("\n".join(log_lines), output_dir, filename="run_summary.txt")

    return results_df


if __name__ == "__main__":
    run_nested_cv()