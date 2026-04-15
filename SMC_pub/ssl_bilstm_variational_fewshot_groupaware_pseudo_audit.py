# ============================================================
# ssl_bilstm_variational_fewshot.py
#
# Purpose:
#   - Read a single feature table
#   - Build one variable-length sequence per trial
#   - Run grouped outer CV by session_id
#   - Pretrain a BiLSTM encoder with a self-supervised denoising
#     sequence-reconstruction task on all outer-train sequences
#   - Select exactly 2 training groups as the labeled support groups
#   - Sample a few labeled trials from those support groups
#   - Fine-tune a Bayesian classifier with a variational softmax head
#     (TensorFlow Probability DenseFlipout)
#   - Return MC predictive probabilities and uncertainty estimates
#   - Run uncertainty-filtered pseudo-labeling on the unlabeled
#     remainder of each outer-train fold
#
# Notes:
#   - This script converts the old regression pipeline into
#     grouped semi-supervised classification.
#   - If CLASS_LABEL_COLUMN is not available, the script binarizes
#     the continuous score using the OUTER-TRAIN median only.
#   - Pseudo-labeling operates only on the unlabeled remainder of the
#     OUTER-TRAIN fold. The OUTER-TEST fold stays untouched.
# ============================================================

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


tfd = tfp.distributions


# =========================
# CONFIG
# =========================
INPUT_FEATURE_TABLE = "single_feature_table.csv"

SEQUENCE_ID_COLUMN = "trial_id"
GROUP_COLUMN = "session_id"
TIME_COLUMN = "timestep_idx"
LABEL_COLUMN = "score"               # used if CLASS_LABEL_COLUMN is None
CLASS_LABEL_COLUMN = None             # e.g. "score_class" if you already have it

# Label construction fallback when CLASS_LABEL_COLUMN is None
FALLBACK_CLASS_MODE = "median_binary"   # {"median_binary", "tertile_3class"}

# Grouped evaluation
OUTER_N_SPLITS = 4
LABELED_GROUPS_PER_FOLD = 2

# Few-shot configuration
FEW_SHOT_TRAIN_PER_CLASS = 4
FEW_SHOT_VAL_PER_CLASS = 2
MIN_CLASSES_REQUIRED = 2

# SSL pretraining
SSL_EPOCHS = 60
SSL_BATCH_SIZE = 64
SSL_LEARNING_RATE = 1e-3
SSL_MASK_PROB = 0.15
SSL_NOISE_STD = 0.05
SSL_EARLY_STOPPING_PATIENCE = 12
SSL_VAL_GROUPS_PER_FOLD = 2

# Fine-tuning
FINE_TUNE_EPOCHS = 80
FINE_TUNE_BATCH_SIZE = 16
FINE_TUNE_LEARNING_RATE = 3e-4
FINE_TUNE_EARLY_STOPPING_PATIENCE = 15
FREEZE_ENCODER_DURING_FINETUNE = False

# MC uncertainty
MC_SAMPLES = 30

# Pseudo-labeling
PSEUDO_LABELING_ENABLED = True
PSEUDO_MAX_ROUNDS = 5
PSEUDO_REFIT_EPOCHS = 20
PSEUDO_MIN_CONFIDENCE = 0.80
PSEUDO_MAX_PREDICTIVE_ENTROPY = 0.45
PSEUDO_MAX_MUTUAL_INFORMATION = 0.10
PSEUDO_MAX_VARIATION_RATIO = 0.20
PSEUDO_MAX_NEW_PER_ROUND = 24
PSEUDO_MAX_NEW_PER_CLASS = 12
PSEUDO_LABEL_SAMPLE_WEIGHT = 0.50

# Model sizes
HP = {
    "ssl_lstm_units": 256,
    "repr_dim": 128,
    "classifier_dense_units": 64,
    "dropout": 0.30,
}

SEED = 42
VERBOSE_FIT = 0

# Metadata columns to exclude from features
META_COLUMNS = {
    "trial_id",
    "session_id",
    "row_index",
    "timestep_idx",
    "score",
    "score_class",
    "test_version",
    "vr_id",
    "eeg_id",
    "eye_id",
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
# DATA CONTAINERS
# =========================
@dataclass
class TrialDataset:
    X_list: List[np.ndarray]
    raw_targets: np.ndarray
    groups: np.ndarray
    trial_ids: np.ndarray
    feature_cols: List[str]
    global_max_len: int
    direct_class_labels: Optional[np.ndarray]


# =========================
# DATA HELPERS
# =========================
def load_feature_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {SEQUENCE_ID_COLUMN, GROUP_COLUMN, TIME_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(
        [GROUP_COLUMN, SEQUENCE_ID_COLUMN, TIME_COLUMN],
        kind="stable",
    ).reset_index(drop=True)

    return df


def build_trial_sequences(df: pd.DataFrame) -> TrialDataset:
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding metadata columns.")

    X_list = []
    raw_targets = []
    groups_list = []
    trial_ids_list = []
    direct_class_labels = [] if CLASS_LABEL_COLUMN and CLASS_LABEL_COLUMN in df.columns else None

    for trial_id, g in df.groupby(SEQUENCE_ID_COLUMN, sort=False):
        g = g.sort_values(TIME_COLUMN, kind="stable")

        x = (
            g[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        raw_target = float(g[LABEL_COLUMN].iloc[0])
        group = g[GROUP_COLUMN].iloc[0]

        X_list.append(x)
        raw_targets.append(raw_target)
        groups_list.append(group)
        trial_ids_list.append(trial_id)

        if direct_class_labels is not None:
            direct_class_labels.append(int(g[CLASS_LABEL_COLUMN].iloc[0]))

    raw_targets = np.asarray(raw_targets, dtype=np.float32)
    groups = np.asarray(groups_list)
    trial_ids = np.asarray(trial_ids_list)
    global_max_len = max(seq.shape[0] for seq in X_list)

    return TrialDataset(
        X_list=X_list,
        raw_targets=raw_targets,
        groups=groups,
        trial_ids=trial_ids,
        feature_cols=feature_cols,
        global_max_len=global_max_len,
        direct_class_labels=(np.asarray(direct_class_labels, dtype=np.int32) if direct_class_labels is not None else None),
    )


def subset_list_by_indices(lst: Sequence[np.ndarray], indices: Sequence[int]) -> List[np.ndarray]:
    return [lst[i] for i in indices]


def fit_feature_scaler(seq_list: List[np.ndarray]) -> StandardScaler:
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


def build_valid_timestep_mask(X_pad: np.ndarray) -> np.ndarray:
    return (np.abs(X_pad).sum(axis=-1, keepdims=True) > 0).astype(np.float32)


# =========================
# LABEL HELPERS
# =========================
def make_class_labels(
    train_raw_targets: np.ndarray,
    target_raw_targets: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if mode == "median_binary":
        threshold = float(np.median(train_raw_targets))
        y_cls = (target_raw_targets >= threshold).astype(np.int32)
        return y_cls, {"threshold": threshold}

    if mode == "tertile_3class":
        q1, q2 = np.quantile(train_raw_targets, [1.0 / 3.0, 2.0 / 3.0])
        bins = np.array([-np.inf, q1, q2, np.inf], dtype=np.float32)
        y_cls = np.digitize(target_raw_targets, bins[1:-1], right=False).astype(np.int32)
        return y_cls, {"q1": float(q1), "q2": float(q2)}

    raise ValueError(f"Unsupported FALLBACK_CLASS_MODE: {mode}")


def get_outer_fold_class_labels(
    direct_class_labels: Optional[np.ndarray],
    raw_targets: np.ndarray,
    outer_train_idx: np.ndarray,
    target_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if direct_class_labels is not None:
        return direct_class_labels[target_idx].astype(np.int32), {}

    y_cls, info = make_class_labels(
        train_raw_targets=raw_targets[outer_train_idx],
        target_raw_targets=raw_targets[target_idx],
        mode=FALLBACK_CLASS_MODE,
    )
    return y_cls, info


def num_classes_from_labels(y: np.ndarray) -> int:
    return int(np.max(y) + 1)


def choose_group_validation_split(
    groups: np.ndarray,
    n_val_groups: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 groups to create a group-aware validation split.")

    n_val = min(max(1, n_val_groups), len(unique_groups) - 1)
    shuffled_groups = unique_groups.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled_groups)

    val_groups = np.asarray(sorted(shuffled_groups[:n_val]))
    train_groups = np.asarray(sorted(shuffled_groups[n_val:]))

    train_idx = np.where(np.isin(groups, train_groups))[0]
    val_idx = np.where(np.isin(groups, val_groups))[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Group-aware validation split produced an empty train or val partition.")

    return train_idx.astype(int), val_idx.astype(int), train_groups, val_groups


# =========================
# GROUP / FEW-SHOT SELECTION
# =========================
def choose_labeled_groups(
    groups_outer_train: np.ndarray,
    y_outer_train_cls: np.ndarray,
    n_pick: int = 2,
) -> np.ndarray:
    unique_groups = np.unique(groups_outer_train)
    if len(unique_groups) < n_pick:
        raise ValueError("Not enough groups available to select labeled support groups.")

    n_classes = len(np.unique(y_outer_train_cls))
    best_tuple = None
    best_key = None

    for combo in itertools.combinations(unique_groups, n_pick):
        mask = np.isin(groups_outer_train, combo)
        y_combo = y_outer_train_cls[mask]
        classes_present = len(np.unique(y_combo))
        num_examples = int(mask.sum())
        class_balance = tuple(np.bincount(y_combo, minlength=n_classes).tolist())
        key = (classes_present, num_examples, min(class_balance), -max(class_balance))
        if (best_key is None) or (key > best_key):
            best_key = key
            best_tuple = combo

    if best_tuple is None:
        raise RuntimeError("Failed to select support groups.")

    return np.asarray(best_tuple)


def sample_few_shot_indices(
    candidate_indices: np.ndarray,
    y_candidate: np.ndarray,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    classes = np.unique(y_candidate)
    if len(classes) < MIN_CLASSES_REQUIRED:
        raise ValueError("Support pool does not contain enough classes for supervised fine-tuning.")

    for cls in classes:
        cls_local = np.where(y_candidate == cls)[0]
        rng.shuffle(cls_local)

        n_train = min(train_per_class, len(cls_local))
        n_val = min(val_per_class, max(0, len(cls_local) - n_train))

        chosen_train_local = cls_local[:n_train]
        chosen_val_local = cls_local[n_train:n_train + n_val]

        train_idx.extend(candidate_indices[chosen_train_local].tolist())
        val_idx.extend(candidate_indices[chosen_val_local].tolist())

    return np.asarray(sorted(train_idx), dtype=int), np.asarray(sorted(val_idx), dtype=int)


# =========================
# SSL AUGMENTATION
# =========================
def corrupt_sequences_for_ssl(
    X: np.ndarray,
    mask_prob: float,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_corrupt = X.copy()
    valid_mask = build_valid_timestep_mask(X).astype(bool)

    feature_mask = rng.random(X.shape, dtype=np.float32) < mask_prob
    feature_mask = feature_mask & valid_mask
    X_corrupt[feature_mask] = 0.0

    gaussian_noise = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
    X_corrupt = X_corrupt + gaussian_noise * valid_mask.astype(np.float32)

    return X_corrupt.astype(np.float32)


# =========================
# MODELS
# =========================
def build_ssl_encoder(max_len: int, n_features: int, hp: dict) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(max_len, n_features), name="sequence_input")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hp["ssl_lstm_units"], return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hp["ssl_lstm_units"] // 2, return_sequences=False)
    )(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    embedding = tf.keras.layers.Dense(hp["repr_dim"], activation="relu", name="embedding")(x)
    return tf.keras.Model(inputs=inputs, outputs=embedding, name="ssl_bilstm_encoder")


def build_ssl_reconstruction_model(encoder: tf.keras.Model, max_len: int, n_features: int, hp: dict) -> tf.keras.Model:
    inputs = encoder.input
    z = encoder(inputs)
    x = tf.keras.layers.RepeatVector(max_len)(z)
    x = tf.keras.layers.LSTM(hp["ssl_lstm_units"] // 2, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    x = tf.keras.layers.LSTM(hp["ssl_lstm_units"], return_sequences=True)(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features, activation="linear"),
        name="reconstruction_head",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ssl_denoising_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=SSL_LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def make_scaled_kl_divergence_fn(num_examples: int):
    denom = tf.cast(max(1, int(num_examples)), tf.float32)

    def scaled_kl(q, p, _):
        return tfd.kl_divergence(q, p) / denom

    return scaled_kl


def build_variational_classifier(
    encoder: tf.keras.Model,
    max_len: int,
    n_features: int,
    n_classes: int,
    hp: dict,
    num_labeled_examples: int,
) -> tf.keras.Model:
    kl_fn = make_scaled_kl_divergence_fn(num_labeled_examples)

    inputs = tf.keras.Input(shape=(max_len, n_features), name="classifier_input")
    x = encoder(inputs)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    x = tf.keras.layers.Dense(hp["classifier_dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    logits = tfp.layers.DenseFlipout(
        n_classes,
        activation=None,
        kernel_divergence_fn=kl_fn,
        bias_divergence_fn=kl_fn,
        name="bayesian_logits",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="variational_softmax_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


# =========================
# TRAINING
# =========================
def pretrain_ssl_encoder(
    X_train_pad: np.ndarray,
    X_val_pad: np.ndarray,
    max_len: int,
    n_features: int,
    hp: dict,
    seed: int,
) -> tf.keras.Model:
    encoder = build_ssl_encoder(max_len=max_len, n_features=n_features, hp=hp)
    ssl_model = build_ssl_reconstruction_model(encoder=encoder, max_len=max_len, n_features=n_features, hp=hp)

    X_ssl_train_in = corrupt_sequences_for_ssl(
        X=X_train_pad,
        mask_prob=SSL_MASK_PROB,
        noise_std=SSL_NOISE_STD,
        seed=seed,
    )
    X_ssl_val_in = corrupt_sequences_for_ssl(
        X=X_val_pad,
        mask_prob=SSL_MASK_PROB,
        noise_std=SSL_NOISE_STD,
        seed=seed + 1,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=SSL_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    ssl_model.fit(
        X_ssl_train_in,
        X_train_pad,
        validation_data=(X_ssl_val_in, X_val_pad),
        epochs=SSL_EPOCHS,
        batch_size=min(SSL_BATCH_SIZE, max(1, len(X_train_pad))),
        verbose=VERBOSE_FIT,
        callbacks=callbacks,
        shuffle=True,
    )

    return encoder


def fine_tune_variational_classifier(
    encoder: tf.keras.Model,
    X_labeled_train: np.ndarray,
    y_labeled_train: np.ndarray,
    X_labeled_val: Optional[np.ndarray],
    y_labeled_val: Optional[np.ndarray],
    n_classes: int,
    max_len: int,
    n_features: int,
    hp: dict,
    num_training_examples_for_kl: int,
    epochs: int,
    sample_weight: Optional[np.ndarray] = None,
) -> tf.keras.Model:
    encoder.trainable = not FREEZE_ENCODER_DURING_FINETUNE

    classifier = build_variational_classifier(
        encoder=encoder,
        max_len=max_len,
        n_features=n_features,
        n_classes=n_classes,
        hp=hp,
        num_labeled_examples=num_training_examples_for_kl,
    )

    callbacks = []
    fit_kwargs = {}
    if X_labeled_val is not None and len(X_labeled_val) > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=FINE_TUNE_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
        )
        fit_kwargs["validation_data"] = (X_labeled_val, y_labeled_val)

    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    classifier.fit(
        X_labeled_train,
        y_labeled_train,
        epochs=epochs,
        batch_size=min(FINE_TUNE_BATCH_SIZE, max(1, len(X_labeled_train))),
        verbose=VERBOSE_FIT,
        callbacks=callbacks,
        shuffle=True,
        **fit_kwargs,
    )
    return classifier


def continue_fine_tuning_classifier(
    classifier: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    epochs: int,
    sample_weight: Optional[np.ndarray] = None,
) -> tf.keras.Model:
    callbacks = []
    fit_kwargs = {}
    if X_val is not None and len(X_val) > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=max(5, min(FINE_TUNE_EARLY_STOPPING_PATIENCE, epochs // 2 + 1)),
                restore_best_weights=True,
                verbose=0,
            )
        )
        fit_kwargs["validation_data"] = (X_val, y_val)

    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    classifier.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=min(FINE_TUNE_BATCH_SIZE, max(1, len(X_train))),
        verbose=VERBOSE_FIT,
        callbacks=callbacks,
        shuffle=True,
        **fit_kwargs,
    )
    return classifier


# =========================
# INFERENCE / UNCERTAINTY
# =========================
def monte_carlo_predict(
    model: tf.keras.Model,
    X: np.ndarray,
    mc_samples: int,
) -> Dict[str, np.ndarray]:
    probs_samples = []
    logits_samples = []

    for _ in range(mc_samples):
        logits = model(X, training=True).numpy()
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        logits_samples.append(logits)
        probs_samples.append(probs)

    logits_samples = np.stack(logits_samples, axis=0)
    probs_samples = np.stack(probs_samples, axis=0)

    mean_probs = probs_samples.mean(axis=0)
    mean_logits = logits_samples.mean(axis=0)
    pred_class = mean_probs.argmax(axis=1)

    eps = 1e-8
    predictive_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, eps, 1.0)), axis=1)
    expected_entropy = -np.mean(
        np.sum(probs_samples * np.log(np.clip(probs_samples, eps, 1.0)), axis=-1),
        axis=0,
    )
    mutual_information = predictive_entropy - expected_entropy
    variation_ratio = 1.0 - np.max(mean_probs, axis=1)

    return {
        "mean_logits": mean_logits,
        "mean_probs": mean_probs,
        "pred_class": pred_class,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "variation_ratio": variation_ratio,
    }


def select_pseudo_label_candidates(
    pool_indices_rel: np.ndarray,
    mc_outputs: Dict[str, np.ndarray],
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mean_probs = mc_outputs["mean_probs"]
    pred_class = mc_outputs["pred_class"]
    predictive_entropy = mc_outputs["predictive_entropy"]
    mutual_information = mc_outputs["mutual_information"]
    variation_ratio = mc_outputs["variation_ratio"]

    confidence = np.max(mean_probs, axis=1)
    eligible_mask = (
        (confidence >= PSEUDO_MIN_CONFIDENCE)
        & (predictive_entropy <= PSEUDO_MAX_PREDICTIVE_ENTROPY)
        & (mutual_information <= PSEUDO_MAX_MUTUAL_INFORMATION)
        & (variation_ratio <= PSEUDO_MAX_VARIATION_RATIO)
    )

    selected_rel = []
    selected_labels = []
    selected_conf = []

    for cls in range(n_classes):
        cls_mask = eligible_mask & (pred_class == cls)
        cls_local = np.where(cls_mask)[0]
        if len(cls_local) == 0:
            continue

        cls_order = np.argsort(-confidence[cls_local])
        cls_local = cls_local[cls_order][:PSEUDO_MAX_NEW_PER_CLASS]

        selected_rel.extend(pool_indices_rel[cls_local].tolist())
        selected_labels.extend(pred_class[cls_local].tolist())
        selected_conf.extend(confidence[cls_local].tolist())

    if len(selected_rel) == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=np.int32),
            {
                "n_selected": 0,
                "class_counts": [0] * n_classes,
                "mean_confidence": None,
            },
        )

    selected_rel = np.asarray(selected_rel, dtype=int)
    selected_labels = np.asarray(selected_labels, dtype=np.int32)
    selected_conf = np.asarray(selected_conf, dtype=np.float32)

    order = np.argsort(-selected_conf)
    if len(order) > PSEUDO_MAX_NEW_PER_ROUND:
        order = order[:PSEUDO_MAX_NEW_PER_ROUND]

    selected_rel = selected_rel[order]
    selected_labels = selected_labels[order]
    selected_conf = selected_conf[order]

    class_counts = np.bincount(selected_labels, minlength=n_classes).tolist()
    info = {
        "n_selected": int(len(selected_rel)),
        "class_counts": class_counts,
        "mean_confidence": float(np.mean(selected_conf)),
        "mean_predictive_entropy": float(np.mean(predictive_entropy[eligible_mask])) if np.any(eligible_mask) else None,
        "mean_mutual_information": float(np.mean(mutual_information[eligible_mask])) if np.any(eligible_mask) else None,
    }
    return selected_rel, selected_labels, info


def run_uncertainty_filtered_pseudo_labeling(
    classifier: tf.keras.Model,
    X_outer_train_pad: np.ndarray,
    y_outer_train_cls: np.ndarray,
    y_labeled_train: np.ndarray,
    labeled_train_rel: np.ndarray,
    labeled_val_rel: np.ndarray,
    X_labeled_val: Optional[np.ndarray],
    y_labeled_val: Optional[np.ndarray],
    n_classes: int,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    base_train_indices = np.asarray(sorted(labeled_train_rel.tolist()), dtype=int)
    train_indices = base_train_indices.copy()
    train_labels = y_labeled_train.astype(np.int32).copy()

    held_out_labeled = np.union1d(labeled_train_rel, labeled_val_rel)
    pool_indices = np.setdiff1d(np.arange(len(X_outer_train_pad)), held_out_labeled)

    pseudo_history = []
    pseudo_audit_records = []

    if len(pool_indices) == 0:
        return classifier, train_indices, train_labels, pseudo_history, {
            "n_selected_total": 0,
            "acc": np.nan,
            "balanced_acc": np.nan,
            "macro_f1": np.nan,
            "class_counts": [0] * n_classes,
        }

    for round_idx in range(1, PSEUDO_MAX_ROUNDS + 1):
        if len(pool_indices) == 0:
            break

        mc_pool = monte_carlo_predict(
            model=classifier,
            X=X_outer_train_pad[pool_indices],
            mc_samples=MC_SAMPLES,
        )

        new_indices, new_labels, info = select_pseudo_label_candidates(
            pool_indices_rel=pool_indices,
            mc_outputs=mc_pool,
            n_classes=n_classes,
        )

        if len(new_indices) == 0:
            pseudo_history.append({
                "round": round_idx,
                "n_selected": 0,
                "remaining_pool": int(len(pool_indices)),
            })
            break

        true_new_labels = y_outer_train_cls[new_indices].astype(np.int32)
        round_metrics = compute_classification_metrics(true_new_labels, new_labels)
        pseudo_audit_records.append({
            "round": round_idx,
            "n_selected": int(len(new_indices)),
            "true_labels": true_new_labels.copy(),
            "pseudo_labels": new_labels.copy(),
            "acc": round_metrics["acc"],
            "balanced_acc": round_metrics["balanced_acc"],
            "macro_f1": round_metrics["macro_f1"],
        })

        train_indices = np.concatenate([train_indices, new_indices]).astype(int)
        train_labels = np.concatenate([train_labels, new_labels]).astype(np.int32)
        pool_indices = np.setdiff1d(pool_indices, new_indices)

        X_round_train = X_outer_train_pad[train_indices]
        y_round_train = train_labels

        n_true = len(base_train_indices)
        n_pseudo = len(train_indices) - n_true
        sample_weight = np.concatenate([
            np.ones(n_true, dtype=np.float32),
            np.full(n_pseudo, PSEUDO_LABEL_SAMPLE_WEIGHT, dtype=np.float32),
        ])

        classifier = continue_fine_tuning_classifier(
            classifier=classifier,
            X_train=X_round_train,
            y_train=y_round_train,
            X_val=X_labeled_val,
            y_val=y_labeled_val,
            epochs=PSEUDO_REFIT_EPOCHS,
            sample_weight=sample_weight,
        )

        pseudo_history.append({
            "round": round_idx,
            "n_selected": int(len(new_indices)),
            "selected_class_counts": info["class_counts"],
            "mean_confidence": info["mean_confidence"],
            "remaining_pool": int(len(pool_indices)),
            "n_total_train_after_round": int(len(train_indices)),
            "n_true_labels": int(n_true),
            "n_pseudo_labels": int(n_pseudo),
            "pseudo_acc": round_metrics["acc"],
            "pseudo_balanced_acc": round_metrics["balanced_acc"],
            "pseudo_macro_f1": round_metrics["macro_f1"],
        })

    if len(pseudo_audit_records) == 0:
        pseudo_audit_summary = {
            "n_selected_total": 0,
            "acc": np.nan,
            "balanced_acc": np.nan,
            "macro_f1": np.nan,
            "class_counts": [0] * n_classes,
        }
    else:
        y_true_all = np.concatenate([row["true_labels"] for row in pseudo_audit_records]).astype(np.int32)
        y_pred_all = np.concatenate([row["pseudo_labels"] for row in pseudo_audit_records]).astype(np.int32)
        overall_metrics = compute_classification_metrics(y_true_all, y_pred_all)
        pseudo_audit_summary = {
            "n_selected_total": int(len(y_true_all)),
            "acc": overall_metrics["acc"],
            "balanced_acc": overall_metrics["balanced_acc"],
            "macro_f1": overall_metrics["macro_f1"],
            "class_counts": np.bincount(y_pred_all, minlength=n_classes).tolist(),
        }

    return classifier, train_indices, train_labels, pseudo_history, pseudo_audit_summary


# =========================
# EVALUATION
# =========================
def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# =========================
# MAIN PIPELINE
# =========================
def run_grouped_ssl_fewshot_cv() -> pd.DataFrame:
    set_all_seeds(SEED)

    df = load_feature_table(INPUT_FEATURE_TABLE)
    ds = build_trial_sequences(df)

    n_trials = len(ds.X_list)
    n_features = len(ds.feature_cols)

    print(f"Trials: {n_trials}")
    print(f"Sessions: {len(np.unique(ds.groups))}")
    print(f"Feature dimension: {n_features}")
    print(f"Global max sequence length: {ds.global_max_len}")

    outer_cv = GroupKFold(n_splits=OUTER_N_SPLITS)
    dummy_X = np.zeros((n_trials, 1), dtype=np.float32)

    all_fold_rows = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(dummy_X, ds.raw_targets, ds.groups), start=1
    ):
        print("\n" + "=" * 72)
        print(f"OUTER FOLD {outer_fold}/{OUTER_N_SPLITS}")

        X_outer_train_seq = subset_list_by_indices(ds.X_list, outer_train_idx)
        X_outer_test_seq = subset_list_by_indices(ds.X_list, outer_test_idx)

        groups_outer_train = ds.groups[outer_train_idx]
        groups_outer_test = ds.groups[outer_test_idx]

        y_outer_train_cls, threshold_info = get_outer_fold_class_labels(
            direct_class_labels=ds.direct_class_labels,
            raw_targets=ds.raw_targets,
            outer_train_idx=outer_train_idx,
            target_idx=outer_train_idx,
        )
        y_outer_test_cls, _ = get_outer_fold_class_labels(
            direct_class_labels=ds.direct_class_labels,
            raw_targets=ds.raw_targets,
            outer_train_idx=outer_train_idx,
            target_idx=outer_test_idx,
        )

        n_classes = num_classes_from_labels(y_outer_train_cls)
        print(f"Outer train sessions: {np.unique(groups_outer_train)}")
        print(f"Outer test sessions : {np.unique(groups_outer_test)}")
        print(f"Classes in outer train: {np.unique(y_outer_train_cls)}")
        if threshold_info:
            print(f"Class construction info: {threshold_info}")

        scaler = fit_feature_scaler(X_outer_train_seq)
        X_outer_train_scaled = transform_sequence_list(X_outer_train_seq, scaler)
        X_outer_test_scaled = transform_sequence_list(X_outer_test_seq, scaler)

        X_outer_train_pad = pad_sequence_list(
            X_outer_train_scaled,
            max_len=ds.global_max_len,
            n_features=n_features,
        )
        X_outer_test_pad = pad_sequence_list(
            X_outer_test_scaled,
            max_len=ds.global_max_len,
            n_features=n_features,
        )

        ssl_train_rel, ssl_val_rel, ssl_train_groups, ssl_val_groups = choose_group_validation_split(
            groups=groups_outer_train,
            n_val_groups=SSL_VAL_GROUPS_PER_FOLD,
            seed=SEED + 10 * outer_fold,
        )

        print(f"SSL train groups: {ssl_train_groups}")
        print(f"SSL val groups  : {ssl_val_groups}")
        print(f"SSL train trials: {len(ssl_train_rel)}")
        print(f"SSL val trials  : {len(ssl_val_rel)}")

        # 1) SSL pretraining on ALL unlabeled outer-train trials with group-aware validation.
        set_all_seeds(SEED + outer_fold)
        encoder = pretrain_ssl_encoder(
            X_train_pad=X_outer_train_pad[ssl_train_rel],
            X_val_pad=X_outer_train_pad[ssl_val_rel],
            max_len=ds.global_max_len,
            n_features=n_features,
            hp=HP,
            seed=SEED + outer_fold,
        )

        # 2) Select exactly 2 support groups from outer-train data.
        support_groups = choose_labeled_groups(
            groups_outer_train=groups_outer_train,
            y_outer_train_cls=y_outer_train_cls,
            n_pick=LABELED_GROUPS_PER_FOLD,
        )
        support_mask = np.isin(groups_outer_train, support_groups)
        support_indices_rel = np.where(support_mask)[0]
        y_support = y_outer_train_cls[support_indices_rel]

        print(f"Selected support groups: {support_groups}")
        print(f"Support trials available: {len(support_indices_rel)}")
        print(f"Support class counts: {np.bincount(y_support, minlength=n_classes)}")

        # 3) Few-shot labeled split taken ONLY from those 2 groups.
        labeled_train_rel, labeled_val_rel = sample_few_shot_indices(
            candidate_indices=support_indices_rel,
            y_candidate=y_support,
            train_per_class=FEW_SHOT_TRAIN_PER_CLASS,
            val_per_class=FEW_SHOT_VAL_PER_CLASS,
            seed=SEED + 100 * outer_fold,
        )

        X_labeled_train = X_outer_train_pad[labeled_train_rel]
        y_labeled_train = y_outer_train_cls[labeled_train_rel]

        X_labeled_val = X_outer_train_pad[labeled_val_rel] if len(labeled_val_rel) > 0 else None
        y_labeled_val = y_outer_train_cls[labeled_val_rel] if len(labeled_val_rel) > 0 else None

        print(f"Few-shot labeled train trials: {len(X_labeled_train)}")
        print(f"Few-shot labeled val trials  : {0 if X_labeled_val is None else len(X_labeled_val)}")
        print(f"Few-shot train class counts  : {np.bincount(y_labeled_train, minlength=n_classes)}")

        # 4) Bayesian fine-tuning.
        set_all_seeds(SEED + 1000 + outer_fold)
        classifier = fine_tune_variational_classifier(
            encoder=encoder,
            X_labeled_train=X_labeled_train,
            y_labeled_train=y_labeled_train,
            X_labeled_val=X_labeled_val,
            y_labeled_val=y_labeled_val,
            n_classes=n_classes,
            max_len=ds.global_max_len,
            n_features=n_features,
            hp=HP,
            num_training_examples_for_kl=len(X_labeled_train),
            epochs=FINE_TUNE_EPOCHS,
        )

        pseudo_history = []
        pseudo_audit_summary = {
            "n_selected_total": 0,
            "acc": np.nan,
            "balanced_acc": np.nan,
            "macro_f1": np.nan,
            "class_counts": [0] * n_classes,
        }
        final_train_indices = labeled_train_rel.copy()
        final_train_labels = y_labeled_train.copy()
        if PSEUDO_LABELING_ENABLED:
            classifier, final_train_indices, final_train_labels, pseudo_history, pseudo_audit_summary = run_uncertainty_filtered_pseudo_labeling(
                classifier=classifier,
                X_outer_train_pad=X_outer_train_pad,
                y_outer_train_cls=y_outer_train_cls,
                y_labeled_train=y_labeled_train,
                labeled_train_rel=labeled_train_rel,
                labeled_val_rel=labeled_val_rel,
                X_labeled_val=X_labeled_val,
                y_labeled_val=y_labeled_val,
                n_classes=n_classes,
            )

        # 5) MC predictive probabilities + uncertainty on the outer test fold.
        test_mc = monte_carlo_predict(
            model=classifier,
            X=X_outer_test_pad,
            mc_samples=MC_SAMPLES,
        )
        y_test_pred = test_mc["pred_class"]
        metrics = compute_classification_metrics(y_outer_test_cls, y_test_pred)

        print("Outer fold test metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  mean predictive entropy: {float(np.mean(test_mc['predictive_entropy'])):.4f}")
        print(f"  mean mutual information: {float(np.mean(test_mc['mutual_information'])):.4f}")
        if pseudo_history:
            print("Pseudo-label rounds:")
            for row in pseudo_history:
                print(
                    f"  round {row['round']}: selected={row['n_selected']}, remaining_pool={row['remaining_pool']}, "
                    f"pseudo_acc={row.get('pseudo_acc', float('nan')):.4f}, "
                    f"pseudo_bal_acc={row.get('pseudo_balanced_acc', float('nan')):.4f}"
                )
            print(
                f"Pseudo-label audit overall: n={pseudo_audit_summary['n_selected_total']}, "
                f"acc={pseudo_audit_summary['acc']:.4f}, "
                f"balanced_acc={pseudo_audit_summary['balanced_acc']:.4f}, "
                f"macro_f1={pseudo_audit_summary['macro_f1']:.4f}"
            )

        total_pseudo = int(max(0, len(final_train_indices) - len(labeled_train_rel)))

        all_fold_rows.append({
            "outer_fold": outer_fold,
            "n_outer_train_trials": int(len(outer_train_idx)),
            "n_outer_test_trials": int(len(outer_test_idx)),
            "n_outer_train_sessions": int(len(np.unique(groups_outer_train))),
            "n_outer_test_sessions": int(len(np.unique(groups_outer_test))),
            "support_groups": ", ".join(map(str, support_groups.tolist())),
            "n_support_trials": int(len(support_indices_rel)),
            "n_labeled_train_trials": int(len(labeled_train_rel)),
            "n_labeled_val_trials": int(len(labeled_val_rel)),
            "n_classes": int(n_classes),
            "n_pseudo_rounds_run": int(len(pseudo_history)),
            "n_total_train_after_pseudo": int(len(final_train_indices)),
            "n_total_pseudo_labels_added": total_pseudo,
            "pseudo_audit_n_selected_total": int(pseudo_audit_summary["n_selected_total"]),
            "pseudo_audit_acc": float(pseudo_audit_summary["acc"]) if not np.isnan(pseudo_audit_summary["acc"]) else np.nan,
            "pseudo_audit_balanced_acc": float(pseudo_audit_summary["balanced_acc"]) if not np.isnan(pseudo_audit_summary["balanced_acc"]) else np.nan,
            "pseudo_audit_macro_f1": float(pseudo_audit_summary["macro_f1"]) if not np.isnan(pseudo_audit_summary["macro_f1"]) else np.nan,
            "mean_predictive_entropy": float(np.mean(test_mc["predictive_entropy"])),
            "mean_mutual_information": float(np.mean(test_mc["mutual_information"])),
            **metrics,
        })

    results_df = pd.DataFrame(all_fold_rows)

    print("\n" + "=" * 72)
    print("FINAL GROUPED SSL FEW-SHOT RESULTS")
    print(results_df)

    numeric_means = results_df.select_dtypes(include=[np.number]).mean(numeric_only=True)
    print("\nMean across outer folds:")
    print(numeric_means)

    return results_df


if __name__ == "__main__":
    run_grouped_ssl_fewshot_cv()
