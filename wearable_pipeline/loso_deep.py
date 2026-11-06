"""Reconstructed subject-held-out E4-style time-series pipeline.
NOT original study code. Input is authorized, timestamped tabular data.
"""
import argparse
import json
import os
import random
import numpy as np
import pandas as pd

SIGNALS = ["eda", "hr", "temp", "acc_x", "acc_y", "acc_z"]
REQUIRED = ["subject_id", "timestamp", "stress_label"] + SIGNALS

def build_windows(frame, seconds=60, min_coverage=0.8):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if not set(frame["stress_label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("stress_label must be binary 0/1")
    samples, labels, groups = [], [], []
    for subject, part in frame.groupby("subject_id", sort=False):
        part = part.sort_values("timestamp").set_index("timestamp")
        if part.index.has_duplicates:
            raise ValueError(f"Duplicate timestamps for subject {subject}; resolve first")
        # Do not interpolate over recording gaps or subject boundaries.
        signals = part[SIGNALS].resample("1s").mean()
        label = part["stress_label"].resample("1s").agg(
            lambda s: s.iloc[0] if len(s) and s.notna().all() and s.nunique() == 1 else np.nan
        )
        # Windows anchored at start of each subject's recording.
        n = len(signals) // seconds
        for i in range(n):
            lo, hi = i * seconds, (i + 1) * seconds
            x = signals.iloc[lo:hi].to_numpy(dtype=np.float32)
            y = label.iloc[lo:hi]
            valid = np.isfinite(x).all(axis=1) & y.notna().to_numpy()
            if valid.mean() < min_coverage or y.dropna().nunique() != 1:
                continue
            # Missing samples retained as NaN; fold-specific training imputation below.
            samples.append(x)
            labels.append(int(y.dropna().iloc[0]))
            groups.append(str(subject))
    if not samples:
        raise ValueError("No valid windows. Check timestamps, sampling coverage and labels.")
    return np.stack(samples), np.asarray(labels), np.asarray(groups)

def transform_fold(x_train, x_val, x_test):
    # Channel-wise statistics from training subjects only; never from validation/test.
    with np.errstate(all="ignore"):
        median = np.nanmedian(x_train, axis=(0, 1))
    if not np.isfinite(median).all():
        raise ValueError("A sensor channel has no finite training observations")
    def impute(x):
        return np.where(np.isfinite(x), x, median[None, None, :])
    a = impute(x_train)
    mean = a.mean(axis=(0, 1))
    std = a.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1, std)
    return tuple(((impute(x) - mean) / std).astype("float32")
                 for x in (x_train, x_val, x_test))

def build_model(architecture, length, channels, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(length, channels))
    if architecture == "cnn":
        x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(inp)
        x = tf.keras.layers.MaxPool1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif architecture == "lstm":
        x = tf.keras.layers.LSTM(32)(inp)
    else:
        raise ValueError("architecture must be cnn or lstm")
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy")
    return model

def evaluate(data, architecture="cnn", epochs=20, seed=42):
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
    X, y, groups = build_windows(data)
    subjects = sorted(np.unique(groups))
    if len(subjects) < 4:
        raise ValueError("At least 4 subjects required: train, validation and test")
    results = []
    for k, held_out in enumerate(subjects):
        test = groups == held_out
        available = [s for s in subjects if s != held_out]
        # Deterministic rotation; validation is a different participant.
        val_subject = available[k % len(available)]
        val = groups == val_subject
        train = ~(test | val)
        if len(np.unique(y[train])) != 2 or len(np.unique(y[val])) != 2:
            results.append({"test_subject":held_out,"status":"skipped_insufficient_training_or_validation_classes"})
            continue
        a, b, c = transform_fold(X[train], X[val], X[test])
        tf.keras.backend.clear_session()
        model = build_model(architecture, X.shape[1], X.shape[2], seed + k)
        counts = np.bincount(y[train], minlength=2)
        weights = {i: float(len(y[train]) / (2 * counts[i])) for i in (0, 1)}
        model.fit(a, y[train], validation_data=(b, y[val]), epochs=epochs,
                  batch_size=32, class_weight=weights, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=4, restore_best_weights=True)])
        score = model.predict(c, verbose=0).ravel()
        pred = (score >= 0.5).astype(int)
        row = {"test_subject":held_out,"validation_subject":val_subject,
               "n_windows":int(test.sum()),"positive_windows":int(y[test].sum()),
               "balanced_accuracy":float(balanced_accuracy_score(y[test],pred)),
               "f1":float(f1_score(y[test],pred,zero_division=0)),
               "roc_auc":float(roc_auc_score(y[test],score)) if len(np.unique(y[test])) == 2 else None}
        results.append(row)
    return {"status":"reconstructed_code_not_original_results","architecture":architecture,
            "n_subjects":len(subjects),"n_windows":len(y),"folds":results}

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--architecture",choices=["cnn","lstm"],default="cnn")
    parser.add_argument("--epochs",type=int,default=20)
    args=parser.parse_args()
    if args.epochs < 1: parser.error("epochs must be positive")
    result=evaluate(pd.read_csv(args.data),args.architecture,args.epochs)
    print(json.dumps(result,indent=2))
