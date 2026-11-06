"""NEWLY RECONSTRUCTED BASELINE. NOT original study code or reproduced results.
Usage: python analysis/baseline.py --data /secure/path/approved.csv
Never upload confidential or identifiable datasets.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score

TARGET = "stress_label"
GROUP = "subject_id"
EXCLUDE = []

def evaluate(df):
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}; adapt schema before running.")
    if GROUP and GROUP not in df.columns:
        raise ValueError(f"Missing participant group column: {GROUP}. Do not use row-wise splitting.")
    needed = [TARGET] + ([GROUP] if GROUP else [])
    clean = df.dropna(subset=needed).copy()
    y = clean[TARGET]
    if set(y.unique()) - {0, 1}:
        raise ValueError("Outcome must be explicitly coded 0/1; do not silently recode.")
    drop_cols = [TARGET] + ([GROUP] if GROUP else []) + EXCLUDE
    X = clean.drop(columns=drop_cols, errors="ignore")
    # Reject identifiers and unreviewed columns before scientific use.
    suspicious = [c for c in X.columns if any(k in c.lower() for k in ["patient_id","subject_id","participant_id","name","email","phone"])]
    if suspicious:
        raise ValueError(f"Remove possible identifiers before analysis: {suspicious}")
    if GROUP:
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        tr, te = next(splitter.split(X, y, groups=clean[GROUP]))
        assert set(clean.iloc[tr][GROUP]).isdisjoint(set(clean.iloc[te][GROUP]))
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]), categorical))
    if not transforms:
        raise ValueError("No predictors found.")
    model = Pipeline([("preprocess",ColumnTransformer(transforms)),
                      ("classifier",LogisticRegression(max_iter=2000, random_state=42))])
    if len(np.unique(y.iloc[tr])) < 2 or len(np.unique(y.iloc[te])) < 2:
        raise ValueError("Both classes required in train and test. Redesign split.")
    model.fit(X.iloc[tr], y.iloc[tr])
    score = model.predict_proba(X.iloc[te])[:, 1]
    return {"test_n":len(te), "roc_auc":roc_auc_score(y.iloc[te],score),
            "pr_auc":average_precision_score(y.iloc[te],score),
            "balanced_accuracy":balanced_accuracy_score(y.iloc[te],score>=.5)}

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True,type=Path)
    args=parser.parse_args()
    if not args.data.is_file():
        parser.error("Data file not found. Provide an authorized local CSV.")
    print("RECONSTRUCTED BASELINE ONLY — NOT PUBLISHED RESULTS")
    print(evaluate(pd.read_csv(args.data)))
