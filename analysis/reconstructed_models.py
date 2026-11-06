"""Independently reconstructed model comparison, NOT original study code/results."""
import argparse, json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss
TARGET = "stress_label"
GROUP = "subject_id"
EXCLUDE = []

def run(path, algorithm="random_forest"):
    df=pd.read_csv(path)
    required=[TARGET]+([GROUP] if GROUP else [])
    if any(c not in df for c in required): raise ValueError("Required outcome/group column missing")
    df=df.dropna(subset=required)
    y=df[TARGET]
    if set(y.unique())!={0,1}: raise ValueError("Target must be binary 0/1 with both classes")
    X=df.drop(columns=required+EXCLUDE,errors="ignore")
    identifiers=[c for c in X if any(k in c.lower() for k in ("patient_id","participant_id","subject_id","email","phone"))]
    if identifiers: raise ValueError(f"Potential identifier predictors: {identifiers}")
    if X.shape[1]==0: raise ValueError("No predictors")
    if GROUP:
        groups=df[GROUP]
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,y,groups))
        assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te]))
        n=groups.iloc[tr].nunique()
        if n<3: raise ValueError("Insufficient participant groups")
        cv=GroupKFold(n_splits=min(n,5))
        fit_groups=groups.iloc[tr]
    else:
        tr,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
        n=int(y.iloc[tr].value_counts().min())
        if n<3: raise ValueError("Insufficient minority class")
        cv=StratifiedKFold(n_splits=min(5,n),shuffle=True,random_state=42)
        fit_groups=None
    if y.iloc[te].nunique()!=2: raise ValueError("Test fold lacks both classes")
    nums=X.select_dtypes(include="number").columns.tolist()
    cats=X.select_dtypes(exclude="number").columns.tolist()
    steps=[]
    if nums: steps.append(("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
    if cats: steps.append(("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats))
    models={
        "logistic":(LogisticRegression(max_iter=2000),{"model__C":[.1,1,10]}),
        "random_forest":(RandomForestClassifier(random_state=42,n_jobs=-1),{"model__n_estimators":[100,200],"model__min_samples_leaf":[1,3,5]}),
        "hist_gradient_boosting":(HistGradientBoostingClassifier(random_state=42),{"model__learning_rate":[.03,.1],"model__max_leaf_nodes":[7,15,31]})
    }
    estimator,params=models[algorithm]
    pipe=Pipeline([("preprocess",ColumnTransformer(steps)),("model",estimator)])
    search=RandomizedSearchCV(pipe,params,n_iter=3,cv=cv,scoring="roc_auc",random_state=42,n_jobs=-1,error_score="raise")
    search.fit(X.iloc[tr],y.iloc[tr],groups=fit_groups)
    p=search.best_estimator_.predict_proba(X.iloc[te])[:,1]
    return {"warning":"RECONSTRUCTED PIPELINE; NOT MANUSCRIPT RESULTS","algorithm":algorithm,"n_test":len(te),"roc_auc":float(roc_auc_score(y.iloc[te],p)),"pr_auc":float(average_precision_score(y.iloc[te],p)),"balanced_accuracy":float(balanced_accuracy_score(y.iloc[te],p>=.5)),"brier":float(brier_score_loss(y.iloc[te],p)),"best_params":search.best_params_}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--algorithm",choices=["logistic","random_forest","hist_gradient_boosting"],default="random_forest")
    args=ap.parse_args()
    print(json.dumps(run(args.data,args.algorithm),indent=2))
