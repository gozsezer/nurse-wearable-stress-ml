"""Independent reconstruction: XGBoost, SHAP and LOSO sensor ablation.
Not original study code; no empirical results are reproduced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
try:
    from .loso_deep import build_windows, SIGNALS
except ImportError:
    from loso_deep import build_windows, SIGNALS

def features(windows):
    columns, names = [], []
    for j, channel in enumerate(SIGNALS):
        a = windows[:, :, j]
        with np.errstate(all="ignore"):
            count = np.isfinite(a).sum(axis=1)
            avg = np.divide(np.where(np.isfinite(a), a, 0).sum(axis=1), count,
                            out=np.full(len(a), np.nan), where=count>0)
            sd = np.sqrt(np.divide(np.where(np.isfinite(a), (a-avg[:,None])**2, 0).sum(axis=1),
                                   count, out=np.full(len(a), np.nan), where=count>0))
            lo = np.where(np.isfinite(a), a, np.inf).min(axis=1)
            hi = np.where(np.isfinite(a), a, -np.inf).max(axis=1)
            lo[count==0], hi[count==0] = np.nan, np.nan
        for stat, values in (("mean",avg),("std",sd),("min",lo),("max",hi)):
            names.append(channel+"_"+stat)
            columns.append(values)
    return np.column_stack(columns), names

def train_model(x, y, seed):
    from xgboost import XGBClassifier
    counts=np.bincount(y,minlength=2)
    if min(counts)==0: raise ValueError("Training requires both classes")
    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("xgb",XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,
            subsample=.9,colsample_bytree=.9,eval_metric="logloss",n_jobs=1,
            random_state=seed,scale_pos_weight=float(counts[0]/counts[1])))])
    model.fit(x,y)
    return model

def score(y,p):
    pred=(p>=.5).astype(int)
    return {"n":int(len(y)),"positives":int(y.sum()),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "f1":float(f1_score(y,pred,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}

def evaluate(windows, labels, groups, shap_csv=None):
    x, names=features(windows)
    subjects=sorted(set(groups))
    if len(subjects)<3: raise ValueError("At least 3 subjects needed")
    rows, shap_rows=[],[]
    for fold, subject in enumerate(subjects):
        tr,te=groups!=subject,groups==subject
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        if len(np.unique(labels[tr]))<2:
            rows.append({"subject":str(subject),"status":"skipped_one_training_class"})
            continue
        model=train_model(x[tr],labels[tr],42+fold)
        baseline=score(labels[te],model.predict_proba(x[te])[:,1])
        ablations={}
        for channel in SIGNALS:
            keep=[j for j,name in enumerate(names) if not name.startswith(channel+"_")]
            reduced=train_model(x[tr][:,keep],labels[tr],42+fold)
            result=score(labels[te],reduced.predict_proba(x[te][:,keep])[:,1])
            ablations[channel]={"reduced":result,
                "delta_balanced_accuracy":baseline["balanced_accuracy"]-result["balanced_accuracy"],
                "delta_roc_auc":baseline["roc_auc"]-result["roc_auc"]
                if baseline["roc_auc"] is not None else None}
        if shap_csv:
            import shap
            transformed=model.named_steps["imputer"].transform(x[te])
            values=np.asarray(shap.TreeExplainer(model.named_steps["xgb"])(transformed).values)
            if values.shape!=(int(te.sum()),len(names)):
                raise ValueError("Unexpected SHAP dimensions")
            shap_rows.append({"subject":str(subject),**{
                name:float(np.mean(np.abs(values[:,j]))) for j,name in enumerate(names)}})
        rows.append({"subject":str(subject),"status":"evaluated",
                     "full_model":baseline,"ablation":ablations})
    if shap_csv and shap_rows:
        path=Path(shap_csv)
        path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(shap_rows).to_csv(path,index=False)
    return {"status":"reconstructed_not_original_results",
            "n_subjects":len(subjects),"feature_names":names,"folds":rows}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--output")
    parser.add_argument("--shap-output")
    args=parser.parse_args()
    windows,labels,groups=build_windows(pd.read_csv(args.data))
    result=evaluate(windows,labels,groups,args.shap_output)
    report=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(report,encoding="utf-8")
    else:
        print(report)
