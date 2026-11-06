# Empatica E4 stress pipeline — independently reconstructed

**Status: newly authored research code, not recovered original scripts. No original nurse data or empirical performance reproduced.**

## Input contract
A locally held, authorized CSV with columns `subject_id,timestamp,eda,hr,temp,acc_x,acc_y,acc_z,stress_label`. Timestamp must be parseable and represent a consistent timezone. `stress_label` must be binary 0/1. All signals are assumed already synchronized or time-stamped; native Empatica E4 files have different sampling rates and need separately documented import and synchronization. **Do not treat this pipeline as a native raw E4 parser.**

## Processing and evaluation
- Within each subject, resample signals to one-second bins (mean); missing seconds remain missing, never filled across subjects.
- Generate non-overlapping 60-second windows; reject windows with insufficient coverage or mixed labels. Window labels must not be constructed from future information.
- Fit missing-value imputation and normalization **inside training subjects only** for each LOSO fold.
- Reserve one training subject for early stopping; never use the held-out test subject for selection.
- Evaluate 1D-CNN and LSTM using subject-held-out predictions; report per-subject and pooled metrics with denominators. Pooled metrics are descriptive because windows within subjects are dependent.
- No SMOTE applied to test data; class weighting on training data is used instead.
- AUC is undefined for a held-out subject containing one class; output null rather than inventing a value.

## Usage
```bash
pip install -r wearable_pipeline/requirements.txt
python wearable_pipeline/loso_deep.py --data /secure/approved.csv --architecture cnn --epochs 20
python wearable_pipeline/loso_deep.py --data /secure/approved.csv --architecture lstm --epochs 20
```
The output JSON is printed to stdout. No individual participant data or trained model is uploaded automatically.

**Scientific limitations:** this is a reference implementation, not a reproduction of reported 0.864 AUC. Preprocessing, label provenance, timing, original sensor frequencies, class balance, participant-specific artifacts, and ethical sharing rights must be audited before scientific interpretation. Hyperparameter selection and external validation are not implemented here.