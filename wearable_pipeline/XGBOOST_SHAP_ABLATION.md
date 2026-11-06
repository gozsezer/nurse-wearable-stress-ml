# Reconstructed XGBoost + SHAP + sensor ablation

**Research code newly written for this repository; not recovered original scripts.** No original participant data or empirical results are included.

## Input
An authorized local CSV with `subject_id,timestamp,eda,hr,temp,acc_x,acc_y,acc_z,stress_label`. The shared `loso_deep.build_windows` function builds 60-second windows from timestamped data; it is not a native E4 file parser. Confirm label provenance and original signal sampling rates.

## Methods
- One held-out participant per test fold (LOSO).
- Four window-level summary features per sensor channel (mean, SD, min, max).
- Imputation learned only from training participants.
- XGBoost with fixed illustrative hyperparameters and training-only class weights.
- Six leave-one-channel-out ablations, each retrained without that channel.
- Optional mean absolute SHAP values on held-out subjects.

**Interpretation:** SHAP measures fitted-model attribution, not causal importance. Ablation differences are descriptive; no hyperparameter search or external validation is implemented. Some participants may have one-class test labels; ROC-AUC is then reported as null.

## Run
```bash
pip install -r wearable_pipeline/requirements_xgb.txt
python wearable_pipeline/xgb_shap_ablation.py --data /secure/approved.csv --output results/loso.json --shap-output results/shap.csv
python -m unittest discover -s wearable_pipeline/tests
```

Do not upload participant data, model artifacts containing private information, or the generated outputs without review. The published study's reported metrics are **not** reproduced by these files.
