import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import shap
import joblib
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, matthews_corrcoef, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_curve,
                             PrecisionRecallDisplay, average_precision_score,
                             classification_report)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
xgb.set_config(verbosity=0)

SEED = 42
np.random.seed(SEED)
BLEND_TRIALS = 50

base_path = Path.cwd()

# 1. Data Loading
dfs = []
for file_name in ["master_leakage_free.csv", "cancer_flagged.csv", "pah_flagged.csv", "cftr_flagged.csv"]:
    df_temp = pd.read_csv(base_path / file_name)
    original_cols = [c for c in df_temp.columns if not c.endswith("_missing")]
    dfs.append(df_temp[original_cols])

df_train = pd.concat(dfs, ignore_index=True)

# 2. Categorical Encoding
cat_columns = df_train.select_dtypes(include=['object', 'category', 'string']).columns
cat_dict = {}

for col in cat_columns:
    train_vals = df_train[col].astype("string").fillna("_MISSING_")
    categories = list(pd.Series(train_vals.unique()).dropna())
    if "_UNK_" not in categories:
        categories.append("_UNK_")
    df_train[col] = pd.Categorical(train_vals, categories=categories)
    cat_dict[col] = categories

X_train = df_train.drop(columns=['Variant_ID', 'Label'])
y_train = df_train['Label']
w_train = compute_sample_weight(class_weight='balanced', y=y_train)
cat_features = list(X_train.select_dtypes(include=['category']).columns)

# 3. Model Hyperparameters
xgb_params = {'tree_method': 'hist', 'enable_categorical': True, 'random_state': SEED, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8}
lgb_params = {'random_state': SEED, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 5, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'verbose': -1}
cat_params = {'random_state': SEED, 'iterations': 300, 'learning_rate': 0.03, 'depth': 5, 'l2_leaf_reg': 3.0, 'cat_features': cat_features, 'verbose': False}

# 4. Out-Of-Fold (OOF) Probability Generation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

for tr_idx, va_idx in skf.split(X_train, y_train):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
    w_tr = w_train[tr_idx]

    m_xgb = xgb.XGBClassifier(**xgb_params).fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    m_lgb = lgb.LGBMClassifier(**lgb_params).fit(X_tr, y_tr, sample_weight=w_tr)
    m_cat = cb.CatBoostClassifier(**cat_params).fit(X_tr, y_tr, sample_weight=w_tr)

    oof_xgb[va_idx] = m_xgb.predict_proba(X_va)[:, 1]
    oof_lgb[va_idx] = m_lgb.predict_proba(X_va)[:, 1]
    oof_cat[va_idx] = m_cat.predict_proba(X_va)[:, 1]

# 5. 80/20 Clinical Simulation & Optuna Blending
benign_idx = np.where(y_train == 0)[0]
pathogenic_idx = np.where(y_train == 1)[0]
target_pathogenic = min(len(benign_idx) // 4, len(pathogenic_idx))

rng = np.random.default_rng(SEED)
sim_idx = np.concatenate([benign_idx, rng.choice(pathogenic_idx, target_pathogenic, replace=False)])

y_sim = y_train.iloc[sim_idx].values
sim_xgb, sim_lgb, sim_cat = oof_xgb[sim_idx], oof_lgb[sim_idx], oof_cat[sim_idx]

def objective(trial):
    w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
    w_lgb = trial.suggest_float('w_lgb', 0.0, 1.0)
    w_cat = trial.suggest_float('w_cat', 0.0, 1.0)
    
    total_w = w_xgb + w_lgb + w_cat
    if total_w == 0: return 0.0
    
    w_xgb, w_lgb, w_cat = w_xgb/total_w, w_lgb/total_w, w_cat/total_w
    sim_blended_prob = (w_xgb * sim_xgb) + (w_lgb * sim_lgb) + (w_cat * sim_cat)
    
    preds = (sim_blended_prob >= 0.50).astype(int)
    f1 = f1_score(y_sim, preds, pos_label=1)
    mcc = matthews_corrcoef(y_sim, preds)
    return (f1 + mcc) / 2.0

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=BLEND_TRIALS)

total = study.best_params['w_xgb'] + study.best_params['w_lgb'] + study.best_params['w_cat']
best_w_xgb = study.best_params['w_xgb'] / total
best_w_lgb = study.best_params['w_lgb'] / total
best_w_cat = study.best_params['w_cat'] / total

# 6. Final Model Training
final_xgb = xgb.XGBClassifier(**xgb_params).fit(X_train, y_train, sample_weight=w_train, verbose=False)
final_lgb = lgb.LGBMClassifier(**lgb_params).fit(X_train, y_train, sample_weight=w_train)
final_cat = cb.CatBoostClassifier(**cat_params).fit(X_train, y_train, sample_weight=w_train)

# 7. Saving the Artifact
ensemble_artifact = {
    'xgb_model': final_xgb,
    'lgb_model': final_lgb,
    'cat_model': final_cat,
    'weights': {'xgb': best_w_xgb, 'lgb': best_w_lgb, 'cat': best_w_cat},
    'features': list(X_train.columns),
    'categories': cat_dict
}

joblib.dump(ensemble_artifact, base_path / "genomic_variant_classifier.pkl")

# 8. Evaluation & Visualizations
oof_blended_prob = (best_w_xgb * oof_xgb) + (best_w_lgb * oof_lgb) + (best_w_cat * oof_cat)
oof_preds = (oof_blended_prob >= 0.50).astype(int)

print("\n--- ENSEMBLE OOF PERFORMANCE ---")
print(f"F1 Score: {f1_score(y_train, oof_preds, pos_label=1):.4f}")
print(f"PR-AUC:   {average_precision_score(y_train, oof_blended_prob):.4f}")
print("-" * 32)
print(classification_report(y_train, oof_preds, target_names=['Benign (0)', 'Pathogenic (1)']))

# Confusion Matrix
cm = confusion_matrix(y_train, oof_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Pathogenic'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Ensemble Confusion Matrix")
plt.savefig(base_path / "confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# PR Curve
precision, recall, _ = precision_recall_curve(y_train, oof_blended_prob)
disp_pr = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision_score(y_train, oof_blended_prob))
disp_pr.plot(color='darkred')
plt.title("Ensemble Precision-Recall Curve")
plt.savefig(base_path / "pr_curve.png", dpi=150, bbox_inches='tight')
plt.close()

# 9. SHAP Explainability Plot
explainer_xgb = shap.TreeExplainer(final_xgb)
explainer_lgb = shap.TreeExplainer(final_lgb)
explainer_cat = shap.TreeExplainer(final_cat)

X_shap_sample = X_train.sample(n=min(1000, len(X_train)), random_state=SEED)

def get_shap_array(explainer, X):
    vals = explainer.shap_values(X)
    if isinstance(vals, list): 
        return vals[1]
    elif len(vals.shape) == 3: 
        return vals[:, :, 1]
    return vals

shap_xgb = get_shap_array(explainer_xgb, X_shap_sample)
shap_lgb = get_shap_array(explainer_lgb, X_shap_sample)
shap_cat = get_shap_array(explainer_cat, X_shap_sample)

blended_shap_values = (best_w_xgb * shap_xgb) + (best_w_lgb * shap_lgb) + (best_w_cat * shap_cat)

plt.figure(figsize=(10, 6))
shap.summary_plot(blended_shap_values, X_shap_sample, show=False)
plt.title("Hybrid Ensemble SHAP Analysis")
plt.tight_layout()
plt.savefig(base_path / "shap_summary_plot.png", dpi=300)
plt.close()