# Genomic Variant Classifier 🧬

## 📌 Project Overview
This project features a highly optimized, hybrid ensemble machine learning pipeline designed to classify genomic variants as **Benign** or **Pathogenic**. The architecture leverages a weighted blend of advanced gradient boosting algorithms to maximize precision and recall on highly imbalanced clinical data.

**Core Tech Stack:** `Python`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `Scikit-Learn`, `Pandas`

## ⚙️ Pipeline Architecture & Methodology
- **Leakage-Free Stratification:** Robust 5-Fold Stratified Out-Of-Fold (OOF) Validation to prevent data leakage.
- **Automated Categorical Encoding:** Dynamic handling of missing values and unseen categorical variables (`_UNK_`).
- **Clinical Simulation:** 80/20 targeted panel-aware jury simulation ratio for validation-side blend analysis.
- **Custom Evaluation Proxy:** Models are strictly evaluated on Pathogenic F1, MCC, and PR-AUC. The final ensemble is optimized using a custom internal proxy: `(F1 + MCC) / 2`.
- **Hyperparameter Optimization:** `Optuna` is utilized strictly to find the optimal probability-level blend weights across models (XGBoost, LightGBM, CatBoost) using a fixed 0.50 decision threshold.
- **Explainable AI (XAI):** Integrated `SHAP` TreeExplainer to ensure model transparency and clinical feature importance.

## 🚀 How to Use & Reproduce

Since the original dataset was provided exclusively for the competition, you will need to provide your own structured datasets to train the model.

### 1. Prerequisites & Installation
Ensure you have Python 3.9+ installed. Clone this repository and install the required dependencies:
```bash
git clone https://github.com/BurcuUzun/genomic-variant-classifier.git
cd genomic-variant-classifier
pip install -r requirements.txt
```

### 2. Data Preparation (Crucial Step)
Place your genomic data files directly into the root directory of this repository. The script expects **four specific CSV files**. You must name them exactly as follows:
- `master_leakage_free.csv`
- `cancer_flagged.csv`
- `pah_flagged.csv`
- `cftr_flagged.csv`

### 3. Execution
Once the data is in place, simply run the training script:
```bash
python train_model.py
```

### 4. Expected Outputs
Upon successful execution, the script will automatically generate the following artifacts in your root directory:
- `genomic_variant_classifier.pkl`: The finalized, Optuna-weighted ensemble model ready for clinical inference.
- `confusion_matrix.png`: Visualizes the model's exact classification performance (Benign vs. Pathogenic) on the clinical simulation dataset.
- `pr_curve.png`: Highlights the Precision-Recall tradeoff, emphasizing the ensemble's robustness against the highly imbalanced nature of genomic data.
- `shap_summary_plot.png`: Provides clinical transparency (XAI) by ranking the most influential genomic features driving the model's pathogenic predictions.
---
*Developed as a submission for the TEKNOFEST AI in Health 2026 competition.*
