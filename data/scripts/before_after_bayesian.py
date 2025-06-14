# ==============================================================================
# Thesis Assistant: Before vs. After Tuning Comparison for XGBoost
# ==============================================================================
# Description:
# This script directly measures the impact of hyperparameter tuning by comparing
# the performance of an XGBoost model with its default settings against the
# same model using the optimal parameters found via Bayesian Optimization.
#
# It uses a robust Stratified K-Fold cross-validation approach with SMOTE
# applied correctly within each fold to ensure a fair and reliable comparison.
# The results will quantify the performance gain achieved through tuning.
#
# Instructions:
# 1.  Install required libraries:
#     pip install pandas scikit-learn imbalanced-learn xgboost
# 2.  Update the `DATA_FILE` path to your augmented dataset.
# 3.  Run the script. It will evaluate both model versions and print a
#     summary table comparing their performance.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load and Prepare Data ---
DATA_FILE = "../data/augmented_data_by_noise.csv"

try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Data loaded successfully from '{DATA_FILE}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Data file '{DATA_FILE}' not found. Please check the path.")
    exit()

# Prevent data leakage
if 'Blood_Glucose_mg/dL' in df.columns:
    df = df.drop(columns=['Blood_Glucose_mg/dL'])
    print("✅ Dropped 'Blood_Glucose_mg/dL' column.")

# Define features (X) and target (y)
TARGET_COLUMN = 'Glucose_classification_class'
FEATURES = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Pulse rate',
    'HbA1c_%', 'Respiratory Rate', 'Acetone PPM 1.1', 'Temperature', 'Humidity'
]

X = df[FEATURES]
y = df[TARGET_COLUMN]

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Target classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- 2. Define Both XGBoost Model Versions ---
models_to_compare = {
    "Default XGBoost": xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    ),
    "Tuned XGBoost": xgb.XGBClassifier(
        n_estimators=477,
        learning_rate=0.010290943451240405,
        max_depth=4,
        subsample=0.10998412035247775,
        colsample_bytree=0.4584385158503276,
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
}

# --- 3. Perform Cross-Validated Evaluation ---
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
results = {}

print(f"\n🚀 Starting {N_SPLITS}-Fold Cross-Validation to compare model versions...")

for model_name, model in models_to_compare.items():
    print(f"\n--- Evaluating: {model_name} ---")

    fold_f1_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        score = f1_score(y_val, y_pred, average='weighted')
        fold_f1_scores.append(score)
        print(f"  Fold {fold + 1}/{N_SPLITS} - F1-score: {score:.4f}")

    results[model_name] = {'Avg F1-score (Weighted)': np.mean(fold_f1_scores)}

# --- 4. Display Final Comparison Table ---
print("\n🎉 Comparison Finished!")
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='Avg F1-score (Weighted)', ascending=False)

print("\n--- XGBoost Performance: Before vs. After Tuning ---")
print(results_df)

# Calculate and print the performance improvement
if "Default XGBoost" in results_df.index and "Tuned XGBoost" in results_df.index:
    default_score = results_df.loc["Default XGBoost", "Avg F1-score (Weighted)"]
    tuned_score = results_df.loc["Tuned XGBoost", "Avg F1-score (Weighted)"]
    improvement = ((tuned_score - default_score) / default_score) * 100
    print(f"\nImprovement from tuning: +{improvement:.2f}%")
