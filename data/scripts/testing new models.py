# ==============================================================================
# Thesis Assistant: Model Benchmarking
# ==============================================================================
# Description:
# This script systematically evaluates and compares the performance of multiple
# state-of-the-art classification models, as outlined in the thesis methodology.
# It uses a robust Stratified K-Fold cross-validation approach, with SMOTE
# applied correctly within each fold to handle class imbalance.
#
# **Version 2 Update**: Corrected a critical data leakage issue by removing the
# 'Blood_Glucose_mg/dL' column before training. This ensures the models are
# evaluated on their ability to predict the class from biomarkers only.
#
# The goal is to benchmark AdaBoost, XGBoost, CatBoost, and LightGBM to
# identify the best-performing model for the given task.
#
# Instructions:
# 1.  Install required libraries:
#     pip install pandas scikit-learn imbalanced-learn xgboost lightgbm catboost
# 2.  Update the `DATA_FILE` path to your augmented dataset.
# 3.  Run the script. It will train and evaluate each model, then print a
#     summary table comparing their performance.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Import Classifiers
from sklearn.ensemble import AdaBoostClassifier
import xgboost_model as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

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

# *** DATA LEAKAGE PREVENTION ***
# The 'Glucose_classification_class' is derived from 'Blood_Glucose_mg/dL'.
# To prevent data leakage, we must drop the direct glucose measurement
# before defining our features and training the models.
if 'Blood_Glucose_mg/dL' in df.columns:
    df = df.drop(columns=['Blood_Glucose_mg/dL'])
    print("✅ Dropped 'Blood_Glucose_mg/dL' column to prevent data leakage.")

# Define features (X) and target (y)
TARGET_COLUMN = 'Glucose_classification_class'
FEATURES = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Pulse rate',
    'HbA1c_%', 'Respiratory Rate', 'Acetone PPM 1.1', 'Temperature', 'Humidity'
]

# Ensure all specified features exist in the dataframe
existing_features = [f for f in FEATURES if f in df.columns]
if len(existing_features) != len(FEATURES):
    print("⚠️ Warning: Some features were not found in the dataframe.")
    FEATURES = existing_features

X = df[FEATURES]
y = df[TARGET_COLUMN]

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Target classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- 2. Define Models for Benchmarking ---
# We use the best parameters found for XGBoost and strong defaults for others.
models = {
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=477,
        learning_rate=0.01029,
        max_depth=4,
        subsample=0.1099,
        colsample_bytree=0.4584,
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    ),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)  # verbose=0 to prevent excessive logging
}

# --- 3. Perform Cross-Validated Evaluation ---
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Dictionary to store the results for each model
results = {}

print(f"\n🚀 Starting {N_SPLITS}-Fold Cross-Validation for {len(models)} models...")

for model_name, model in models.items():
    print(f"\n--- Evaluating: {model_name} ---")

    # Lists to store scores for each fold
    fold_accuracy = []
    fold_f1 = []
    fold_precision = []
    fold_recall = []

    # Loop through each fold
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded)):
        # Split data for this fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        # Create a pipeline: SMOTE -> Classifier
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        # Train the pipeline on the fold's training data
        pipeline.fit(X_train, y_train)

        # Make predictions on the validation data
        y_pred = pipeline.predict(X_val)

        # Calculate and store metrics for this fold
        fold_accuracy.append(accuracy_score(y_val, y_pred))
        fold_f1.append(f1_score(y_val, y_pred, average='weighted'))
        fold_precision.append(precision_score(y_val, y_pred, average='weighted'))
        fold_recall.append(recall_score(y_val, y_pred, average='weighted'))

        print(f"  Fold {fold + 1}/{N_SPLITS} - F1-score: {fold_f1[-1]:.4f}")

    # Store the average scores for the model
    results[model_name] = {
        'Avg Accuracy': np.mean(fold_accuracy),
        'Avg F1-score (Weighted)': np.mean(fold_f1),
        'Avg Precision (Weighted)': np.mean(fold_precision),
        'Avg Recall (Weighted)': np.mean(fold_recall)
    }

# --- 4. Display Final Comparison Table ---
print("\n🎉 Benchmarking Finished!")

results_df = pd.DataFrame(results).T  # Transpose for better readability
results_df = results_df.sort_values(by='Avg F1-score (Weighted)', ascending=False)

print("\n--- Model Performance Comparison ---")
print(results_df)

print("\nBased on this comparison, you can select the best overall model for your final evaluation.")
