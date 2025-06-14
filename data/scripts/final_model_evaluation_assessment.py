# ==============================================================================
# Thesis Assistant: Final Model Training, Evaluation, and Interpretation
# ==============================================================================
# Description:
# This script represents the final stage of the machine learning pipeline. It
# takes the best model (XGBoost) with its optimized hyperparameters and trains
# it on a portion of the augmented dataset. It then evaluates the model's
# performance on a completely unseen hold-out test set.
#
# **Version 2 Update**: Added SHAP (SHapley Additive exPlanations) analysis
# to provide deeper model interpretability, as outlined in the thesis.
#
# The script generates four key outputs for the thesis:
# 1.  A detailed Classification Report.
# 2.  A Confusion Matrix visualization.
# 3.  A standard Feature Importance plot.
# 4.  A SHAP Summary Plot for feature impact analysis.
#
# Instructions:
# 1.  Install required libraries:
#     pip install pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn shap
# 2.  Update the `DATA_FILE` path to your augmented dataset.
# 3.  Run the script to get the final performance metrics and plots.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
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

# Drop the leaky column before any other processing
if 'Blood_Glucose_mg/dL' in df.columns:
    df = df.drop(columns=['Blood_Glucose_mg/dL'])
    print("✅ Dropped 'Blood_Glucose_mg/dL' column to prevent data leakage.")

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
class_names = le.classes_
print(f"Target classes encoded: {dict(zip(class_names, le.transform(class_names)))}")

# --- 2. Create the Final Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)
print(f"\nData split into training set ({X_train.shape[0]} samples) and test set ({X_test.shape[0]} samples).")

# --- 3. Apply SMOTE to the Training Data ONLY ---
print("\nApplying SMOTE to the training data to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Training data resampled. New shape: {X_train_resampled.shape}")

# --- 4. Initialize and Train the Final XGBoost Model ---
print("\nTraining the final XGBoost model with optimized hyperparameters...")
final_model = xgb.XGBClassifier(
    n_estimators=477,
    learning_rate=0.010290943451240405,
    max_depth=4,
    subsample=0.10998412035247775,
    colsample_bytree=0.4584385158503276,
    objective='multi:softmax',
    num_class=len(class_names),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

final_model.fit(X_train_resampled, y_train_resampled)
print("✅ Final model trained successfully.")

# --- 5. Evaluate the Model on the Unseen Test Set ---
print("\nEvaluating model performance on the unseen test set...")
y_pred = final_model.predict(X_test)

print("\n--- Classification Report ---")
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.show()

# --- 6. Analyze Feature Importance (Standard) ---
print("\n--- Feature Importance (Gain-based) ---")
feature_importances = pd.DataFrame({
    'feature': FEATURES,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importances)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
plt.title('Feature Importance for Glucose Classification (Gain-based)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

# --- 7. Analyze Feature Importance with SHAP ---
print("\n--- SHAP Analysis for Model Interpretability ---")
print("Calculating SHAP values... This may take a moment.")

# Create an explainer object
explainer = shap.TreeExplainer(final_model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

print("Plotting SHAP summary plot...")
# The summary plot shows the overall impact of each feature
# For multi-class, shap_values is a list of arrays. We can plot them all.
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    class_names=class_names,
    feature_names=FEATURES
)

# A beeswarm plot gives more detail on individual predictions
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURES,
    class_names=class_names
)

print("\n🎉 Final evaluation complete.")