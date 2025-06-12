import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
BALANCED_TRAIN_DATA_FILE = "../data/smote_balanced_training_data.csv"
PROCESSED_TEST_DATA_FILE = "../data/processed_test_data.csv"
TARGET_COLUMN = 'Glucose_Level_Class'

# --- 1. Load Preprocessed Training and Test Datasets ---
def load_data(file_path, dataset_name="data"):
    """Loads data from a CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {dataset_name}: {file_path} (Shape: {df.shape})")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {dataset_name} file '{file_path}' not found.")
        exit()
    return None

df_train_balanced = load_data(BALANCED_TRAIN_DATA_FILE, "balanced training data")
df_test_processed = load_data(PROCESSED_TEST_DATA_FILE, "processed test data")

# --- 2. Prepare Data for Model Training ---
print("\n⚙️  Preparing data for model training...")

# Separate features (X) and target (y)
X_train = df_train_balanced.drop(TARGET_COLUMN, axis=1)
y_train_str = df_train_balanced[TARGET_COLUMN]
X_test = df_test_processed.drop(TARGET_COLUMN, axis=1)
y_test_str = df_test_processed[TARGET_COLUMN]

# Align columns as a safeguard
if not X_train.columns.equals(X_test.columns):
    print("⚠️ Warning: Training and testing feature columns do not match perfectly. Aligning...")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Re-encode the target variable
le = LabelEncoder()
y_train = le.fit_transform(y_train_str)
y_test = le.transform(y_test_str)
target_names_for_report = list(le.classes_)
print(f"  Target variable re-encoded. Classes: {target_names_for_report}")
print(f"  Training set shape: X_train={X_train.shape}")
print(f"  Test set shape:     X_test={X_test.shape}")

# Note: Features are assumed to be already scaled and one-hot encoded from previous scripts.

# --- 3. Bayesian Optimization with K-Fold CV for LightGBM ---
print("\n🧠 Starting Bayesian Optimization for LightGBM (with K-Fold CV)...")

# Define the hyperparameter search space
search_spaces = {
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 12),
    'num_leaves': Integer(10, 100),
    'subsample': Real(0.6, 1.0, 'uniform'),
    'colsample_bytree': Real(0.6, 1.0, 'uniform'),
    'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
    'reg_lambda': Real(1e-9, 1.0, 'log-uniform'),
}

# Define the model
lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=42)

# Set up Stratified K-Fold cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up BayesSearchCV
# n_iter: Number of parameter settings to sample. More iterations can lead to better results.
bayes_search = BayesSearchCV(
    estimator=lgbm,
    search_spaces=search_spaces,
    n_iter=50,  # Number of iterations
    cv=cv_strategy,
    n_jobs=-1,  # Use all available CPU cores
    verbose=1,
    scoring='f1_weighted', # Optimize for weighted F1-score, a good metric for classification
    random_state=42
)

# Run the hyperparameter search on the training data
try:
    print("  Fitting BayesSearchCV... (This may take some time)")
    bayes_search.fit(X_train, y_train)
    print("✅ Bayesian Optimization complete.")

    # Get the best model found by the search
    best_tuned_model = bayes_search.best_estimator_
    print("\n⭐ Best Hyperparameters Found:")
    print(bayes_search.best_params_)
    print(f"\n⭐ Best Weighted F1-score during Cross-Validation: {bayes_search.best_score_:.4f}")

except Exception as e:
    print(f"❌ Error during Bayesian Optimization: {e}")
    exit()

# --- 4. Evaluate the Best Tuned Model on the Test Set ---
print("\n📊 Evaluating the best tuned model on the held-out test set...")
try:
    y_pred = best_tuned_model.predict(X_test)
    y_pred_proba = best_tuned_model.predict_proba(X_test)

    print(f"\n--- Final Results for Tuned LightGBM ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names_for_report, zero_division=0))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=le.transform(target_names_for_report))
    cm_df = pd.DataFrame(cm, index=target_names_for_report, columns=target_names_for_report)
    print(cm_df)

    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted', labels=le.transform(target_names_for_report))
    print(f"  Weighted AUC-ROC (ovr): {auc_score:.4f}")

except Exception as e:
    print(f"  ❌ Error during final evaluation: {e}")

# --- 5. SHAP Analysis for Model Interpretability (Corrected) ---
print("\n🔍 Starting SHAP Analysis to explain the final model...")

# Create a SHAP TreeExplainer object for our tree-based model (LightGBM)
explainer = shap.TreeExplainer(best_tuned_model)

# Calculate SHAP values for the test set. Using the explainer as a function
# is the modern API and returns a more robust Explanation object.
shap_explanation = explainer(X_test)

# For multi-class classification, shap_values is a list of arrays (one for each class).
# Let's visualize the feature importances for all classes.

# SHAP Global Summary Bar Plot
# This plot shows the mean absolute SHAP value for each feature, giving a
# clear ranking of global feature importance.
print("  Generating SHAP Global Feature Importance Plot (Bar)...")
shap.summary_plot(
    shap_explanation,
    X_test,
    plot_type="bar",
    class_names=target_names_for_report,
    title="Global Feature Importance (SHAP Bar Plot)"
)
plt.show()

# SHAP Summary Plots (Beeswarm) for each class
print("\n  Generating SHAP Summary Plots (Beeswarm) for each class...")
# This type of plot is very rich:
# - Each dot is a single prediction for a single sample.
# - Y-axis: Features, ordered by importance.
# - X-axis: SHAP value (impact on model output). Positive values push the prediction towards this class.
# - Color: Shows the original value of the feature (Red = high value, Blue = low value).
for i, class_name in enumerate(target_names_for_report):
    print(f"    - SHAP plot for class: {class_name}")
    # We pass the shap_values for a specific class `shap_explanation.values[:,:,i]`
    # and the base data `X_test`.
    shap.summary_plot(
        shap_explanation.values[:,:,i],
        X_test,
        show=False,
        plot_type="dot" # 'dot' is the beeswarm plot
    )
    plt.title(f"SHAP Summary for Class: {class_name}", fontsize=15)
    plt.show()

# --- 6. Save the Final Tuned Model and Preprocessors ---
print("\n💾 Saving the final tuned model and preprocessors for future use...")
joblib.dump(best_tuned_model, 'tuned_lgbm_classifier.joblib')
# We also need the other artifacts from the SMOTE script to predict on new, raw data:
# - scaler.joblib
# - label_encoder.joblib
# - training_feature_columns.joblib
# This step just saves the final, tuned model.
print("✅ Final tuned model saved to 'tuned_lgbm_classifier.joblib'")

print("\n🎉 Full pipeline (Tuning, Evaluation, SHAP Analysis) finished.")