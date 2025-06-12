import pandas as pd
import numpy as np
import joblib  # For loading saved model and objects
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress UndefinedMetricWarning for precision/recall when a class has no predictions
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- Configuration for New Data Testing ---
NEW_REAL_DATA_FILE = "../data/test_new_data.csv"  # <<< YOUR NEW REAL DATASET
SAVED_MODEL_FILE = 'best_glucose_classifier.joblib'
SAVED_SCALER_FILE = 'scaler.joblib'
SAVED_LABEL_ENCODER_FILE = 'label_encoder.joblib'
SAVED_TRAINING_COLUMNS_FILE = 'training_feature_columns.joblib'

TARGET_COLUMN = 'Glucose_Level_Class'  # Should be consistent

# Define categorical features that would have been one-hot encoded during training
# This list should match the one used when the model was trained
CATEGORICAL_FEATURES_IN_X_RAW = ['Type_of_Diabetes', 'Gender']  # Example

# --- 1. Load the Saved Model and Preprocessing Objects ---
print("🔄 Loading saved model and preprocessors...")
try:
    model = joblib.load(SAVED_MODEL_FILE)
    scaler = joblib.load(SAVED_SCALER_FILE)
    le = joblib.load(SAVED_LABEL_ENCODER_FILE)
    training_columns = joblib.load(SAVED_TRAINING_COLUMNS_FILE)  # List of feature names model expects
    print("✅ Model and preprocessors loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: Could not load a required file. Ensure all .joblib files exist: {e}")
    exit()
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# --- 2. Load Your New Real Data ---
print(f"\n🔄 Loading new real dataset: {NEW_REAL_DATA_FILE}...")
try:
    df_new_real = pd.read_csv(NEW_REAL_DATA_FILE)
    print(f"✅ Successfully loaded new real data. Shape: {df_new_real.shape}")
except FileNotFoundError:
    print(f"❌ Error: New real data file '{NEW_REAL_DATA_FILE}' not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading new real data file: {e}")
    exit()

# --- 3. Preprocess the New Real Data (CRITICAL: Apply IDENTICAL steps as training data) ---
print("\n⚙️  Preprocessing new real data...")

# Make a copy to avoid potential SettingWithCopyWarning
X_new_real = df_new_real.copy()

# a) Separate target variable (if present for evaluation)
if TARGET_COLUMN in X_new_real.columns:
    y_new_real_str = X_new_real.pop(TARGET_COLUMN)  # Use .pop to get target and remove from X
    y_new_real_encoded = le.transform(y_new_real_str)
    print(f"  Target variable '{TARGET_COLUMN}' separated and encoded for new data.")
else:
    print(
        f"⚠️ Warning: Target column '{TARGET_COLUMN}' not found in new real data. Proceeding with feature processing only.")
    y_new_real_str = None
    y_new_real_encoded = None

# b) Drop the raw BGL column from the features to prevent data leakage
if 'Blood_Glucose_mg/dL' in X_new_real.columns:
    X_new_real = X_new_real.drop(columns=['Blood_Glucose_mg/dL'])
    print("  Dropped 'Blood_Glucose_mg/dL' from the new data's feature set.")
else:
    print("  'Blood_Glucose_mg/dL' was not found in the new data's features to be dropped (which is expected).")

# c) Handle initial NaNs if necessary (this logic must match your original preprocessing)
if 'HbA1c_%' in X_new_real.columns and X_new_real['HbA1c_%'].isnull().any():
    # Using the median of the *new data* as a fallback. A more robust approach
    # would be to save and load the median/mean from the original training data.
    hba1c_median_new_data = X_new_real['HbA1c_%'].median()
    X_new_real['HbA1c_%'] = X_new_real['HbA1c_%'].fillna(hba1c_median_new_data)
    print(f"  Filled NaNs in 'HbA1c_%' of new data with its median: {hba1c_median_new_data:.2f}")
# (Add similar logic for other columns like Breath_Acetone_ppm, β-Hydroxybutyrate_mmol/L if they had NaN handling)


# d) One-Hot Encode Categorical Features
actual_categorical_features_new_data = [col for col in CATEGORICAL_FEATURES_IN_X_RAW if col in X_new_real.columns]
if actual_categorical_features_new_data:
    print(f"  One-hot encoding categorical features in new data: {actual_categorical_features_new_data}")
    X_new_real = pd.get_dummies(X_new_real, columns=actual_categorical_features_new_data, drop_first=True)
else:
    print("  No specified categorical features found in new data for one-hot encoding.")

# e) Align Columns with Training Data (Crucial after one-hot encoding)
print("  Aligning columns of new data with training data columns...")
# Use reindex to align columns, which is a robust way to handle this
# It adds missing columns with NaN (which we fill with 0) and ensures the order is correct.
X_new_real_aligned = X_new_real.reindex(columns=training_columns, fill_value=0)
print(f"  New data columns aligned. Shape before scaling: {X_new_real_aligned.shape}")

# f) Scale Numerical Features using the LOADED scaler
# This final processed dataframe is the one we will use for prediction
print("  Scaling new data features using the loaded scaler...")
try:
    # Get the feature names the scaler was fitted on, IN ORDER.
    # This requires scikit-learn version >= 0.24.
    if hasattr(scaler, 'feature_names_in_'):
        scaler_cols_ordered = scaler.feature_names_in_
    else:
        # Fallback for older scikit-learn versions: define the list manually.
        # This list MUST match the order and names from your SMOTE script exactly.
        print(
            "  Warning: scaler.feature_names_in_ not found. Using manually defined list for scaling. Ensure order is correct.")
        scaler_cols_ordered = [
            'Age', 'BMI', 'HbA1c_%', 'Breath_Acetone_ppm', 'β-Hydroxybutyrate_mmol/L',
            'Temp_C', 'Humidity_%', 'Fasting_Hours'
        ]
        # Filter to only columns that are actually in the aligned dataframe
        scaler_cols_ordered = [col for col in scaler_cols_ordered if col in X_new_real_aligned.columns]

    # Create a copy to work with, which will be our final processed DataFrame
    X_new_real_processed = X_new_real_aligned.copy()

    if scaler_cols_ordered.size > 0:  # Or check len() if it's a list
        # Apply the scaler transformation ONLY to the numerical columns it was trained on,
        # ensuring they are presented in the correct order.
        X_new_real_processed[scaler_cols_ordered] = scaler.transform(X_new_real_aligned[scaler_cols_ordered])
        print("  New data features scaled successfully.")
    else:
        print("  No numerical columns identified to scale.")

except Exception as e:
    print(f"❌ Error scaling new data: {e}")
    exit()

# --- 4. Make Predictions on the Preprocessed New Real Data ---
print("\n🤖 Making predictions on the new real data...")
try:
    y_pred_new_real = model.predict(X_new_real_processed)  # This variable is now defined
    y_pred_new_real_str = le.inverse_transform(y_pred_new_real)  # Convert predictions back to string labels

    if hasattr(model, "predict_proba"):
        y_pred_proba_new_real = model.predict_proba(X_new_real_processed)
    else:
        y_pred_proba_new_real = None
    print("✅ Predictions made.")
except Exception as e:
    print(f"❌ Error during prediction on new data: {e}")
    exit()

# --- 5. Evaluate Performance (if true labels are available for the new data) ---
if y_new_real_encoded is not None:
    print("\n📊 Evaluating model on the new real test set...")
    accuracy_new_real = accuracy_score(y_new_real_encoded, y_pred_new_real)
    print(f"  Accuracy on new real data: {accuracy_new_real:.4f}")

    print("\n  Classification Report on new real data:")
    target_names_from_loaded_le = list(le.classes_)
    print(classification_report(y_new_real_encoded, y_pred_new_real, target_names=target_names_from_loaded_le,
                                zero_division=0))

    print("  Confusion Matrix on new real data:")
    cm_new_real = confusion_matrix(y_new_real_encoded, y_pred_new_real,
                                   labels=le.transform(target_names_from_loaded_le))
    cm_df_new_real = pd.DataFrame(cm_new_real, index=target_names_from_loaded_le, columns=target_names_from_loaded_le)
    print(cm_df_new_real)

    if y_pred_proba_new_real is not None and len(np.unique(y_new_real_encoded)) > 1:
        try:
            auc_score_new_real = roc_auc_score(y_new_real_encoded, y_pred_proba_new_real, multi_class='ovr',
                                               average='weighted', labels=le.transform(target_names_from_loaded_le))
            print(f"  Weighted AUC-ROC on new real data: {auc_score_new_real:.4f}")
        except ValueError as e_auc:
            print(f"  Could not calculate AUC-ROC on new data: {e_auc}")
else:
    print(
        "\nℹ️ True labels for new real data not available for evaluation. Predictions generated and can be inspected (e.g., y_pred_new_real_str).")

print("\n🎉 Testing on new real data script finished.")
