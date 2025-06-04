import pandas as pd
import numpy as np
import joblib # For loading saved model and objects
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
# Ensure all necessary preprocessing functions (like classify_glucose if used on raw BGL before other steps)
# and column name constants are available or redefined here if not part of saved objects.

# --- Configuration for New Data Testing ---
NEW_REAL_DATA_FILE = "../data/test_new_data_correlated_v1.csv"  # <<< YOUR NEW REAL DATASET
SAVED_MODEL_FILE = 'best_glucose_classifier.joblib'
SAVED_SCALER_FILE = 'scaler.joblib'
SAVED_LABEL_ENCODER_FILE = 'label_encoder.joblib'
SAVED_TRAINING_COLUMNS_FILE = 'training_feature_columns.joblib'

TARGET_COLUMN = 'Glucose_Level_Class' # Should be consistent

# Define categorical features that would have been one-hot encoded during training
# This list should match the one used when the model was trained
CATEGORICAL_FEATURES_IN_X_RAW = ['Type_of_Diabetes', 'Gender'] # Example

# --- 1. Load the Saved Model and Preprocessing Objects ---
print("🔄 Loading saved model and preprocessors...")
try:
    model = joblib.load(SAVED_MODEL_FILE)
    scaler = joblib.load(SAVED_SCALER_FILE)
    le = joblib.load(SAVED_LABEL_ENCODER_FILE)
    training_columns = joblib.load(SAVED_TRAINING_COLUMNS_FILE) # List of feature names model expects
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

# Make a copy to avoid SettingWithCopyWarning if df_new_real is used later
X_new_real = df_new_real.copy()

# a) Separate target variable (if present for evaluation)
if TARGET_COLUMN in X_new_real.columns:
    y_new_real_str = X_new_real.pop(TARGET_COLUMN) # Use .pop to remove it from X_new_real
    y_new_real_encoded = le.transform(y_new_real_str) # Use the LOADED label encoder
    print(f"  Target variable '{TARGET_COLUMN}' separated and encoded for new data.")
else:
    print(f"⚠️ Warning: Target column '{TARGET_COLUMN}' not found in new real data. Will proceed with feature processing only (for prediction).")
    y_new_real_str = None
    y_new_real_encoded = None

# b) Initial NaN Handling (Example - must match your original preprocessing logic)
#    This part needs to be IDENTICAL to how your *original* training data was preprocessed
#    BEFORE it was fed into the train_test_split and SMOTE pipeline.
#    For example, if you imputed HbA1c with a mean from the *original training set*, you'd ideally use that saved mean here,
#    or apply a consistent strategy like median imputation on the new data if that's what you did.
#    For simplicity, let's assume your original preprocessing (that created augmented_glucose_data_noise_added_cleaned.csv)
#    already handled NaNs robustly before being input to the SMOTE script.
#    If df_new_real is truly raw, you'd replicate those initial cleaning steps.
#    Example using median imputation for any column that had it in training:
#    if 'HbA1c_%' in X_new_real.columns and X_new_real['HbA1c_%'].isnull().any():
#        # Ideally, use a saved mean/median from the original training data.
#        # For now, let's assume new data needs its own imputation if NaNs are present
#        hba1c_median_new_data = X_new_real['HbA1c_%'].median()
#        X_new_real['HbA1c_%'] = X_new_real['HbA1c_%'].fillna(hba1c_median_new_data)
#        print(f"  Filled NaNs in 'HbA1c_%' of new data with its median: {hba1c_median_new_data:.2f}")
#    (Repeat for other columns like Breath_Acetone_ppm, β-Hydroxybutyrate_mmol/L if they had similar initial NaN/negative handling)

# c) One-Hot Encode Categorical Features
actual_categorical_features_new_data = [col for col in CATEGORICAL_FEATURES_IN_X_RAW if col in X_new_real.columns]
if actual_categorical_features_new_data:
    print(f"  One-hot encoding categorical features in new data: {actual_categorical_features_new_data}")
    X_new_real = pd.get_dummies(X_new_real, columns=actual_categorical_features_new_data, drop_first=True)
else:
    print("  No specified categorical features found in new data for one-hot encoding.")

# d) Align Columns with Training Data (Crucial after one-hot encoding)
print("  Aligning columns of new data with training data columns...")
X_new_real_aligned = pd.DataFrame(columns=training_columns) # Create empty df with training columns
for col in training_columns:
    if col in X_new_real.columns:
        X_new_real_aligned[col] = X_new_real[col]
    else:
        X_new_real_aligned[col] = 0 # Add missing columns (from OHE) and fill with 0
# Ensure order is identical
X_new_real_processed = X_new_real_aligned[training_columns]
print(f"  New data aligned. Shape before scaling: {X_new_real_processed.shape}")


# e) Scale Numerical Features using the LOADED scaler
# Identify numerical features based on the training_columns (as OHE might have changed dtypes)
# This assumes training_columns contains all features the model expects.
# We need to infer numerical columns from the loaded scaler's properties if possible or define them.
# A simpler way is to ensure your `training_columns` list was created AFTER OHE
# and before scaling, then find numeric columns within *that list*.
# For now, let's assume all columns in training_columns are numeric if they were scaled,
# or that the scaler can handle non-numeric columns by ignoring them (StandardScaler does this).
# More robust: Identify which columns the scaler was originally fitted on.
# For simplicity, if all training_columns were originally meant to be scaled (after OHE):
try:
    X_new_real_processed = pd.DataFrame(scaler.transform(X_new_real_processed), columns=training_columns)
    print("  New data features scaled successfully using the loaded scaler.")
except ValueError as e:
    print(f"❌ Error scaling new data. Ensure columns match what the scaler expects: {e}")
    print("   Make sure only numerical columns that were originally scaled are being passed to scaler.transform,")
    print("   and that they are in the same order as when the scaler was fitted.")
    exit()


# --- 4. Make Predictions on the Preprocessed New Real Data ---
print("\n🤖 Making predictions on the new real data...")
try:
    y_pred_new_real = model.predict(X_new_real_processed)
    y_pred_new_real_str = le.inverse_transform(y_pred_new_real) # Convert predictions back to string labels

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
    # Use target_names from the loaded label encoder
    target_names_from_loaded_le = list(le.classes_)
    print(classification_report(y_new_real_encoded, y_pred_new_real, target_names=target_names_from_loaded_le, zero_division=0))

    print("  Confusion Matrix on new real data:")
    cm_new_real = confusion_matrix(y_new_real_encoded, y_pred_new_real, labels=le.transform(target_names_from_loaded_le))
    cm_df_new_real = pd.DataFrame(cm_new_real, index=target_names_from_loaded_le, columns=target_names_from_loaded_le)
    print(cm_df_new_real)

    if y_pred_proba_new_real is not None and len(np.unique(y_new_real_encoded)) > 1 and y_pred_proba_new_real.shape[1] == len(target_names_from_loaded_le):
        try:
            auc_score_new_real = roc_auc_score(y_new_real_encoded, y_pred_proba_new_real, multi_class='ovr', average='weighted', labels=le.transform(target_names_from_loaded_le))
            print(f"  Weighted AUC-ROC on new real data: {auc_score_new_real:.4f}")
        except ValueError as e_auc:
            print(f"  Could not calculate AUC-ROC on new data: {e_auc}")
    # ... (other AUC messages as in previous script) ...
else:
    print("\nℹ️ True labels for new real data not available for evaluation. Predictions generated and can be inspected (e.g., y_pred_new_real_str).")


print("\n🎉 Testing on new real data script finished.")