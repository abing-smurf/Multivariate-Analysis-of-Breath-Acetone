import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb  # For LightGBM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress UndefinedMetricWarning for precision/recall when a class has no predictions
# This can happen if a class is very small or a model performs poorly.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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
    except Exception as e:
        print(f"❌ Error loading {dataset_name} file '{file_path}': {e}")
        exit()


df_train_balanced = load_data(BALANCED_TRAIN_DATA_FILE, "balanced training data")
df_test_processed = load_data(PROCESSED_TEST_DATA_FILE, "processed test data")

# --- 2. Prepare Data for Model Training ---
print("\n⚙️  Preparing data for model training...")

# Separate features (X) and target (y) for training data
if TARGET_COLUMN not in df_train_balanced.columns:
    print(f"❌ Error: Target column '{TARGET_COLUMN}' not found in the balanced training DataFrame.")
    exit()
X_train = df_train_balanced.drop(TARGET_COLUMN, axis=1)
y_train_str = df_train_balanced[TARGET_COLUMN]

# Separate features (X) and target (y) for test data
if TARGET_COLUMN not in df_test_processed.columns:
    print(f"❌ Error: Target column '{TARGET_COLUMN}' not found in the processed test DataFrame.")
    exit()
X_test = df_test_processed.drop(TARGET_COLUMN, axis=1)
y_test_str = df_test_processed[TARGET_COLUMN]

# Ensure column order and presence is consistent between X_train and X_test
# This should ideally be handled during the preprocessing step that created these files.
# However, as a safeguard:
if not X_train.columns.equals(X_test.columns):
    print("⚠️ Warning: Training and testing feature columns do not match perfectly.")
    print(f"   X_train columns: {X_train.columns.tolist()}")
    print(f"   X_test columns: {X_test.columns.tolist()}")

    # Attempt to align based on X_train columns
    common_cols = X_train.columns.intersection(X_test.columns)
    missing_in_test = X_train.columns.difference(X_test.columns)

    if missing_in_test.any():
        print(
            f"   Columns in X_train but missing in X_test (will be filled with 0 in X_test): {missing_in_test.tolist()}")
        for col in missing_in_test:
            X_test[col] = 0  # Add missing columns and fill with 0 (or other appropriate value)

    # Ensure X_test has the same columns in the same order as X_train
    try:
        X_test = X_test[X_train.columns]
        print("   Columns in X_test reordered and aligned with X_train.")
    except KeyError as e:
        print(
            f"❌ Error aligning X_test columns with X_train: {e}. Some columns in X_train might be missing in X_test even after attempting to add them.")
        print("   This typically indicates a more fundamental mismatch in how the datasets were prepared.")
        exit()

# Re-encode the target variable (string labels to numerical)
# Fit LabelEncoder on the combined unique target labels to ensure all labels are known
le = LabelEncoder()
all_known_labels = pd.concat([y_train_str, y_test_str]).unique()
le.fit(all_known_labels)

y_train = le.transform(y_train_str)
y_test = le.transform(y_test_str)

print(f"  Target variable '{TARGET_COLUMN}' re-encoded for modeling.")
print(f"  Training set shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"  Test set shapes:     X_test={X_test.shape}, y_test={y_test.shape}")
target_names_for_report = list(le.classes_)
print(f"  LabelEncoder classes: {target_names_for_report} are mapped to {list(range(len(target_names_for_report)))}")

# Note: Features in X_train and X_test are assumed to be already scaled
# and one-hot encoded from the previous SMOTE data preparation script.

# --- 3. Define and Train Models ---
# Using default parameters for a first pass. Hyperparameter tuning can be added later.
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr', solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "MLP Classifier": MLPClassifier(random_state=42, max_iter=500, early_stopping=True, hidden_layer_sizes=(64, 32)),
    # Added early stopping and a basic architecture
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_estimators=100)
}

trained_models = {}
print("\n🏋️  Training models...")

for model_name, model_instance in models.items():
    print(f"  Training {model_name}...")
    try:
        model_instance.fit(X_train, y_train)
        trained_models[model_name] = model_instance
        print(f"    ✅ {model_name} trained successfully.")
    except Exception as e:
        print(f"    ❌ Error training {model_name}: {e}")

# --- 4. Make Predictions and Evaluate Models ---
print("\n📊 Evaluating models on the test set...")

for model_name, model in trained_models.items():
    print(f"\n--- Results for {model_name} ---")
    try:
        y_pred = model.predict(X_test)

        # Ensure y_pred_proba is available and correctly shaped for multi-class AUC
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        else:  # For models like SVM without probability=True by default
            y_pred_proba = None
            print(
                "    Note: predict_proba not available for this model, AUC might not be calculated or use decision_function.")

        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {accuracy:.4f}")

        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names_for_report, zero_division=0))

        print("  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=le.transform(target_names_for_report))  # Ensure labels order
        cm_df = pd.DataFrame(cm, index=target_names_for_report, columns=target_names_for_report)
        print(cm_df)

        # AUC-ROC for multi-class
        if y_pred_proba is not None and len(np.unique(y_test)) > 1 and y_pred_proba.shape[1] == len(
                target_names_for_report):
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted',
                                          labels=le.transform(target_names_for_report))
                print(f"  Weighted AUC-ROC (ovr): {auc_score:.4f}")
            except ValueError as e_auc:
                print(f"  Could not calculate AUC-ROC: {e_auc}")
        elif y_pred_proba is None:
            print("  AUC-ROC not calculated (predict_proba not available).")
        elif y_pred_proba.shape[1] != len(target_names_for_report):
            print(
                f"  AUC-ROC not calculated (predict_proba shape {y_pred_proba.shape} mismatch with number of classes {len(target_names_for_report)}).")
        else:
            print("  AUC-ROC not calculated (only one class present in y_test or other issue).")

    except Exception as e:
        print(f"  ❌ Error during prediction or evaluation for {model_name}: {e}")

# In your model training script, after a model is trained and you've identified it as your best one
import joblib # For saving and loading scikit-learn models and objects

# --- Determine your best model ---
# You would typically look at the evaluation metrics printed above
# to decide which model is 'best'. For this example, let's assume
# Random Forest was chosen.
best_model_name = "Random Forest"  # CHANGE THIS based on your evaluation

if best_model_name in trained_models:
    best_model_object = trained_models[best_model_name]

    print(f"\n💾 Saving chosen model ({best_model_name}) and related preprocessors...")

    # 1. Save the chosen best model object
    joblib.dump(best_model_object, 'best_glucose_classifier.joblib')
    print(f"  ✅ Saved best model object ({best_model_name}) to 'best_glucose_classifier.joblib'")

    # 2. Save the LabelEncoder (fitted in this script)
    joblib.dump(le, 'label_encoder.joblib')
    print(f"  ✅ Saved LabelEncoder object to 'label_encoder.joblib'")

    # 3. Save the training feature columns (from X_train in this script)
    joblib.dump(X_train.columns.tolist(), 'training_feature_columns.joblib')
    print(f"  ✅ Saved training feature column names to 'training_feature_columns.joblib'")

    print(
        "\n📌 Reminder: The 'scaler.joblib' (StandardScaler) should have been saved from your SMOTE data preparation script, as it was fitted there.")
    print(
        "   You will need 'best_glucose_classifier.joblib', 'scaler.joblib', 'label_encoder.joblib', and 'training_feature_columns.joblib' to make predictions on new, raw data.")

else:
    print(f"⚠️ Error: Model '{best_model_name}' not found in trained_models. Cannot save.")

print("\n🎉 Model training, evaluation, and artifact saving (for chosen model) script finished.")

print("\n🎉 Model training and evaluation script finished.")
