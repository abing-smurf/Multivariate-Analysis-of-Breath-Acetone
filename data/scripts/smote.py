import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # For target encoding and feature scaling
from imblearn.over_sampling import SMOTE
from collections import Counter # To display class distributions

# --- Configuration ---
AUGMENTED_DATA_FILE = "../data/augmented_glucose_data_noise_added_cleaned.csv" # From your noise-adding script
TARGET_COLUMN = 'Glucose_Level_Class'

# Define categorical features in your dataset (excluding the target)
# These will need to be one-hot encoded before SMOTE if SMOTE variant doesn't handle them.
# Standard SMOTE in imblearn expects all numerical input for X.
# Example:
CATEGORICAL_FEATURES_IN_X = ['Type_of_Diabetes', 'Gender'] # Add actual categorical feature names
# Filter to only those present in the dataframe later

# --- 1. Load Your Augmented Data ---
try:
    df_augmented = pd.read_csv(AUGMENTED_DATA_FILE)
    print(f"✅ Successfully loaded augmented data: {AUGMENTED_DATA_FILE} (Shape: {df_augmented.shape})")
except FileNotFoundError:
    print(f"❌ Error: Augmented data file '{AUGMENTED_DATA_FILE}' not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading augmented data file '{AUGMENTED_DATA_FILE}': {e}")
    exit()

# --- 2. Prepare Features (X) and Target (y) ---

# Define columns to drop from the feature set to prevent data leakage
COLS_TO_DROP_FROM_FEATURES = ['Glucose_Level_Class', 'Blood_Glucose_mg/dL']

X = df_augmented.drop(columns=COLS_TO_DROP_FROM_FEATURES) # Drop BOTH the label and its source
y = df_augmented[TARGET_COLUMN]

# --- 3. Preprocessing for SMOTE ---
# a) Encode Categorical Features in X (e.g., using One-Hot Encoding)
# SMOTE requires numerical input.
print("\n⚙️  Preprocessing features for SMOTE...")
actual_categorical_features = [col for col in CATEGORICAL_FEATURES_IN_X if col in X.columns]
if actual_categorical_features:
    print(f"  One-hot encoding categorical features: {actual_categorical_features}")
    X = pd.get_dummies(X, columns=actual_categorical_features, drop_first=True) # drop_first to avoid multicollinearity
else:
    print("  No specified categorical features found in X for one-hot encoding.")

# b) Encode Target Variable (y)
# SMOTE expects numerical labels for the target variable.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  Target variable '{TARGET_COLUMN}' encoded. Classes: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

# c) Identify all numerical columns in X for scaling
# (after one-hot encoding, new columns are created, existing numerical ones remain)
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
if not numerical_features:
    print("❌ Error: No numerical features found in X after preprocessing. SMOTE cannot proceed.")
    exit()
print(f"  Numerical features to be scaled: {numerical_features}")


# --- 4. Split Data into Training and Testing Sets ---
# CRITICAL: Split BEFORE applying SMOTE
# Using stratification to maintain class proportions in train/test split based on y_encoded
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)
print(f"\nData split into training and testing sets.")
print(f"  Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"  Testing set shape:  X_test={X_test.shape}, y_test={y_test.shape}")

# Display class distribution BEFORE SMOTE on the training set
print("\nClass distribution in original training data (y_train):")
print(Counter(le.inverse_transform(y_train))) # Show original labels

# d) Scale Numerical Features (AFTER splitting, fit ONLY on X_train)
# SMOTE is distance-based (due to k-NN), so scaling is recommended.
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features]) # Use same scaler fitted on train
joblib.dump(scaler, 'scaler.joblib')
print("  ✅ Fitted StandardScaler saved to 'scaler.joblib'")
print("  Numerical features scaled (StandardScaler).")


# --- 5. Apply SMOTE to the Training Data ---
print("\nApplying SMOTE to the training data...")
# You can adjust k_neighbors. Default is 5.
# sampling_strategy can be 'auto' (resample all but majority), 'minority', a float, or a dict.
# For multi-class, a dictionary can specify desired samples per class,
# or 'not majority' / 'all' / 'auto' will resample all classes but the majority.
# To make all classes have the same number of samples as the majority class:
smote = SMOTE(random_state=42, sampling_strategy='auto') # 'auto' resamples all classes but the majority.
                                                       # To make all classes equal to majority:
                                                       # Determine majority class size:
                                                       # counts = Counter(y_train)
                                                       # majority_size = max(counts.values())
                                                       # strategy = {label: majority_size for label in counts.keys() if counts[label] < majority_size}
                                                       # if strategy: smote = SMOTE(random_state=42, sampling_strategy=strategy)

try:
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("✅ SMOTE applied successfully to the training data.")

    # Display class distribution AFTER SMOTE on the training set
    print("\nClass distribution in SMOTE-resampled training data (y_train_smote):")
    print(Counter(le.inverse_transform(y_train_smote))) # Show original labels
    print(f"  SMOTE Training set shape: X_train_smote={X_train_smote.shape}, y_train_smote={y_train_smote.shape}")

    # Now, X_train_smote and y_train_smote are ready to be used for training your model.
    # X_test and y_test remain unchanged and will be used for evaluation.

except Exception as e:
    print(f"❌ Error applying SMOTE: {e}")
    print("   This can happen if a class has too few samples (e.g., less than k_neighbors for SMOTE).")
    print("   Current k_neighbors for SMOTE is likely 5 (default). Check class counts in y_train.")
    exit()
    # (Assuming the previous parts of your SMOTE script have been executed, and you have:
    # X_train_smote, y_train_smote, X_test, y_test,
    # X_train (DataFrame before SMOTE, after OHE and scaling - to get column names),
    # le (the fitted LabelEncoder for your target variable),
    # TARGET_COLUMN (the name of your target variable string)
    # )

print("\n💾 Saving SMOTE-processed training and test datasets...")

# --- Prepare and Save Balanced Training Data ---
try:
    # Convert X_train_smote (likely a NumPy array from SMOTE) back to a DataFrame
    # Use columns from X_train (which has undergone OHE and scaling)
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=X_train.columns)

    # Convert y_train_smote (encoded labels) back to original string labels
    y_train_smote_original_labels = pd.Series(le.inverse_transform(y_train_smote), name=TARGET_COLUMN)

    # Combine features and target for the balanced training set
    df_train_balanced = pd.concat([X_train_smote_df.reset_index(drop=True),
                                   y_train_smote_original_labels.reset_index(drop=True)], axis=1)

    train_output_filename = "../data/smote_balanced_training_data.csv"
    df_train_balanced.to_csv(train_output_filename, index=False)
    print(f"✅ SMOTE-balanced training data saved to '{train_output_filename}'. Shape: {df_train_balanced.shape}")
    print("   Sample of saved balanced training data:")
    print(df_train_balanced.head(3))

except Exception as e:
    print(f"❌ Error saving SMOTE-balanced training data: {e}")

# --- Prepare and Save Processed Test Data ---
try:
    # Convert X_test (likely a NumPy array after scaling) back to a DataFrame
    # Use columns from X_train (as X_test was transformed using the same scaler and has same features)
    X_test_df = pd.DataFrame(X_test, columns=X_train.columns)

    # Convert y_test (encoded labels) back to original string labels
    y_test_original_labels = pd.Series(le.inverse_transform(y_test), name=TARGET_COLUMN)

    # Combine features and target for the processed test set
    df_test_processed = pd.concat([X_test_df.reset_index(drop=True),
                                   y_test_original_labels.reset_index(drop=True)], axis=1)

    test_output_filename = "../data/processed_test_data.csv"
    df_test_processed.to_csv(test_output_filename, index=False)
    print(f"\n✅ Processed test data saved to '{test_output_filename}'. Shape: {df_test_processed.shape}")
    print("   Sample of saved processed test data:")
    print(df_test_processed.head(3))

except Exception as e:
    print(f"❌ Error saving processed test data: {e}")

print("\n🎉 Data saving process complete.")

# --- Next Steps would be Model Training ---
# model.fit(X_train_smote, y_train_smote)
# predictions = model.predict(X_test)
# evaluate(predictions, y_test)

