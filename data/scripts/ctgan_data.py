import pandas as pd
import numpy as np # Import numpy for np.nan
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler

# Define column names (CHANGE IF YOURS ARE DIFFERENT)
BREATH_ACETONE_COLUMN = 'Breath_Acetone_ppm'
BHB_COLUMN = 'β-Hydroxybutyrate_mmol/L' # New column for Beta-Hydroxybutyrate

# --- 1. Load and Preprocess Your Original Data ---
try:
    df_real = pd.read_csv("../data/augmented_glucose_data_noise_added_cleaned.csv")
except FileNotFoundError:
    print("Error: 'augmented_glucose_data.csv' not found. Please ensure the file exists.")
    exit()

print("✅ Original Data Loaded. Shape:", df_real.shape)
# print("Data types before processing:\n", df_real.dtypes) # Can be verbose

# --- Initial NaN Check (Before any processing) ---
# print("\nInitial NaN values in df_real:")
# print(df_real.isnull().sum())

# --- Data Preprocessing for df_real ---
# Fill missing HbA1c values
if 'HbA1c_%' in df_real.columns:
    if df_real['HbA1c_%'].isnull().any():
        hba1c_mean = df_real['HbA1c_%'].mean()
        df_real['HbA1c_%'] = df_real['HbA1c_%'].fillna(hba1c_mean)
        print(f"\nFilled NaNs in 'HbA1c_%' with mean: {hba1c_mean:.2f}")

# Handle negative Breath Acetone values by setting them to NaN
if BREATH_ACETONE_COLUMN in df_real.columns:
    negative_acetone_count = (df_real[BREATH_ACETONE_COLUMN] < 0).sum()
    if negative_acetone_count > 0:
        df_real.loc[df_real[BREATH_ACETONE_COLUMN] < 0, BREATH_ACETONE_COLUMN] = np.nan
        # print(f"\nConverted {negative_acetone_count} negative '{BREATH_ACETONE_COLUMN}' values to NaN in df_real.")
    if df_real[BREATH_ACETONE_COLUMN].isnull().any():
        acetone_median = df_real[BREATH_ACETONE_COLUMN].median()
        df_real[BREATH_ACETONE_COLUMN] = df_real[BREATH_ACETONE_COLUMN].fillna(acetone_median)
        print(f"Filled NaNs in '{BREATH_ACETONE_COLUMN}' with median: {acetone_median:.2f}")
# else:
    # print(f"\nWarning: Column '{BREATH_ACETONE_COLUMN}' not found in df_real.")

# Handle negative β-Hydroxybutyrate_mmol/L values by setting them to NaN
if BHB_COLUMN in df_real.columns:
    negative_bhb_count = (df_real[BHB_COLUMN] < 0).sum()
    if negative_bhb_count > 0:
        df_real.loc[df_real[BHB_COLUMN] < 0, BHB_COLUMN] = np.nan
        # print(f"\nConverted {negative_bhb_count} negative '{BHB_COLUMN}' values to NaN in df_real.")
    if df_real[BHB_COLUMN].isnull().any():
        bhb_median = df_real[BHB_COLUMN].median()
        df_real[BHB_COLUMN] = df_real[BHB_COLUMN].fillna(bhb_median)
        print(f"Filled NaNs in '{BHB_COLUMN}' with median: {bhb_median:.2f}")
# else:
    # print(f"\nWarning: Column '{BHB_COLUMN}' not found in df_real.")


if 'Blood_Glucose_mg/dL' not in df_real.columns:
    print("Error: 'Blood_Glucose_mg/dL' column is required to create 'Glucose_Level_Class' but not found.")
    exit()

def classify_glucose_level(value):
    if pd.isna(value):
        return 'Undefined' # Or handle as appropriate for your case
    if value < 70:
        return 'Low'
    elif 70 <= value <= 139:
        return 'Normal'
    else:
        return 'High'

df_real['Glucose_Level_Class'] = df_real['Blood_Glucose_mg/dL'].apply(classify_glucose_level)

print("\n📊 Overall Missing Values in df_real (after initial preprocessing):")
missing_values_real = df_real.isnull().sum()
missing_report_real = pd.DataFrame({'Missing Values': missing_values_real})
print(missing_report_real[missing_report_real['Missing Values'] > 0])
if missing_report_real['Missing Values'].sum() == 0:
    print("No missing values in df_real after initial preprocessing.")

print("\n✅ Original Data Sample (after initial preprocessing):")
print(df_real.head(3))
print("\nDistribution of Glucose Levels in Original Data:")
print(df_real['Glucose_Level_Class'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# --- 2. Identify Discrete (Categorical) and Numerical Columns ---
discrete_columns = ['Glucose_Level_Class']
if 'Gender' in df_real.columns: # Example, if you have a Gender column
    discrete_columns.append('Gender')
discrete_columns = [col for col in discrete_columns if col in df_real.columns]
print(f"\nDiscrete columns for CTGAN: {discrete_columns}")

# Identify numerical columns for normalization (excluding IDs or already discrete columns)
# Make sure to list all columns that are numeric and should be scaled
numerical_cols_to_scale = [
    'Age', 'BMI', 'Blood_Glucose_mg/dL', 'HbA1c_%',
    BREATH_ACETONE_COLUMN, BHB_COLUMN,
    'Temp_C', 'Humidity_%', 'Fasting_Hours'
]
# Filter to only include columns that actually exist in df_real
numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in df_real.columns]
print(f"Numerical columns to be scaled: {numerical_cols_to_scale}")


# --- PREPARE DATA FOR CTGAN TRAINING ---
df_real_for_training = df_real.copy()

# Drop rows with any remaining NaNs before normalization and training
# This is crucial as scalers and CTGAN cannot handle NaNs in the training data
print(f"\nShape of data before dropping NaNs for Normalization/CTGAN training: {df_real_for_training.shape}")
df_real_for_training.dropna(subset=numerical_cols_to_scale + discrete_columns, inplace=True)
print(f"Shape of data after dropping NaNs for Normalization/CTGAN training: {df_real_for_training.shape}")

if df_real_for_training.empty:
    print("Error: DataFrame is empty after dropping NaNs. CTGAN cannot be trained. Please check data quality or imputation strategy.")
    exit()

# --- NEW: Normalize Numerical Features in Real Data for Training ---
scalers = {} # To store scalers for inverse transformation later
df_normalized_for_training = df_real_for_training.copy()

if numerical_cols_to_scale: # Proceed only if there are numerical columns to scale
    print("\n🔄 Normalizing numerical features for CTGAN training...")
    for col in numerical_cols_to_scale:
        if col in df_normalized_for_training.columns: # Ensure column exists
            scaler = MinMaxScaler()
            df_normalized_for_training[col] = scaler.fit_transform(df_normalized_for_training[[col]])
            scalers[col] = scaler # Store the scaler
            print(f"Normalized '{col}'. Min: {df_normalized_for_training[col].min():.2f}, Max: {df_normalized_for_training[col].max():.2f}")
    print("✅ Normalization complete for training data.")
    print("Sample of normalized data for training:")
    print(df_normalized_for_training[numerical_cols_to_scale].head(3))
else:
    print("\nNo numerical columns specified or found for scaling.")


# --- 3. Initialize and Train the CTGAN Synthesizer ---
# CTGAN will be trained on the df_normalized_for_training
ctgan_model = CTGAN(epochs=500, verbose=True) # Using 500 epochs
print("\n🚀 Starting CTGAN model training on NORMALIZED data...")
try:
    # Ensure discrete_columns are present in df_normalized_for_training
    valid_discrete_columns = [col for col in discrete_columns if col in df_normalized_for_training.columns]
    ctgan_model.fit(df_normalized_for_training, valid_discrete_columns)
    print("✅ CTGAN model training complete.")
except Exception as e:
    print(f"❌ Error during CTGAN training: {e}")
    exit()

# --- 4. Generate Synthetic Data CONDITIONALLY ---
# The synthetic data will be generated in the NORMALIZED scale
n_total_synthetic = 900 # Target total, actual might be slightly different due to rounding
# Adjust proportions based on your desired target distribution or real data distribution
# Example: trying to match real data distribution if df_real_for_training is representative
if not df_real_for_training.empty and 'Glucose_Level_Class' in df_real_for_training.columns:
    class_proportions = df_real_for_training['Glucose_Level_Class'].value_counts(normalize=True)
    n_samples_low = int(n_total_synthetic * class_proportions.get('Low', 0))
    n_samples_normal = int(n_total_synthetic * class_proportions.get('Normal', 0))
    n_samples_high = int(n_total_synthetic * class_proportions.get('High', 0))
    # Adjust for any 'Undefined' or other classes if they exist and you want to sample them
else: # Fallback if proportions can't be determined
    n_samples_low = int(n_total_synthetic * 0.33)
    n_samples_normal = int(n_total_synthetic * 0.34)
    n_samples_high = int(n_total_synthetic * 0.33)


print(f"\n🧪 Generating {n_samples_low} 'Low', {n_samples_normal} 'Normal', {n_samples_high} 'High' synthetic samples (normalized scale)...")
try:
    synthetic_low_normalized = ctgan_model.sample(n_samples_low, condition_column='Glucose_Level_Class', condition_value='Low')
    synthetic_normal_normalized = ctgan_model.sample(n_samples_normal, condition_column='Glucose_Level_Class', condition_value='Normal')
    synthetic_high_normalized = ctgan_model.sample(n_samples_high, condition_column='Glucose_Level_Class', condition_value='High')
    df_synthetic_normalized = pd.concat([synthetic_low_normalized, synthetic_normal_normalized, synthetic_high_normalized], ignore_index=True)
    print("✅ Conditional synthetic data generated (in normalized scale).")
    print("Sample of raw synthetic data (normalized):")
    print(df_synthetic_normalized.head(3))
except Exception as e:
    print(f"Error during conditional sampling: {e}")
    exit()

# --- NEW: Inverse Transform Synthetic Data to Original Scale ---
df_synthetic_conditional = df_synthetic_normalized.copy()
if scalers and not df_synthetic_conditional.empty: # Proceed only if scalers exist and data is not empty
    print("\n🔄 Inverse transforming synthetic data to original scale...")
    for col, scaler_obj in scalers.items():
        if col in df_synthetic_conditional.columns: # Ensure column exists in synthetic data
            df_synthetic_conditional[col] = scaler_obj.inverse_transform(df_synthetic_conditional[[col]])
            print(f"Inverse transformed '{col}'. Example val: {df_synthetic_conditional[col].iloc[0]:.2f} (if exists)")
    print("✅ Inverse transformation complete.")
    print("Sample of synthetic data (original scale):")
    print(df_synthetic_conditional.head(3))
elif df_synthetic_conditional.empty:
    print("Warning: Synthetic data is empty, skipping inverse transform.")
else:
    print("\nNo scalers available or numerical columns were scaled. Skipping inverse transform.")


# --- 5. Clean the Synthetic Data (Post-processing and Clipping on ORIGINAL SCALE data) ---
print("\n🧼 Cleaning and clipping synthetic data (original scale)...")
if not df_synthetic_conditional.empty:
    if 'Age' in df_synthetic_conditional.columns:
        df_synthetic_conditional['Age'] = df_synthetic_conditional['Age'].clip(lower=18, upper=80) # Adjusted upper bound
    if 'Blood_Glucose_mg/dL' in df_synthetic_conditional.columns:
        df_synthetic_conditional['Blood_Glucose_mg/dL'] = df_synthetic_conditional['Blood_Glucose_mg/dL'].clip(lower=20, upper=700) # Adjusted bounds
    if 'HbA1c_%' in df_synthetic_conditional.columns:
        df_synthetic_conditional['HbA1c_%'] = df_synthetic_conditional['HbA1c_%'].clip(lower=3.0, upper=20.0) # Adjusted bounds
    if 'BMI' in df_synthetic_conditional.columns:
        df_synthetic_conditional['BMI'] = df_synthetic_conditional['BMI'].clip(lower=15, upper=50) # Adjusted bounds

    # Handle negative Breath Acetone in SYNTHETIC data
    if BREATH_ACETONE_COLUMN in df_synthetic_conditional.columns:
        neg_synth_acetone = (df_synthetic_conditional[BREATH_ACETONE_COLUMN] < 0).sum()
        if neg_synth_acetone > 0:
            # Option 1: Set to a small positive floor (e.g., 0 or 0.001) if NaNs are problematic later
            df_synthetic_conditional.loc[df_synthetic_conditional[BREATH_ACETONE_COLUMN] < 0, BREATH_ACETONE_COLUMN] = 0.0
            # Option 2: Set to NaN and then impute if preferred
            # df_synthetic_conditional.loc[df_synthetic_conditional[BREATH_ACETONE_COLUMN] < 0, BREATH_ACETONE_COLUMN] = np.nan
            print(f"Corrected {neg_synth_acetone} negative '{BREATH_ACETONE_COLUMN}' values in synthetic data (set to 0.0).")

    # Handle negative β-Hydroxybutyrate in SYNTHETIC data
    if BHB_COLUMN in df_synthetic_conditional.columns:
        neg_synth_bhb = (df_synthetic_conditional[BHB_COLUMN] < 0).sum()
        if neg_synth_bhb > 0:
            df_synthetic_conditional.loc[df_synthetic_conditional[BHB_COLUMN] < 0, BHB_COLUMN] = 0.0
            print(f"Corrected {neg_synth_bhb} negative '{BHB_COLUMN}' values in synthetic data (set to 0.0).")
else:
    print("Warning: Synthetic data is empty, skipping cleaning.")


# --- 6. Reclassify BGL and Compare to CTGAN Condition ---
if not df_synthetic_conditional.empty and 'Blood_Glucose_mg/dL' in df_synthetic_conditional.columns:
    df_synthetic_conditional['Predicted_Class_From_BGL'] = df_synthetic_conditional['Blood_Glucose_mg/dL'].apply(classify_glucose_level)
    if 'Glucose_Level_Class' in df_synthetic_conditional.columns: # This is the conditioned column
        df_synthetic_conditional['Condition_Match'] = df_synthetic_conditional['Glucose_Level_Class'] == df_synthetic_conditional['Predicted_Class_From_BGL']
    else:
        df_synthetic_conditional['Condition_Match'] = False # Should not happen if conditioned
else:
    df_synthetic_conditional['Condition_Match'] = False # Default if BGL column is missing

# --- 7. Report Mismatch ---
if not df_synthetic_conditional.empty and 'Condition_Match' in df_synthetic_conditional.columns:
    print("\n🔍 Conditional Generation Match Analysis (Generated BGL vs. Conditioned Class):")
    print(df_synthetic_conditional['Condition_Match'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    mismatches = df_synthetic_conditional[~df_synthetic_conditional['Condition_Match']]
    if not mismatches.empty:
        print(f"\nFound {len(mismatches)} mismatches. Examples:")
        print(mismatches[['Blood_Glucose_mg/dL', 'Glucose_Level_Class', 'Predicted_Class_From_BGL']].head())
    else:
        print("No mismatches found between conditioned class and re-classified BGL after cleaning.")

# --- 8. Keep Only Matching Samples (Strict Mode) ---
# This step ensures that the BGL values, after generation and clipping, still fall into the
# category that was originally conditioned on.
if not df_synthetic_conditional.empty and 'Condition_Match' in df_synthetic_conditional.columns:
    df_cleaned_strict = df_synthetic_conditional[df_synthetic_conditional['Condition_Match']].copy()
    print(f"\nApplied strict matching: Kept {len(df_cleaned_strict)} out of {len(df_synthetic_conditional)} synthetic samples.")
else:
    df_cleaned_strict = df_synthetic_conditional.copy() # Or an empty DataFrame if source is empty
    print("\nSkipping strict matching or source data was empty.")

columns_to_drop_from_final = ['Condition_Match', 'Predicted_Class_From_BGL']
df_cleaned_strict.drop(columns=[col for col in columns_to_drop_from_final if col in df_cleaned_strict.columns], inplace=True, errors='ignore')


# --- 9. Apply Column-Specific Formatting for Output ---
print("\n⚙️ Formatting specific columns for CSV output...")
if not df_cleaned_strict.empty:
    if 'Age' in df_cleaned_strict.columns:
        df_cleaned_strict['Age'] = df_cleaned_strict['Age'].round(0).astype(int)
    if 'Fasting_Hours' in df_cleaned_strict.columns and pd.api.types.is_numeric_dtype(df_cleaned_strict['Fasting_Hours']):
        df_cleaned_strict['Fasting_Hours'] = df_cleaned_strict['Fasting_Hours'].round(0).astype(int)
    if 'Temp_C' in df_cleaned_strict.columns and pd.api.types.is_numeric_dtype(df_cleaned_strict['Temp_C']):
        df_cleaned_strict['Temp_C'] = df_cleaned_strict['Temp_C'].round(1) # Temp often to 1 decimal
    if 'Humidity_%' in df_cleaned_strict.columns and pd.api.types.is_numeric_dtype(df_cleaned_strict['Humidity_%']):
        df_cleaned_strict['Humidity_%'] = df_cleaned_strict['Humidity_%'].round(0).astype(int)
    if 'BMI' in df_cleaned_strict.columns:
        df_cleaned_strict['BMI'] = df_cleaned_strict['BMI'].round(1)
    if 'Blood_Glucose_mg/dL' in df_cleaned_strict.columns:
        df_cleaned_strict['Blood_Glucose_mg/dL'] = df_cleaned_strict['Blood_Glucose_mg/dL'].round(1) # Often BGL is to 1 decimal or whole number
    if 'HbA1c_%' in df_cleaned_strict.columns:
        df_cleaned_strict['HbA1c_%'] = df_cleaned_strict['HbA1c_%'].round(1) # HbA1c often to 1 decimal
    if BREATH_ACETONE_COLUMN in df_cleaned_strict.columns:
        df_cleaned_strict[BREATH_ACETONE_COLUMN] = df_cleaned_strict[BREATH_ACETONE_COLUMN].round(3)
    if BHB_COLUMN in df_cleaned_strict.columns:
        df_cleaned_strict[BHB_COLUMN] = df_cleaned_strict[BHB_COLUMN].round(3)
    print("✅ Formatting complete.")
else:
    print("Warning: df_cleaned_strict is empty, skipping formatting.")

# --- Overall Missing Values Report for df_cleaned_strict (BEFORE Saving) ---
print("\n📊 Overall Missing Values in df_cleaned_strict (after all processing, before saving):")
if not df_cleaned_strict.empty:
    missing_values_synthetic = df_cleaned_strict.isnull().sum()
    missing_report_synthetic = pd.DataFrame({'Missing Values': missing_values_synthetic})
    print(missing_report_synthetic[missing_report_synthetic['Missing Values'] > 0])
    if missing_report_synthetic['Missing Values'].sum() == 0:
        print("No missing values in final df_cleaned_strict.")
else:
    print("Warning: df_cleaned_strict is empty, cannot report missing values.")

# --- 10. Save the Cleaned, Verified, and Formatted Synthetic Data ---
output_filename = "../data/synthetic_data_ctgan_norm_conditional_verified_formatted.csv"
try:
    if not df_cleaned_strict.empty:
        df_cleaned_strict.to_csv(output_filename, index=False)
        print(f"\n✅ Strictly verified, formatted, and normalized-then-inversed synthetic data saved to '{output_filename}'")
    else:
        print(f"\n⚠️ No data in df_cleaned_strict to save to '{output_filename}'. All samples may have been filtered out.")
except Exception as e:
    print(f"\n❌ Error saving synthetic data: {e}")

# --- 11. Final Report ---
print("NaNs in normalized synthetic acetone:", df_synthetic_normalized[BREATH_ACETONE_COLUMN].isnull().sum())
print("Infinities in normalized synthetic acetone:", np.isinf(df_synthetic_normalized[BREATH_ACETONE_COLUMN]).sum())
print("Describe normalized synthetic acetone:")
print(df_synthetic_normalized[BREATH_ACETONE_COLUMN].describe())
print("\n🧾 Final Verified and Formatted Synthetic Data Sample:")
print(df_cleaned_strict.head())

if not df_cleaned_strict.empty and 'Glucose_Level_Class' in df_cleaned_strict.columns:
    print("\n📊 Class Distribution (Final Verified Synthetic):")
    print(df_cleaned_strict['Glucose_Level_Class'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print(f"\n📁 Final Verified Synthetic Data Shape: {df_cleaned_strict.shape}")
if not df_real_for_training.empty:
    print(f"Original data shape for CTGAN training (after dropna): {df_real_for_training.shape}")
    if df_real_for_training.shape[0] > 0 and not df_cleaned_strict.empty:
        print(f"\n Ratio of final verified synthetic samples to original training samples: {df_cleaned_strict.shape[0] / df_real_for_training.shape[0]:.2f}")

print("\n🎉 Script finished.")