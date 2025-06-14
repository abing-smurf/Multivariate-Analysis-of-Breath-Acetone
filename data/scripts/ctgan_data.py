import pandas as pd
import numpy as np
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Your Original Data ---

# IMPORTANT: Replace with the name of your input file.
input_filename = "your_dataset.csv"

try:
    df_real = pd.read_csv(input_filename)
    # If 'Height' and 'Weight' columns exist, drop them.
    df_real.drop(columns=['Height', 'Weight'], inplace=True, errors='ignore')

except FileNotFoundError:
    print(f"Error: '{input_filename}' not found. Please ensure the file exists in the correct directory.")
    # As a fallback for demonstration, create a dummy dataframe
    print("Creating a dummy dataframe for demonstration purposes...")
    data = {
        'Age': [55, 62, 48, 70, 35],
        'Blood Pressure': ['120/80', '140/90', '110/70', '150/95', '115/75'],
        'BMI': [25.7, 29.4, 24.0, 29.3, 22.0],
        'Pulse rate': [72, 80, 68, 85, 75],
        'HbA1c_%': [5.7, 6.5, 5.2, 7.8, 5.5],
        'Respiratory Rate': [16, 18, 15, 20, 17],
        'Acetone PPM 1.1': [1.2, 1.8, 0.9, 2.5, 1.1],
        'Temperature': [36.5, 36.8, 36.4, 37.0, 36.6],
        'Humidity': [50, 55, 48, 60, 52],
        'Glucose_classification_class': [0, 1, 0, 2, 0]  # Example: 0=Normal, 1=Prediabetic, 2=Diabetic
    }
    df_real = pd.DataFrame(data)

print(f"✅ Original Data Loaded. Shape: {df_real.shape}")
print("Original Data Sample:")
print(df_real.head(3))

# --- 2. Preprocess Data ---
# This section handles missing values and corrects invalid data before training.

# NEW: Parse Blood Pressure column
if 'Blood Pressure' in df_real.columns:
    print("\nProcessing 'Blood Pressure' column...")
    # Split the column into two new ones, Systolic and Diastolic
    bp_split = df_real['Blood Pressure'].str.split('/', expand=True)

    # Assign to new columns and convert to numeric, coercing errors to NaN
    df_real['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
    df_real['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')

    # Drop the original 'Blood Pressure' column
    df_real.drop('Blood Pressure', axis=1, inplace=True)
    print("✅ 'Blood Pressure' split into 'Systolic_BP' and 'Diastolic_BP'.")

# Fill missing HbA1c values using the mean
if 'HbA1c_%' in df_real.columns and df_real['HbA1c_%'].isnull().any():
    hba1c_mean = df_real['HbA1c_%'].mean()
    df_real['HbA1c_%'].fillna(hba1c_mean, inplace=True)
    print(f"Filled NaNs in 'HbA1c_%' with mean: {hba1c_mean:.2f}")

# Handle potential negative 'Acetone PPM 1.1' values by setting to NaN, then filling with median
if 'Acetone PPM 1.1' in df_real.columns:
    negative_acetone_count = (df_real['Acetone PPM 1.1'] < 0).sum()
    if negative_acetone_count > 0:
        df_real.loc[df_real['Acetone PPM 1.1'] < 0, 'Acetone PPM 1.1'] = np.nan
        print(f"Converted {negative_acetone_count} negative 'Acetone PPM 1.1' values to NaN.")

    if df_real['Acetone PPM 1.1'].isnull().any():
        acetone_median = df_real['Acetone PPM 1.1'].median()
        df_real['Acetone PPM 1.1'].fillna(acetone_median, inplace=True)
        print(f"Filled NaNs in 'Acetone PPM 1.1' with median: {acetone_median:.2f}")

# Check for any other missing values and report
print("\n📊 Missing Values Report (after preprocessing):")
if df_real.isnull().sum().sum() == 0:
    print("No missing values found in the dataset.")
else:
    print(df_real.isnull().sum())

# --- 3. Identify Discrete (Categorical) and Numerical Columns ---
discrete_columns = ['Glucose_classification_class']
print(f"\nDiscrete columns for CTGAN: {discrete_columns}")

# UPDATED: 'Height' and 'Weight' have been removed from this list.
numerical_cols_to_scale = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Pulse rate',
    'HbA1c_%', 'Respiratory Rate', 'Acetone PPM 1.1', 'Temperature', 'Humidity'
]
# Filter to only include columns that actually exist in the dataframe
numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in df_real.columns]
print(f"Numerical columns to be scaled: {numerical_cols_to_scale}")

# --- 4. Prepare Data for CTGAN Training ---
df_real_for_training = df_real.copy()

# Drop any rows with remaining NaNs before training to prevent errors.
print(f"\nShape before dropping NaNs for training: {df_real_for_training.shape}")
df_real_for_training.dropna(subset=numerical_cols_to_scale + discrete_columns, inplace=True)
print(f"Shape after dropping NaNs for training: {df_real_for_training.shape}")

if df_real_for_training.empty:
    print("Error: DataFrame is empty after dropping NaNs. CTGAN cannot be trained.")
    exit()

# --- 5. Normalize Numerical Features ---
scalers = {}  # Dictionary to store scalers for each column for later reuse
df_normalized_for_training = df_real_for_training.copy()

print("\n🔄 Normalizing numerical features for CTGAN training...")
for col in numerical_cols_to_scale:
    scaler = MinMaxScaler()
    df_normalized_for_training[col] = scaler.fit_transform(df_normalized_for_training[[col]])
    scalers[col] = scaler  # Store the fitted scaler
print("✅ Normalization complete for training data.")

# --- 6. Initialize and Train the CTGAN Synthesizer ---
ctgan_model = CTGAN(epochs=500, verbose=True)
print("\n🚀 Starting CTGAN model training on NORMALIZED data...")
try:
    ctgan_model.fit(df_normalized_for_training, discrete_columns)
    print("✅ CTGAN model training complete.")
except Exception as e:
    print(f"❌ Error during CTGAN training: {e}")
    exit()

# --- 7. Generate Synthetic Data CONDITIONALLY ---
num_samples_per_class = 500
class_labels = df_real_for_training['Glucose_classification_class'].unique()

print(f"\n🧪 Generating {num_samples_per_class} samples for each class: {class_labels}...")

all_synthetic_data = []
try:
    for label in class_labels:
        synthetic_class_normalized = ctgan_model.sample(
            num_samples_per_class,
            condition_column='Glucose_classification_class',
            condition_value=label
        )
        all_synthetic_data.append(synthetic_class_normalized)

    df_synthetic_normalized = pd.concat(all_synthetic_data, ignore_index=True)
    print("✅ Conditional synthetic data generated (in normalized scale).")
except Exception as e:
    print(f"Error during conditional sampling: {e}")
    exit()

# --- 8. Inverse Transform Synthetic Data to Original Scale ---
df_synthetic_original_scale = df_synthetic_normalized.copy()
print("\n🔄 Inverse transforming synthetic data to original scale...")
for col, scaler_obj in scalers.items():
    if col in df_synthetic_original_scale.columns:
        df_synthetic_original_scale[col] = scaler_obj.inverse_transform(df_synthetic_original_scale[[col]])
print("✅ Inverse transformation complete.")
print("Sample of synthetic data (original scale):")
print(df_synthetic_original_scale.head(3))

# --- 9. Clean and Format the Final Synthetic Data ---
print("\n🧼 Cleaning and formatting synthetic data...")
df_final_synthetic = df_synthetic_original_scale.copy()

# Clip values to realistic ranges
df_final_synthetic['Age'] = df_final_synthetic['Age'].clip(lower=18, upper=100)
df_final_synthetic['Systolic_BP'] = df_final_synthetic['Systolic_BP'].clip(lower=80, upper=200)
df_final_synthetic['Diastolic_BP'] = df_final_synthetic['Diastolic_BP'].clip(lower=50, upper=120)
df_final_synthetic['BMI'] = df_final_synthetic['BMI'].clip(lower=15, upper=50)
df_final_synthetic['Pulse rate'] = df_final_synthetic['Pulse rate'].clip(lower=40, upper=160)
df_final_synthetic['HbA1c_%'] = df_final_synthetic['HbA1c_%'].clip(lower=3.0, upper=20.0)
df_final_synthetic['Respiratory Rate'] = df_final_synthetic['Respiratory Rate'].clip(lower=10, upper=30)
df_final_synthetic['Acetone PPM 1.1'] = df_final_synthetic['Acetone PPM 1.1'].clip(lower=0)

# Apply formatting for cleaner output
df_final_synthetic['Age'] = df_final_synthetic['Age'].round(0).astype(int)
df_final_synthetic['Systolic_BP'] = df_final_synthetic['Systolic_BP'].round(0).astype(int)
df_final_synthetic['Diastolic_BP'] = df_final_synthetic['Diastolic_BP'].round(0).astype(int)
df_final_synthetic['BMI'] = df_final_synthetic['BMI'].round(1)
df_final_synthetic['Pulse rate'] = df_final_synthetic['Pulse rate'].round(0).astype(int)
df_final_synthetic['Respiratory Rate'] = df_final_synthetic['Respiratory Rate'].round(0).astype(int)
df_final_synthetic['Temperature'] = df_final_synthetic['Temperature'].round(1)
df_final_synthetic['Humidity'] = df_final_synthetic['Humidity'].round(1)
df_final_synthetic['HbA1c_%'] = df_final_synthetic['HbA1c_%'].round(1)
df_final_synthetic['Acetone PPM 1.1'] = df_final_synthetic['Acetone PPM 1.1'].round(3)

print("✅ Formatting complete.")

# --- 10. Save the Final Synthetic Data ---
output_filename = "../data/synthetic_data_final.csv"
try:
    df_final_synthetic.to_csv(output_filename, index=False)
    print(f"\n✅ Final synthetic data saved to '{output_filename}'")
except Exception as e:
    print(f"\n❌ Error saving final synthetic data: {e}")

# --- 11. Final Report ---
print("\n🧾 Final Synthetic Data Sample:")
print(df_final_synthetic.head())

print("\n📊 Class Distribution (Final Synthetic Data):")
print(
    df_final_synthetic['Glucose_classification_class'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print(f"\n📁 Final Synthetic Data Shape: {df_final_synthetic.shape}")
print("\n🎉 Script finished.")
