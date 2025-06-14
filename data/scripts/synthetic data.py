# ==============================================================================
# Thesis Assistant: Data Augmentation via Noise Addition
# ==============================================================================
# Description:
# This script augments a small dataset by creating new samples through noise
# addition. It takes the original data, resamples it, adds a small, random
# gaussian noise to the numerical features, and then cleans the resulting
# data by clipping it to realistic bounds. This is a simpler alternative to
# CTGAN, more suitable for very small datasets.
#
# **Version 3 Update**: Fixed a bug where Glucose_classification_class was not
# re-calculated after noise was added to Blood_Glucose_mg/dL. This ensures
# data consistency and prevents downstream plotting errors.
#
# Instructions:
# 1. Install required libraries: pip install pandas numpy
# 2. Place your original dataset in an accessible path.
# 3. Update the `ORIGINAL_DATA_FILE` and `AUGMENTED_DATA_OUTPUT_FILE` variables.
# 4. Run the script. A new augmented CSV file will be created.
# ==============================================================================

import pandas as pd
import numpy as np

# --- Configuration ---
ORIGINAL_DATA_FILE = "../data/raw_data.csv"
AUGMENTED_DATA_OUTPUT_FILE = "../data/augmented_data_by_noise.csv"

# Parameters for Synthetic Data Generation
N_SYNTHETIC_SAMPLES = 500  # Number of synthetic samples to generate
NOISE_SCALE_FACTOR = 0.05  # Factor for noise level (e.g., 0.05 = 5% of a feature's standard deviation)

# List of numeric columns to apply noise to
# The script will only use the columns from this list that it finds in your file.
NUMERIC_COLS_FOR_NOISE = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Pulse rate',
    'HbA1c_%', 'Respiratory Rate', 'Acetone PPM 1.1', 'Temperature', 'Humidity'
]

# Define realistic bounds for cleaning synthetic numerical features {column: (min, max)}
FEATURE_BOUNDS = {
    'Age': (18, 90),
    'Systolic_BP': (80, 200),
    'Diastolic_BP': (50, 120),
    'BMI': (15, 50),
    'Pulse rate': (40, 160),
    'HbA1c_%': (3.0, 20.0),
    'Respiratory Rate': (10, 30),
    'Acetone PPM 1.1': (0.0, 50.0),  # Assuming a max of 50 ppm for breath acetone
    'Temperature': (35.0, 42.0),
    'Humidity': (0, 100)
}


# --- Helper Functions ---

def classify_glucose(value):
    """Classifies glucose value into Low, Normal, High, or Undefined."""
    if pd.isna(value):
        return 'Undefined'
    if value < 70:
        return 'Low'
    elif 70 <= value <= 139:
        return 'Normal'
    else:  # > 139
        return 'High'


def load_and_prepare_data(file_path):
    """Loads and prepares the initial data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully from '{file_path}'. Shape: {df.shape}")

        # Omit Height and Weight if they exist
        df.drop(columns=['Height', 'Weight'], inplace=True, errors='ignore')

        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        # Create a dummy dataframe for demonstration
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
            'Glucose_classification_class': [0, 1, 0, 2, 0]  # This will be recalculated anyway
        }
        return pd.DataFrame(data)
    except Exception as e:
        print(f"❌ Error loading file '{file_path}': {e}")
        exit()


def preprocess_data(df):
    """Preprocesses the dataset: handles Blood Pressure, fills NaNs, and classifies glucose."""
    print("\n⚙️  Preprocessing data...")
    df_processed = df.copy()

    # --- Handle Blood Pressure ---
    if 'Blood Pressure' in df_processed.columns:
        print("  Parsing 'Blood Pressure' into 'Systolic_BP' and 'Diastolic_BP'.")
        bp_split = df_processed['Blood Pressure'].str.split('/', expand=True)
        df_processed['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
        df_processed['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
        df_processed.drop(columns=['Blood Pressure'], inplace=True)

    # --- Fill Missing Values ---
    if 'HbA1c_%' in df_processed.columns and df_processed['HbA1c_%'].isnull().any():
        mean_hba1c = df_processed['HbA1c_%'].mean()
        df_processed['HbA1c_%'].fillna(mean_hba1c, inplace=True)
        print(f"  Filled NaNs in 'HbA1c_%' with mean: {mean_hba1c:.2f}")

    if 'Acetone PPM 1.1' in df_processed.columns:
        neg_acetone_count = (df_processed['Acetone PPM 1.1'] < 0).sum()
        if neg_acetone_count > 0:
            df_processed.loc[df_processed['Acetone PPM 1.1'] < 0, 'Acetone PPM 1.1'] = np.nan
        if df_processed['Acetone PPM 1.1'].isnull().any():
            median_acetone = df_processed['Acetone PPM 1.1'].median()
            df_processed['Acetone PPM 1.1'].fillna(median_acetone, inplace=True)
            print(f"  Handled negative/missing values in 'Acetone PPM 1.1'.")

    # Check which of the target columns actually exist in the dataframe.
    existing_cols_to_check = [col for col in NUMERIC_COLS_FOR_NOISE if col in df_processed.columns]

    # Drop any rows with NaN values in the columns we'll use for modeling
    df_processed.dropna(subset=existing_cols_to_check, inplace=True)

    print("✅ Data preprocessing complete.")
    return df_processed


def generate_synthetic_data_with_noise(df_source, n_samples, noise_factor, bounds):
    """Generates synthetic data by resampling, adding noise, and cleaning."""
    print(f"\n🛠️  Generating {n_samples} synthetic samples with noise...")
    if df_source.empty:
        print("❌ Source DataFrame is empty. Cannot generate synthetic data.")
        return pd.DataFrame()

    # Resample the original data with replacement
    synthetic_df = df_source.sample(n=n_samples, replace=True, random_state=42).copy()
    synthetic_df.reset_index(drop=True, inplace=True)

    print("  Adding noise to numerical features...")
    # Add Blood_Glucose_mg/dL to the list for noise addition if it exists
    cols_to_add_noise = NUMERIC_COLS_FOR_NOISE + ['Blood_Glucose_mg/dL']

    for col in set(cols_to_add_noise):  # Use set to avoid duplicates
        if col in df_source.columns:
            std_dev = df_source[col].std()
            if pd.notna(std_dev) and std_dev > 0:
                noise = np.random.normal(loc=0, scale=noise_factor * std_dev, size=n_samples)
                synthetic_df[col] += noise
            else:
                print(f"    ⚠️ Warning: Standard deviation for '{col}' is NaN or zero. Noise not added.")

    print("  Cleaning synthetic data by clipping to defined bounds...")
    for col, (min_val, max_val) in bounds.items():
        if col in synthetic_df.columns:
            synthetic_df[col] = synthetic_df[col].clip(lower=min_val, upper=max_val)

    # *** FIXED THE LOGIC ERROR HERE ***
    # Re-classify glucose levels AFTER noise has been added to BGL values.
    if 'Blood_Glucose_mg/dL' in synthetic_df.columns:
        print("  Re-classifying 'Glucose_classification_class' based on new BGL values.")
        synthetic_df['Glucose_classification_class'] = synthetic_df['Blood_Glucose_mg/dL'].apply(classify_glucose)

    # Apply integer formatting after clipping
    int_cols = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Pulse rate', 'Respiratory Rate']
    for col in int_cols:
        if col in synthetic_df.columns:
            synthetic_df[col] = synthetic_df[col].round(0).astype(int)

    print("✅ Synthetic data generation and cleaning complete.")
    return synthetic_df


# --- Main Execution ---
if __name__ == "__main__":
    df_original = load_and_prepare_data(ORIGINAL_DATA_FILE)

    # Initial classification for the real data
    if 'Blood_Glucose_mg/dL' in df_original.columns:
        df_original['Glucose_classification_class'] = df_original['Blood_Glucose_mg/dL'].apply(classify_glucose)
    else:
        print("❌ Critical Error: 'Blood_Glucose_mg/dL' not found in the original dataset.")
        exit()

    df_real_processed = preprocess_data(df_original)

    if df_real_processed.empty:
        print("❌ No data left after preprocessing. Halting execution.")
        exit()

    df_synthetic_cleaned = generate_synthetic_data_with_noise(
        df_real_processed,
        N_SYNTHETIC_SAMPLES,
        NOISE_SCALE_FACTOR,
        FEATURE_BOUNDS
    )

    # Combine real and synthetic data
    df_augmented = pd.concat([df_real_processed, df_synthetic_cleaned], ignore_index=True)
    print(
        f"\n🔗 Original ({len(df_real_processed)}) and Synthetic ({len(df_synthetic_cleaned)}) data combined. Total: {len(df_augmented)} rows."
    )

    # Save the augmented dataset
    try:
        df_augmented.to_csv(AUGMENTED_DATA_OUTPUT_FILE, index=False)
        print(f"\n💾 Augmented data saved to '{AUGMENTED_DATA_OUTPUT_FILE}'")
    except Exception as e:
        print(f"\n❌ Error saving augmented data: {e}")

    print("\n📋 Sample of Augmented Data (First 5 rows - Real):")
    print(df_augmented.head())
    print("\n📋 Sample of Augmented Data (Last 5 rows - Synthetic):")
    print(df_augmented.tail())

    print("\n🎉 Script finished.")
