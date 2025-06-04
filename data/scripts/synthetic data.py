import pandas as pd
import numpy as np

# --- Configuration ---
ORIGINAL_DATA_FILE = "../data/dummy_glucose_data.csv"
AUGMENTED_DATA_OUTPUT_FILE = "../data/augmented_glucose_data_noise_added_cleaned.csv"  # Output file

# Column Names (Adjust if your CSV uses different names)
COL_HBA1C = 'HbA1c_%'
COL_BGL = 'Blood_Glucose_mg/dL'
COL_GLUCOSE_CLASS = 'Glucose_Level_Class'
COL_ACETONE = 'Breath_Acetone_ppm'
COL_BHB = 'β-Hydroxybutyrate_mmol/L'  # Beta-Hydroxybutyrate
COL_AGE = 'Age'
COL_BMI = 'BMI'
COL_TEMP = 'Temp_C'
COL_HUMIDITY = 'Humidity_%'
COL_FASTING = 'Fasting_Hours'

# Parameters for Synthetic Data Generation
N_SYNTHETIC_SAMPLES = 500  # Number of synthetic samples to generate
NOISE_SCALE_FACTOR = 0.05  # Factor for noise level (e.g., 0.05 = 5% of std dev)

# List of numeric columns to apply noise to and clean
NUMERIC_COLS_FOR_NOISE = [
    COL_ACETONE, COL_BGL, COL_HBA1C, COL_BHB,
    COL_AGE, COL_BMI, COL_TEMP, COL_HUMIDITY, COL_FASTING
]

# Define realistic bounds for cleaning synthetic numerical features {column: (min, max)}
# Use None if no bound is needed on one side (e.g., (0, None) means min is 0, no upper bound)
FEATURE_BOUNDS = {
    COL_AGE: (18, 90),  # Example: Age between 18 and 90
    COL_BMI: (10, 60),  # Example: BMI between 10 and 60
    COL_BGL: (20, 700),  # Example: BGL between 20 and 700 mg/dL
    COL_HBA1C: (3.0, 20.0),  # Example: HbA1c between 3% and 20%
    COL_ACETONE: (0.0, 500.0),  # Example: Acetone >= 0, up to 500 ppm (adjust as needed)
    COL_BHB: (0.0, 30.0),  # Example: BHB >= 0, up to 30 mmol/L (adjust as needed)
    COL_TEMP: (35.0, 42.0),  # Example: Body Temp in Celsius
    COL_HUMIDITY: (0, 100),  # Example: Humidity percentage
    COL_FASTING: (0, 72)  # Example: Fasting hours
}


# --- Helper Functions ---

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully from '{file_path}'. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found. Please ensure the file exists.")
        exit()
    except Exception as e:
        print(f"❌ Error loading file '{file_path}': {e}")
        exit()


def classify_glucose(value):
    """Classifies glucose value into Low, Normal, High, or Undefined."""
    if pd.isna(value):
        return 'Undefined'
    if value < 70:
        return 'Low'
    elif 70 <= value <= 139:  # Common definition for Normal fasting/preprandial
        return 'Normal'
    else:  # > 139
        return 'High'


def preprocess_real_data(df):
    """Preprocesses the real dataset: fills NaNs, handles initial out-of-range, classifies glucose."""
    print("\n⚙️  Preprocessing real data...")
    df_processed = df.copy()

    # Fill missing HbA1c
    if COL_HBA1C in df_processed.columns and df_processed[COL_HBA1C].isnull().any():
        mean_hba1c = df_processed[COL_HBA1C].mean()
        df_processed[COL_HBA1C].fillna(mean_hba1c, inplace=True)
        print(f"  Filled NaNs in '{COL_HBA1C}' with mean: {mean_hba1c:.2f}")

    # Handle out-of-range and NaNs for Acetone
    if COL_ACETONE in df_processed.columns:
        neg_acetone_count = (df_processed[COL_ACETONE] < 0).sum()
        if neg_acetone_count > 0:
            df_processed.loc[df_processed[COL_ACETONE] < 0, COL_ACETONE] = np.nan  # Convert negatives to NaN first
            print(f"  Converted {neg_acetone_count} negative '{COL_ACETONE}' values to NaN.")
        if df_processed[COL_ACETONE].isnull().any():
            median_acetone = df_processed[COL_ACETONE].median()
            df_processed[COL_ACETONE].fillna(median_acetone, inplace=True)
            print(f"  Filled NaNs in '{COL_ACETONE}' with median: {median_acetone:.2f}")

    # Handle out-of-range and NaNs for BHB
    if COL_BHB in df_processed.columns:
        neg_bhb_count = (df_processed[COL_BHB] < 0).sum()
        if neg_bhb_count > 0:
            df_processed.loc[df_processed[COL_BHB] < 0, COL_BHB] = np.nan  # Convert negatives to NaN first
            print(f"  Converted {neg_bhb_count} negative '{COL_BHB}' values to NaN.")
        if df_processed[COL_BHB].isnull().any():
            median_bhb = df_processed[COL_BHB].median()
            df_processed[COL_BHB].fillna(median_bhb, inplace=True)
            print(f"  Filled NaNs in '{COL_BHB}' with median: {median_bhb:.2f}")

    # Ensure BGL column exists
    if COL_BGL not in df_processed.columns:
        print(f"❌ Error: '{COL_BGL}' column is required but not found in the dataset.")
        exit()

    # Classify glucose levels
    df_processed[COL_GLUCOSE_CLASS] = df_processed[COL_BGL].apply(classify_glucose)
    print(f"  Classified '{COL_GLUCOSE_CLASS}'.")

    print("✅ Real data preprocessing complete.")
    return df_processed


def generate_synthetic_data_with_noise(df_source, numeric_cols_to_noise, n_samples, noise_factor, bounds):
    """Generates synthetic data by resampling, adding noise, and cleaning."""
    print(f"\n🛠️  Generating {n_samples} synthetic samples with noise...")
    if df_source.empty:
        print("❌ Source DataFrame is empty. Cannot generate synthetic data.")
        return pd.DataFrame()

    synthetic_df = df_source.sample(n=n_samples, replace=True, random_state=42).copy()
    synthetic_df.reset_index(drop=True, inplace=True)  # Reset index for new df

    print("  Adding noise to numerical features...")
    for col in numeric_cols_to_noise:
        if col in df_source.columns and col in synthetic_df.columns:  # Check if col exists in both
            std_dev = df_source[col].std()
            # Ensure std_dev is not NaN or zero before generating noise
            if pd.notna(std_dev) and std_dev > 0:
                noise = np.random.normal(loc=0, scale=noise_factor * std_dev, size=n_samples)
                synthetic_df[col] += noise
            else:
                print(f"    ⚠️ Warning: Standard deviation for '{col}' is NaN or zero. Noise not added.")
        else:
            print(f"    ⚠️ Warning: Column '{col}' not found in source or synthetic df. Skipping noise addition.")

    print("  Cleaning synthetic data (clipping to bounds)...")
    for col, (min_val, max_val) in bounds.items():
        if col in synthetic_df.columns:
            original_min = synthetic_df[col].min()  # For logging
            original_max = synthetic_df[col].max()  # For logging
            synthetic_df[col] = synthetic_df[col].clip(lower=min_val, upper=max_val)
            # print(f"    Clipped '{col}'. Before: ({original_min:.2f}-{original_max:.2f}), After: ({synthetic_df[col].min():.2f}-{synthetic_df[col].max():.2f})")
        else:
            print(f"    ⚠️ Warning: Column '{col}' not found in synthetic_df for clipping.")

    # Re-classify glucose levels AFTER noise addition and clipping of BGL
    if COL_BGL in synthetic_df.columns:
        synthetic_df[COL_GLUCOSE_CLASS] = synthetic_df[COL_BGL].apply(classify_glucose)
        print(f"  Re-classified '{COL_GLUCOSE_CLASS}' for synthetic data.")
    else:
        print(f"    ⚠️ Warning: '{COL_BGL}' not found in synthetic_df. Cannot re-classify glucose levels.")

    print("✅ Synthetic data generation and initial cleaning complete.")
    return synthetic_df


# --- Main Execution ---
if __name__ == "__main__":
    df_original = load_data(ORIGINAL_DATA_FILE)

    # Ensure all numeric columns targeted for noise exist in the dataframe
    # and filter out any that don't to prevent errors later
    existing_numeric_cols_for_noise = [col for col in NUMERIC_COLS_FOR_NOISE if col in df_original.columns]
    if len(existing_numeric_cols_for_noise) != len(NUMERIC_COLS_FOR_NOISE):
        print("⚠️ Warning: Some specified numeric columns for noise were not found in the original data.")
        print(f"   Using these existing columns for noise addition: {existing_numeric_cols_for_noise}")

    df_real_processed = preprocess_real_data(df_original)

    # Check for remaining NaNs in processed real data before generating synthetic data
    # CTGAN part was removed, but for noise addition, source data should ideally be clean
    # If df_source for sample() has NaNs, synthetic_df might have them too before noise.
    # df.sample() copies NaNs.
    # If source_df[col].std() is NaN due to all NaNs in col, noise isn't added correctly.
    # Best to ensure df_real_processed used as source for df.sample is mostly clean.
    # The preprocess_real_data function already handles NaNs for some key columns.
    # For a robust approach, you might consider dropping rows with any NaNs in critical columns
    # from df_real_processed before using it as a source for sampling if issues persist.
    # e.g., df_real_processed_for_sampling = df_real_processed.dropna(subset=existing_numeric_cols_for_noise)

    df_synthetic_cleaned = generate_synthetic_data_with_noise(
        df_real_processed,  # Use the preprocessed real data as the source
        existing_numeric_cols_for_noise,
        N_SYNTHETIC_SAMPLES,
        NOISE_SCALE_FACTOR,
        FEATURE_BOUNDS
    )

    # Combine real and synthetic data
    df_augmented = pd.concat([df_real_processed, df_synthetic_cleaned], ignore_index=True)
    print(
        f"\n🔗 Original ({len(df_real_processed)}) and Synthetic ({len(df_synthetic_cleaned)}) data combined. Total: {len(df_augmented)} rows.")

    # Save the augmented dataset
    try:
        df_augmented.to_csv(AUGMENTED_DATA_OUTPUT_FILE, index=False)
        print(f"\n💾 Augmented data saved to '{AUGMENTED_DATA_OUTPUT_FILE}'")
    except Exception as e:
        print(f"\n❌ Error saving augmented data: {e}")

    print("\n📋 Sample of Augmented Data (First 5 rows):")
    print(df_augmented.head())
    print("\n📋 Sample of Augmented Data (Last 5 rows - likely synthetic):")
    print(df_augmented.tail())

    print("\n🎉 Script finished.")