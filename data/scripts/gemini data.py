import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# --- Configuration ---
N_SAMPLES_PER_CLASS = 50  # Number of samples to generate for each glucose class
N_TOTAL_SAMPLES = N_SAMPLES_PER_CLASS * 3
OUTPUT_FILENAME = "../data/test_new_data_correlated_v1.csv"

# Define column names (constants for easy reference)
COL_ACETONE = 'Breath_Acetone_ppm'
COL_BGL = 'Blood_Glucose_mg/dL'
COL_HBA1C = 'HbA1c_%'
COL_BHB = 'β-Hydroxybutyrate_mmol/L'
COL_AGE = 'Age'
COL_BMI = 'BMI'
COL_TEMP = 'Temp_C'
COL_HUMIDITY = 'Humidity_%'
COL_FASTING = 'Fasting_Hours'
COL_GLUCOSE_CLASS = 'Glucose_Level_Class'


# --- Helper Functions ---
def classify_glucose_level(value):
    if pd.isna(value):
        return 'Undefined'
    if value < 70:
        return 'Low'
    elif 70 <= value <= 139:
        return 'Normal'
    else:  # >= 140
        return 'High'


def generate_correlated_value(base_value, scale_factor, noise_std_dev_ratio=0.1, positive_corr=True):
    """Generates a value correlated to a base_value."""
    correlation_effect = base_value * scale_factor
    noise = np.random.normal(0, np.abs(
        correlation_effect * noise_std_dev_ratio) + 1e-6)  # Add small epsilon to avoid 0 std
    if positive_corr:
        return base_value + correlation_effect + noise
    else:
        return base_value - correlation_effect + noise


# --- Data Generation ---
all_data = []

print(f"Generating {N_SAMPLES_PER_CLASS} samples for each class: Low, Normal, High...")

# Generate 'Low' BGL samples
for _ in range(N_SAMPLES_PER_CLASS):
    bgl = np.round(np.random.uniform(20, 69.9), 1)
    fasting_hours = np.random.randint(8, 17)  # Tend to be higher for low BGL

    # HbA1c: Lower, somewhat correlated with BGL but with more variance for hypoglycemia
    hba1c = np.round(np.random.normal(loc=5.0 + (bgl / 70), scale=0.8), 2)
    hba1c = np.clip(hba1c, 4.0, 7.5)  # Hypoglycemia doesn't always mean perfect long-term control

    # Ketones: Can be slightly elevated in hypoglycemia/fasting
    base_ketone_propensity = 0.5 + (fasting_hours / 16.0)  # Higher fasting -> higher base
    bhb = np.round(np.random.uniform(0.3, 2.5) * base_ketone_propensity, 2)
    acetone = np.round(bhb * np.random.uniform(1.5, 3.5) + np.random.normal(0, 0.5), 2)  # Acetone correlated with BHB
    bhb = np.clip(bhb, 0.1, 5.0)
    acetone = np.clip(acetone, 0.1, 10.0)

    age = np.random.randint(18, 80)
    bmi = np.round(np.random.uniform(18.5, 35.0), 2)  # Less direct correlation for 'Low'
    temp_c = np.round(np.random.uniform(35.5, 37.5), 1)  # Normal temp range
    humidity = np.round(np.random.uniform(30, 70), 1)

    all_data.append({
        COL_ACETONE: acetone, COL_BGL: bgl, COL_HBA1C: hba1c, COL_BHB: bhb,
        COL_AGE: age, COL_BMI: bmi, COL_TEMP: temp_c, COL_HUMIDITY: humidity,
        COL_FASTING: fasting_hours, COL_GLUCOSE_CLASS: 'Low'
    })

# Generate 'Normal' BGL samples
for _ in range(N_SAMPLES_PER_CLASS):
    bgl = np.round(np.random.uniform(70, 139.9), 1)
    fasting_hours = np.random.randint(4, 13)  # Varied

    # HbA1c: Correlated with BGL, in normal to prediabetic range
    hba1c = np.round(np.random.normal(loc=4.5 + (bgl / 30.0), scale=0.5), 2)
    hba1c = np.clip(hba1c, 4.5, 7.0)

    # Ketones: Generally lower for normal BGL, unless long fasting
    base_ketone_propensity = 0.1 + (fasting_hours / 20.0)
    bhb = np.round(np.random.uniform(0.1, 1.0) * base_ketone_propensity, 2)
    acetone = np.round(bhb * np.random.uniform(1.0, 2.5) + np.random.normal(0, 0.2), 2)
    bhb = np.clip(bhb, 0.0, 2.0)
    acetone = np.clip(acetone, 0.0, 5.0)

    age = np.random.randint(18, 80)
    bmi = np.round(np.random.uniform(18.5, 40.0), 2)
    temp_c = np.round(np.random.uniform(36.0, 37.8), 1)
    humidity = np.round(np.random.uniform(30, 80), 1)

    all_data.append({
        COL_ACETONE: acetone, COL_BGL: bgl, COL_HBA1C: hba1c, COL_BHB: bhb,
        COL_AGE: age, COL_BMI: bmi, COL_TEMP: temp_c, COL_HUMIDITY: humidity,
        COL_FASTING: fasting_hours, COL_GLUCOSE_CLASS: 'Normal'
    })

# Generate 'High' BGL samples
for _ in range(N_SAMPLES_PER_CLASS):
    bgl = np.round(np.random.uniform(140, 350), 1)  # Wider range for high
    fasting_hours = np.random.randint(2, 12)  # Can be low if postprandial high

    # HbA1c: Strongly correlated with high BGL
    hba1c = np.round(np.random.normal(loc=6.0 + (bgl / 25.0), scale=1.0), 2)
    hba1c = np.clip(hba1c, 6.5, 15.0)

    # Ketones: Can be elevated, especially if BGL is very high
    base_ketone_propensity = 0.2 + (bgl / 150.0) - (fasting_hours / 24.0)  # Higher BGL -> higher, recent meal -> lower
    bhb = np.round(np.random.uniform(0.2, 4.0) * np.clip(base_ketone_propensity, 0.1, 3.0), 2)
    acetone = np.round(bhb * np.random.uniform(2.0, 4.0) + np.random.normal(0, 1.0), 2)
    bhb = np.clip(bhb, 0.1, 10.0)
    acetone = np.clip(acetone, 0.2, 20.0)

    age = np.random.randint(30, 80)  # High BGL might be more common in older or specific populations
    # BMI: Might tend to be higher for 'High' BGL (Type 2 context)
    bmi_base = 22 + (bgl / 50.0)
    bmi = np.round(np.random.normal(loc=bmi_base, scale=3.0), 2)
    bmi = np.clip(bmi, 18.5, 45.0)

    temp_c = np.round(np.random.uniform(36.0, 38.5), 1)  # Can be slightly elevated if unwell
    humidity = np.round(np.random.uniform(30, 80), 1)

    all_data.append({
        COL_ACETONE: acetone, COL_BGL: bgl, COL_HBA1C: hba1c, COL_BHB: bhb,
        COL_AGE: age, COL_BMI: bmi, COL_TEMP: temp_c, COL_HUMIDITY: humidity,
        COL_FASTING: fasting_hours, COL_GLUCOSE_CLASS: 'High'
    })

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Introduce some missing values (optional, for realism) ---
# Example: 5% missing for HbA1c, 3% for BHB
hba1c_missing_indices = np.random.choice(df.index, size=int(N_TOTAL_SAMPLES * 0.05), replace=False)
df.loc[hba1c_missing_indices, COL_HBA1C] = np.nan

bhb_missing_indices = np.random.choice(df.index, size=int(N_TOTAL_SAMPLES * 0.03), replace=False)
df.loc[bhb_missing_indices, COL_BHB] = np.nan

print(f"\nIntroduced some missing values. Counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# --- Verify Glucose Level Classification (should match the generation logic) ---
# This is more of a sanity check as we assigned it directly
df[COL_GLUCOSE_CLASS + "_Verify"] = df[COL_BGL].apply(classify_glucose_level)
mismatched_classes = df[df[COL_GLUCOSE_CLASS] != df[COL_GLUCOSE_CLASS + "_Verify"]]
if not mismatched_classes.empty:
    print(
        f"\n⚠️ Warning: {len(mismatched_classes)} samples have mismatched generated vs. calculated glucose classes. Review generation logic.")
    print(mismatched_classes[[COL_BGL, COL_GLUCOSE_CLASS, COL_GLUCOSE_CLASS + "_Verify"]].head())
df.drop(columns=[COL_GLUCOSE_CLASS + "_Verify"], inplace=True)

# --- Create data directory if it doesn't exist ---
output_dir = os.path.dirname(OUTPUT_FILENAME)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# --- Save to CSV ---
try:
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n✅ Test data with more realistic correlations generated and saved to '{OUTPUT_FILENAME}'")
    print(f"   Total samples: {len(df)}")
    print(
        f"   Class distribution:\n{df[COL_GLUCOSE_CLASS].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'}")
except Exception as e:
    print(f"\n❌ Error saving data to '{OUTPUT_FILENAME}': {e}")

print("\nSample of generated data:")
print(df.head())
print("\nDescriptive statistics of generated data:")
print(df.describe())

