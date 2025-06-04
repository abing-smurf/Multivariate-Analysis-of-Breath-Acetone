import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For KS test

# --- Configuration ---
REAL_DATA_FILE = "../data/dummy_glucose_data.csv"  # Or your actual preprocessed real data file
SYNTHETIC_DATA_FILE = "../data/augmented_glucose_data_noise_added_cleaned.csv"

# Define columns you want to compare
# These should be the numerical columns present in both datasets after all processing
# Ensure these columns exist in both loaded dataframes
NUMERICAL_COLUMNS_TO_COMPARE = [
    'Age', 'BMI', 'Blood_Glucose_mg/dL', 'HbA1c_%',
    'Breath_Acetone_ppm', 'β-Hydroxybutyrate_mmol/L', # Use your defined constants if preferred
    'Temp_C', 'Humidity_%', 'Fasting_Hours'
]

# Define discrete columns you want to compare distributions for (e.g., value counts)
DISCRETE_COLUMNS_TO_COMPARE = [
    'Glucose_Level_Class'
    # Add other discrete columns like 'Gender', 'Type_of_diabetes' if present
]

# --- 1. Load Datasets ---
try:
    df_real = pd.read_csv(REAL_DATA_FILE)
    print(f"✅ Successfully loaded real data: {REAL_DATA_FILE} (Shape: {df_real.shape})")
except FileNotFoundError:
    print(f"❌ Error: Real data file '{REAL_DATA_FILE}' not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading real data file '{REAL_DATA_FILE}': {e}")
    exit()

try:
    df_synthetic = pd.read_csv(SYNTHETIC_DATA_FILE)
    print(f"✅ Successfully loaded synthetic data: {SYNTHETIC_DATA_FILE} (Shape: {df_synthetic.shape})")
except FileNotFoundError:
    print(f"❌ Error: Synthetic data file '{SYNTHETIC_DATA_FILE}' not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading synthetic data file '{SYNTHETIC_DATA_FILE}': {e}")
    exit()

# --- Filter columns to only those present in both dataframes ---
common_numerical_cols = [col for col in NUMERICAL_COLUMNS_TO_COMPARE if col in df_real.columns and col in df_synthetic.columns]
common_discrete_cols = [col for col in DISCRETE_COLUMNS_TO_COMPARE if col in df_real.columns and col in df_synthetic.columns]

if not common_numerical_cols:
    print("\n⚠️ No common numerical columns found to compare. Please check your column lists and CSV files.")
if not common_discrete_cols:
    print("\n⚠️ No common discrete columns found to compare. Please check your column lists and CSV files.")


# Add a 'Source' column for combined plotting
df_real['Source'] = 'Real'
df_synthetic['Source'] = 'Synthetic'
combined_df = pd.concat([df_real, df_synthetic], ignore_index=True)

# --- 2. Visual Comparison ---
print("\n📊 Starting Visual Comparison...")

# KDE Plots for Numerical Columns
if common_numerical_cols:
    print("\n  Generating KDE Plots...")
    for col in common_numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=combined_df, x=col, hue='Source', fill=True, common_norm=False, alpha=0.5)
        plt.title(f'Distribution Comparison for {col} (KDE)', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.tight_layout()
        plt.show()
        print(f"    ✅ KDE plot for {col} generated.")
else:
    print("  Skipping KDE plots as no common numerical columns were identified.")

# Boxplots for Numerical Columns
if common_numerical_cols:
    print("\n  Generating Boxplots...")
    for col in common_numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=combined_df, x='Source', y=col)
        plt.title(f'Boxplot Comparison for {col}', fontsize=15)
        plt.xlabel('Data Source', fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.tight_layout()
        plt.show()
        print(f"    ✅ Boxplot for {col} generated.")
else:
    print("  Skipping Boxplots as no common numerical columns were identified.")

# Count Plots for Discrete Columns
if common_discrete_cols:
    print("\n  Generating Count Plots for Discrete Columns...")
    for col in common_discrete_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=combined_df, x=col, hue='Source', dodge=True)
        plt.title(f'Distribution Comparison for {col} (Counts)', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print(f"    ✅ Count plot for {col} generated.")
else:
    print("  Skipping Count plots as no common discrete columns were identified.")


# --- 3. Numerical Comparison ---
print("\n🔢 Starting Numerical Comparison...")

# Descriptive Statistics for Numerical Columns
if common_numerical_cols:
    print("\n  Descriptive Statistics:")
    stats_real = df_real[common_numerical_cols].describe().T
    stats_synthetic = df_synthetic[common_numerical_cols].describe().T

    comparison_stats = pd.concat([stats_real.add_suffix('_Real'), stats_synthetic.add_suffix('_Synthetic')], axis=1)
    print(comparison_stats)
else:
    print("  Skipping Descriptive Statistics as no common numerical columns were identified.")


# Kolmogorov-Smirnov (KS) Test for Numerical Columns
# Null Hypothesis (H0): The two samples are drawn from the same distribution.
# If p-value > alpha (e.g., 0.05), we do not reject H0.
if common_numerical_cols:
    print("\n  Kolmogorov-Smirnov (KS) Test for distribution similarity (comparing Real vs. Synthetic):")
    print("  (p-value > 0.05 suggests distributions are similar)")
    ks_results = []
    for col in common_numerical_cols:
        # Ensure no NaNs in columns being compared for KS test, or handle them
        real_col_data = df_real[col].dropna()
        synthetic_col_data = df_synthetic[col].dropna()
        if len(real_col_data) > 1 and len(synthetic_col_data) > 1: # KS test requires at least 2 samples
            ks_statistic, p_value = stats.ks_2samp(real_col_data, synthetic_col_data)
            ks_results.append({'Column': col, 'KS Statistic': ks_statistic, 'P-Value': p_value})
        else:
            ks_results.append({'Column': col, 'KS Statistic': np.nan, 'P-Value': 'Not enough data'})

    ks_df = pd.DataFrame(ks_results)
    print(ks_df)
else:
    print("  Skipping KS Test as no common numerical columns were identified.")


# Value Counts for Discrete Columns
if common_discrete_cols:
    print("\n  Value Counts for Discrete Columns (Normalized Percentage %):")
    for col in common_discrete_cols:
        print(f"\n    Column: {col}")
        real_counts = df_real[col].value_counts(normalize=True).mul(100).round(2)
        synthetic_counts = df_synthetic[col].value_counts(normalize=True).mul(100).round(2)

        counts_comparison_df = pd.DataFrame({
            'Real (%)': real_counts,
            'Synthetic (%)': synthetic_counts
        }).fillna(0) # Fill with 0 if a category is missing in one dataset
        print(counts_comparison_df)
else:
    print("  Skipping Value Counts as no common discrete columns were identified.")

# Correlation Matrix Comparison for Numerical Columns
if common_numerical_cols and len(common_numerical_cols) > 1: # Correlation needs at least 2 columns
    print("\n  Correlation Matrix Comparison (Heatmaps):")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_real[common_numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix - Real Data', fontsize=15)
    plt.tight_layout()
    plt.show()
    print("    ✅ Correlation heatmap for Real Data generated.")

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_synthetic[common_numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix - Synthetic Data', fontsize=15)
    plt.tight_layout()
    plt.show()
    print("    ✅ Correlation heatmap for Synthetic Data generated.")
else:
    print("  Skipping Correlation Matrix comparison (not enough common numerical columns).")

print("\n🎉 Comparison Script Finished.")