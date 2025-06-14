# ==============================================================================
# Thesis Assistant: Synthetic vs. Real Data Quality Evaluation
# ==============================================================================
# Description:
# This script provides a comprehensive comparison between a real dataset and a
# synthetically generated one. It uses a combination of visual plots (KDE,
# boxplots, count plots, heatmaps) and statistical tests (Descriptive Stats,
# KS-Test, Jensen-Shannon Divergence) to evaluate the quality and fidelity
# of the synthetic data.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# IMPORTANT: Update these file paths to match your files.
REAL_DATA_FILE = "../data/raw_data.csv"  # The original data file used to train the CTGAN
SYNTHETIC_DATA_FILE = "../data/augmented_data_by_noise.csv" # The output from the CTGAN script

# Define columns to compare based on the latest CTGAN script
NUMERICAL_COLUMNS_TO_COMPARE = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Pulse rate',
    'HbA1c_%', 'Respiratory Rate', 'Acetone PPM 1.1', 'Temperature', 'Humidity'
]
DISCRETE_COLUMNS_TO_COMPARE = [
    'Glucose_classification_class'
]
# Number of bins for discretizing continuous data for JS Divergence
N_BINS_FOR_JS = 30


# --- 1. Load and Preprocess Datasets ---
try:
    df_real_raw = pd.read_csv(REAL_DATA_FILE)
    print(f"✅ Successfully loaded real data: {REAL_DATA_FILE} (Shape: {df_real_raw.shape})")
except FileNotFoundError:
    print(f"❌ Error: Real data file '{REAL_DATA_FILE}' not found.")
    exit()

try:
    df_synthetic = pd.read_csv(SYNTHETIC_DATA_FILE)
    print(f"✅ Successfully loaded synthetic data: {SYNTHETIC_DATA_FILE} (Shape: {df_synthetic.shape})")
except FileNotFoundError:
    print(f"❌ Error: Synthetic data file '{SYNTHETIC_DATA_FILE}' not found.")
    exit()

# Preprocess the real data exactly as in the generation script to make it comparable
print("\nPreprocessing real data for a fair comparison...")
df_real = df_real_raw.drop(columns=['Height', 'Weight'], errors='ignore')
if 'Blood Pressure' in df_real.columns:
    bp_split = df_real['Blood Pressure'].str.split('/', expand=True)
    df_real['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
    df_real['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
    df_real = df_real.drop('Blood Pressure', axis=1)

# Drop NaN rows from both dataframes to ensure statistical tests work correctly
df_real.dropna(inplace=True)
df_synthetic.dropna(inplace=True)
print("✅ Preprocessing and NaN removal complete.")


# --- Filter columns to only those present in both dataframes ---
common_numerical_cols = [col for col in NUMERICAL_COLUMNS_TO_COMPARE if
                         col in df_real.columns and col in df_synthetic.columns]
common_discrete_cols = [col for col in DISCRETE_COLUMNS_TO_COMPARE if
                        col in df_real.columns and col in df_synthetic.columns]

if not common_numerical_cols and not common_discrete_cols:
    print("\n❌ No common columns found to compare. Please check your column lists and CSV files.")
    exit()

# Add a 'Source' column for combined plotting
df_real['Source'] = 'Real'
df_synthetic['Source'] = 'Synthetic'
combined_df_plot = pd.concat([df_real, df_synthetic], ignore_index=True)


# --- 2. Visual Comparison ---
print("\n📊 Starting Visual Comparison...")
# KDE Plots
if common_numerical_cols:
    print("\n  Generating KDE Plots...")
    for col in common_numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=combined_df_plot, x=col, hue='Source', fill=True, common_norm=False, alpha=0.5)
        plt.title(f'Distribution Comparison for {col} (KDE)', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

# Boxplots
if common_numerical_cols:
    print("\n  Generating Boxplots...")
    for col in common_numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=combined_df_plot, x='Source', y=col)
        plt.title(f'Boxplot Comparison for {col}', fontsize=15)
        plt.xlabel('Data Source', fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

# Count Plots for Discrete Columns
if common_discrete_cols:
    print("\n  Generating Count Plots for Discrete Columns...")
    for col in common_discrete_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=combined_df_plot, x=col, hue='Source', dodge=True)
        plt.title(f'Distribution Comparison for {col} (Counts)', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.show()

# --- 3. Numerical Comparison ---
print("\n🔢 Starting Numerical Comparison...")

# Descriptive Statistics
if common_numerical_cols:
    print("\n  Descriptive Statistics:")
    stats_real = df_real[common_numerical_cols].describe().T
    stats_synthetic = df_synthetic[common_numerical_cols].describe().T
    comparison_stats = pd.concat([stats_real.add_suffix('_Real'), stats_synthetic.add_suffix('_Synthetic')], axis=1)
    print(comparison_stats)

# Kolmogorov-Smirnov (KS) Test
if common_numerical_cols:
    print("\n  Kolmogorov-Smirnov (KS) Test for distribution similarity:")
    print("  (A high p-value, e.g., > 0.05, suggests the distributions are similar)")
    ks_results = []
    for col in common_numerical_cols:
        ks_statistic, p_value = stats.ks_2samp(df_real[col], df_synthetic[col])
        ks_results.append({'Column': col, 'KS Statistic': ks_statistic, 'P-Value': p_value})
    ks_df = pd.DataFrame(ks_results)
    print(ks_df.round(4))

# Jensen-Shannon (JS) Divergence
def get_probability_distribution(data_series, n_bins, global_min, global_max):
    counts, _ = np.histogram(data_series, bins=n_bins, range=(global_min, global_max))
    # Add a small epsilon to avoid zero probabilities
    probabilities = counts / counts.sum()
    return np.where(probabilities == 0, 1e-10, probabilities)

if common_numerical_cols:
    print("\n  Jensen-Shannon (JS) Divergence for Numerical Columns (0=identical, higher=more different):")
    js_numerical_results = []
    for col in common_numerical_cols:
        real_data = df_real[col]
        synthetic_data = df_synthetic[col]
        global_min, global_max = min(real_data.min(), synthetic_data.min()), max(real_data.max(), synthetic_data.max())
        p = get_probability_distribution(real_data, N_BINS_FOR_JS, global_min, global_max)
        q = get_probability_distribution(synthetic_data, N_BINS_FOR_JS, global_min, global_max)
        js_div = jensenshannon(p, q, base=2)
        js_numerical_results.append({'Column': col, 'JS Divergence': js_div})
    js_numerical_df = pd.DataFrame(js_numerical_results)
    print(js_numerical_df.round(4))

if common_discrete_cols:
    print("\n  Jensen-Shannon (JS) Divergence for Discrete Columns:")
    js_discrete_results = []
    for col in common_discrete_cols:
        p_counts = df_real[col].value_counts(normalize=True)
        q_counts = df_synthetic[col].value_counts(normalize=True)
        all_categories = p_counts.index.union(q_counts.index)
        p = p_counts.reindex(all_categories, fill_value=1e-10)
        q = q_counts.reindex(all_categories, fill_value=1e-10)
        js_div = jensenshannon(p, q, base=2)
        js_discrete_results.append({'Column': col, 'JS Divergence': js_div})
    js_discrete_df = pd.DataFrame(js_discrete_results)
    print(js_discrete_df.round(4))

# Correlation Matrix Comparison
if len(common_numerical_cols) > 1:
    print("\n  Correlation Matrix Comparison (Heatmaps):")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    sns.heatmap(df_real[common_numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Correlation Matrix - Real Data', fontsize=15)
    sns.heatmap(df_synthetic[common_numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Correlation Matrix - Synthetic Data', fontsize=15)
    plt.tight_layout()
    plt.show()

print("\n🎉 Comparison Script Finished.")
