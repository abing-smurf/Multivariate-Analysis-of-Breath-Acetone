import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # For KS test
from scipy.spatial.distance import jensenshannon  # For JS Divergence
from scipy.stats import entropy  # For Kullback-Leibler divergence, used in JS

# --- Configuration ---
REAL_DATA_FILE = "../data/dummy_glucose_data.csv"  # Or your actual preprocessed real data file
SYNTHETIC_DATA_FILE = "../data/smote_balanced_training_data.csv"

# Define columns you want to compare
NUMERICAL_COLUMNS_TO_COMPARE = [
    'Age', 'BMI', 'Blood_Glucose_mg/dL', 'HbA1c_%',
    'Breath_Acetone_ppm', 'β-Hydroxybutyrate_mmol/L',
    'Temp_C', 'Humidity_%', 'Fasting_Hours'
]
DISCRETE_COLUMNS_TO_COMPARE = [
    'Glucose_Level_Class'
]
# Number of bins for discretizing continuous data for JS Divergence
N_BINS_FOR_JS = 30  # You can adjust this

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
common_numerical_cols = [col for col in NUMERICAL_COLUMNS_TO_COMPARE if
                         col in df_real.columns and col in df_synthetic.columns]
common_discrete_cols = [col for col in DISCRETE_COLUMNS_TO_COMPARE if
                        col in df_real.columns and col in df_synthetic.columns]

if not common_numerical_cols:
    print("\n⚠️ No common numerical columns found to compare. Please check your column lists and CSV files.")
if not common_discrete_cols:
    print("\n⚠️ No common discrete columns found to compare. Please check your column lists and CSV files.")

# Add a 'Source' column for combined plotting
df_real_plot = df_real.copy()  # Use copies for plotting to avoid modifying original dfs if script is re-run
df_synthetic_plot = df_synthetic.copy()
df_real_plot['Source'] = 'Real'
df_synthetic_plot['Source'] = 'Synthetic'
combined_df_plot = pd.concat([df_real_plot, df_synthetic_plot], ignore_index=True)

# --- 2. Visual Comparison (No changes here, kept for completeness) ---
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
        plt.tight_layout()
        plt.show()
        # print(f"    ✅ KDE plot for {col} generated.")
else:
    print("  Skipping KDE plots as no common numerical columns were identified.")
# Boxplots
if common_numerical_cols:
    print("\n  Generating Boxplots...")
    for col in common_numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=combined_df_plot, x='Source', y=col)
        plt.title(f'Boxplot Comparison for {col}', fontsize=15)
        plt.xlabel('Data Source', fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.tight_layout()
        plt.show()
        # print(f"    ✅ Boxplot for {col} generated.")
else:
    print("  Skipping Boxplots as no common numerical columns were identified.")
# Count Plots
if common_discrete_cols:
    print("\n  Generating Count Plots for Discrete Columns...")
    for col in common_discrete_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=combined_df_plot, x=col, hue='Source', dodge=True)
        plt.title(f'Distribution Comparison for {col} (Counts)', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        # print(f"    ✅ Count plot for {col} generated.")
else:
    print("  Skipping Count plots as no common discrete columns were identified.")

# --- 3. Numerical Comparison ---
print("\n🔢 Starting Numerical Comparison...")

# Descriptive Statistics
if common_numerical_cols:
    print("\n  Descriptive Statistics:")
    stats_real = df_real[common_numerical_cols].describe().T
    stats_synthetic = df_synthetic[common_numerical_cols].describe().T
    comparison_stats = pd.concat([stats_real.add_suffix('_Real'), stats_synthetic.add_suffix('_Synthetic')], axis=1)
    print(comparison_stats)
else:
    print("  Skipping Descriptive Statistics as no common numerical columns were identified.")

# Kolmogorov-Smirnov (KS) Test
if common_numerical_cols:
    print("\n  Kolmogorov-Smirnov (KS) Test for distribution similarity (comparing Real vs. Synthetic):")
    print("  (p-value > 0.05 suggests distributions are similar)")
    ks_results = []
    for col in common_numerical_cols:
        real_col_data = df_real[col].dropna()
        synthetic_col_data = df_synthetic[col].dropna()
        if len(real_col_data) > 1 and len(synthetic_col_data) > 1:
            ks_statistic, p_value = stats.ks_2samp(real_col_data, synthetic_col_data)
            ks_results.append({'Column': col, 'KS Statistic': ks_statistic, 'P-Value': p_value})
        else:
            ks_results.append({'Column': col, 'KS Statistic': np.nan, 'P-Value': 'Not enough data'})
    ks_df = pd.DataFrame(ks_results)
    print(ks_df)
else:
    print("  Skipping KS Test as no common numerical columns were identified.")


# --- NEW: Jensen-Shannon Divergence ---
def get_probability_distribution(data_series, n_bins, global_min, global_max):
    """Helper function to get binned probability distribution for numerical data."""
    # Create histogram counts using common bins
    counts, bin_edges = np.histogram(data_series.dropna(), bins=n_bins, range=(global_min, global_max))
    # Convert counts to probabilities
    probabilities = counts / counts.sum()
    # Add a small epsilon to avoid zero probabilities for entropy calculation
    probabilities = np.where(probabilities == 0, 1e-10, probabilities)
    return probabilities, bin_edges


if common_numerical_cols:
    print("\n  Jensen-Shannon (JS) Divergence for Numerical Columns (0=identical, ~0.693=max different for log_e):")
    js_numerical_results = []
    for col in common_numerical_cols:
        real_data = df_real[col].dropna()
        synthetic_data = df_synthetic[col].dropna()

        if len(real_data) == 0 or len(synthetic_data) == 0:
            js_numerical_results.append({'Column': col, 'JS Divergence': np.nan, 'Note': 'Empty data for column'})
            continue

        # Determine common range for binning
        global_min = min(real_data.min(), synthetic_data.min())
        global_max = max(real_data.max(), synthetic_data.max())

        if global_min == global_max:  # Handle case where all values are the same
            js_numerical_results.append(
                {'Column': col, 'JS Divergence': 0.0 if len(real_data) > 0 and len(synthetic_data) > 0 else np.nan,
                 'Note': 'All values identical or insufficient data'})
            continue

        p, _ = get_probability_distribution(real_data, N_BINS_FOR_JS, global_min, global_max)
        q, _ = get_probability_distribution(synthetic_data, N_BINS_FOR_JS, global_min, global_max)

        try:
            js_div = jensenshannon(p, q)  # Uses natural logarithm by default
            js_numerical_results.append({'Column': col, 'JS Divergence': js_div, 'Note': ''})
        except ValueError as e:
            js_numerical_results.append({'Column': col, 'JS Divergence': np.nan, 'Note': f'Error: {e}'})

    js_numerical_df = pd.DataFrame(js_numerical_results)
    print(js_numerical_df)
else:
    print("  Skipping JS Divergence for numerical columns.")

if common_discrete_cols:
    print("\n  Jensen-Shannon (JS) Divergence for Discrete Columns (0=identical, ~0.693=max different for log_e):")
    js_discrete_results = []
    for col in common_discrete_cols:
        p_counts = df_real[col].value_counts(normalize=True)
        q_counts = df_synthetic[col].value_counts(normalize=True)

        # Align categories: create a combined index and reindex both series, filling missing with 0
        all_categories = p_counts.index.union(q_counts.index)
        p = p_counts.reindex(all_categories, fill_value=0.0)
        q = q_counts.reindex(all_categories, fill_value=0.0)

        # Add a small epsilon to avoid zero probabilities
        p = np.where(p == 0, 1e-10, p)
        q = np.where(q == 0, 1e-10, q)

        # Ensure they sum to 1 after adding epsilon (normalize again if necessary)
        p /= np.sum(p)
        q /= np.sum(q)

        try:
            js_div = jensenshannon(p, q)
            js_discrete_results.append({'Column': col, 'JS Divergence': js_div})
        except ValueError as e:
            js_discrete_results.append({'Column': col, 'JS Divergence': np.nan, 'Note': f'Error: {e}'})

    js_discrete_df = pd.DataFrame(js_discrete_results)
    print(js_discrete_df)
else:
    print("  Skipping JS Divergence for discrete columns.")

# Value Counts for Discrete Columns (No changes here)
if common_discrete_cols:
    print("\n  Value Counts for Discrete Columns (Normalized Percentage %):")
    for col in common_discrete_cols:
        print(f"\n    Column: {col}")
        real_counts = df_real[col].value_counts(normalize=True).mul(100).round(2)
        synthetic_counts = df_synthetic[col].value_counts(normalize=True).mul(100).round(2)
        counts_comparison_df = pd.DataFrame({
            'Real (%)': real_counts,
            'Synthetic (%)': synthetic_counts
        }).fillna(0)
        print(counts_comparison_df)
else:
    print("  Skipping Value Counts as no common discrete columns were identified.")

# Correlation Matrix Comparison (No changes here)
if common_numerical_cols and len(common_numerical_cols) > 1:
    print("\n  Correlation Matrix Comparison (Heatmaps):")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_real[common_numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix - Real Data', fontsize=15)
    plt.tight_layout()
    plt.show()
    # print("    ✅ Correlation heatmap for Real Data generated.")

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_synthetic[common_numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix - Synthetic Data', fontsize=15)
    plt.tight_layout()
    plt.show()
    # print("    ✅ Correlation heatmap for Synthetic Data generated.")
else:
    print("  Skipping Correlation Matrix comparison (not enough common numerical columns).")

print("\n🎉 Comparison Script Finished.")
