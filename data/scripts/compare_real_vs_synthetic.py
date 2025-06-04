import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load real and synthetic datasets
real_df = pd.read_csv("../data/dummy_glucose_data.csv")
synthetic_df = pd.read_csv("../data/test_new_data.csv")  # Update this if you saved it with a different name

# Preview shape and columns
print("Real Data Shape:", real_df.shape)
print("Synthetic Data Shape:", synthetic_df.shape)
print("\nColumns match:", list(real_df.columns) == list(synthetic_df.columns))

# Combine for comparison
real_df['Source'] = 'Real'
synthetic_df['Source'] = 'ctgan'
combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

# Compare numeric distributions
numeric_cols = real_df.select_dtypes(include='number').columns

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=combined_df, x=col, hue='Source', fill=True, common_norm=False, alpha=0.5)
    plt.title(f'Distribution Comparison for {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

# Optional: Boxplots
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=combined_df, x='Source', y=col)
    plt.title(f'Boxplot Comparison for {col}')
    plt.tight_layout()
    plt.show()
