import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# Update this path to your augmented data file
DATA_FILE = "../data/augmented_data_by_noise.csv"
TARGET_COLUMN = "Glucose_classification_class"

# List of numerical features to plot against the target class
FEATURES_TO_PLOT = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Pulse rate',
    'HbA1c_%', 'Respiratory Rate', 'Acetone PPM 1.1', 'Temperature', 'Humidity'
]

# --- Main Script ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Successfully loaded data from '{DATA_FILE}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Data file '{DATA_FILE}' not found. Please check the path.")
    exit()
except Exception as e:
    print(f"❌ An error occurred while loading the data: {e}")
    exit()

# Define the order for the x-axis categories
class_order = ['Low', 'Normal', 'High']
# Filter the dataframe to only include classes in the specified order
df = df[df[TARGET_COLUMN].isin(class_order)]

print("\n📊 Generating plots for each feature by glucose class...")

for feature in FEATURES_TO_PLOT:
    if feature not in df.columns:
        print(f"⚠️ Warning: Column '{feature}' not found in the dataframe. Skipping plot.")
        continue

    plt.figure(figsize=(12, 7))

    # Create a boxplot to show the main distribution (quartiles, median)
    sns.boxplot(
        x=TARGET_COLUMN,
        y=feature,
        data=df,
        order=class_order,
        palette="pastel",
        showfliers=False  # Hide outlier points as the stripplot will show them
    )

    # Overlay a stripplot to show individual data points with some jitter
    sns.stripplot(
        x=TARGET_COLUMN,
        y=feature,
        data=df,
        order=class_order,
        jitter=True,
        alpha=0.5,  # Use transparency to see point density
        palette="muted",
        size=4
    )

    plt.title(f'Distribution of "{feature}" by Glucose Class', fontsize=16, fontweight='bold')
    plt.xlabel("Glucose Level Class", fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

print("\n🎉 All plots generated. Script finished.")
