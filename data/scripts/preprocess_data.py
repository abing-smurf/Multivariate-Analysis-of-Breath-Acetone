import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

sns.set(style="whitegrid")

# Load the CSV
df = pd.read_csv("../data/test_new_data.csv")

# Preview
print("\n First 5 rows:")
print(df.head())

print("\n Data Types and Null Values:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include="all"))

# Handle missing values
df['HbA1c_%'] = df['HbA1c_%'].fillna(df['HbA1c_%'].mean())
df['Glucose_Level_Class'] = df['Glucose_Level_Class'].fillna(df['Glucose_Level_Class'].mode()[0])

print("\nMissing values per column:")
print(df.isnull().sum())

# Classification function
def classify_glucose_level(value):
    if value < 70:
        return 'Low'
    elif 70 <= value <= 139:
        return 'Normal'
    else:
        return 'High'

# Apply new classification
df['Glucose_Level_Class'] = df['Blood_Glucose_mg/dL'].apply(classify_glucose_level)

# Verify result
print(df[['Blood_Glucose_mg/dL', 'Glucose_Level_Class']].head(10))

# Save the preprocessed version
df.to_csv("../data/test_new_data.csv", index=False)

# Plot distributions
df.hist(bins=20, figsize=(12, 10), edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()
