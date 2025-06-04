import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 100

# Simulated data
data = {
    "Breath_Acetone_ppm": np.round(np.random.uniform(0.2, 10, n), 2),
    "Blood_Glucose_mg/dL": np.round(np.random.uniform(20, 300, n), 1),
    "HbA1c_%": np.round(np.random.uniform(4.5, 13.0, n), 2),
    "β-Hydroxybutyrate_mmol/L": np.round(np.random.uniform(0.1, 6.0, n), 2),
    "Age": np.random.randint(18, 80, n),
    "BMI": np.round(np.random.uniform(18.5, 40.0, n), 2),
    "Temp_C": np.round(np.random.uniform(35, 38, n), 1),
    "Humidity_%": np.round(np.random.uniform(30, 80, n), 1),
    "Fasting_Hours": np.random.randint(4, 16, n),
}

# Convert to DataFrame
df = pd.DataFrame(data)


# Classify glucose level based on Blood_Glucose_mg/dL
def classify_glucose_level(value):
    if value < 70:
        return 'Low'
    elif 70 <= value <= 139:
        return 'Normal'
    else:
        return 'High'

df["Glucose_Level_Class"] = df["Blood_Glucose_mg/dL"].apply(classify_glucose_level)

# Introduce missing values into some random rows (optional)
df.loc[np.random.choice(df.index, size=5), 'HbA1c_%'] = np.nan


# Save to CSV
df.to_csv("../data/test_new_data.csv", index=False)

print("Test data generated and saved to data/test_new_data.csv")
