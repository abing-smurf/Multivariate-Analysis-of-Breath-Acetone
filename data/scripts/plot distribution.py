import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your classified data
df = pd.read_csv("../data/synthetic_data_ctgan_conditional_verified.csv")

# Plot BGL vs Class
plt.figure(figsize=(10, 6))
sns.stripplot(x="Glucose_Level_Class", y="Blood_Glucose_mg/dL", data=df, jitter=True, palette="Set2", size=7)
plt.title("Blood Glucose Levels by Classification")
plt.xlabel("Glucose Level Class")
plt.ylabel("Blood Glucose (mg/dL)")
plt.grid(True)
plt.tight_layout()
plt.show()
