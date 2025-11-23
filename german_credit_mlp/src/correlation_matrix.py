import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/german.data-numeric", sep='\s+', header=None)

# Feature names (based on your dataset)
feature_names = [
    "Checking account status",
    "Duration (months)",
    "Credit history",
    "Purpose",
    "Credit amount",
    "Savings account/bonds",
    "Employment since",
    "Installment rate %",
    "Personal status and sex",
    "Other debtors/guarantors",
    "Residence since",
    "Property",
    "Age",
    "Other installment plans",
    "Housing",
    "Existing credits",
    "Job",
    "Dependents",
    "Telephone",
    "Foreign worker",
    "Binary flag 1",
    "Binary flag 2",
    "Binary flag 3",
    "Binary flag 4"
]

# Assign column names to DataFrame
df.columns = feature_names + ["Target"]

# Compute correlation matrix
corr_matrix = df.drop(columns=["Target"]).corr()

# Plot correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, xticklabels=feature_names, yticklabels=feature_names)
plt.title("Correlation Matrix Heatmap")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("results/correlation_matrix_named.png")
plt.close()

print("Saved correlation_matrix_named.png ✅")
