import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load model and scaler
mlp = joblib.load("results/mlp_model.joblib")
scaler = joblib.load("results/scaler.joblib")

# Load and prepare data
df = pd.read_csv("data/german.data-numeric", sep='\s+', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_scaled = scaler.transform(X)

# Cross-Validation with SMOTE
from sklearn.model_selection import StratifiedKFold

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Create pipeline to apply SMOTE and then train the model
pipeline = Pipeline([('smote', smote), ('mlp', mlp)])

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation
from sklearn.model_selection import cross_validate
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'f1_weighted': 'f1_weighted'
}

scores = cross_validate(pipeline, X_scaled, y, cv=skf, scoring=scoring, return_estimator=True)

# Get the best model based on accuracy
best_index = np.argmax(scores['test_accuracy'])
best_model = scores['estimator'][best_index]

# Permutation Importance
result = permutation_importance(best_model, X_scaled, y, n_repeats=10, random_state=42)

importance_means = result.importances_mean
indices = np.argsort(importance_means)[::-1]

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

sorted_feature_names = [feature_names[i] for i in indices[:10]]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(10), importance_means[indices][:10])
plt.xticks(range(10), sorted_feature_names, rotation=45, ha='right')
plt.title("Top 10 Feature Importances (Permutation with SMOTE & Cross-Validation)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("results/permutation_importance.png")
plt.close()

print("Saved permutation_importance.png ✅")


# Print full feature importances
print("\n=== All Feature Importances (sorted) ===")
for idx in indices:
    print(f"{feature_names[idx]:<30}: {importance_means[idx]:.4f}")
