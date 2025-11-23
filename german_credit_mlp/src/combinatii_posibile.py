import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/german.data-numeric", sep="\s+", header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE + pipeline setup
smote = SMOTE(random_state=42)
pipeline = Pipeline([
    ('smote', smote),
    ('mlp', MLPClassifier(
        activation='relu',
        solver='adam',
        max_iter=2000,
        early_stopping=True,
        random_state=42
    ))
])

# Define parameter grid
param_grid = {
    'mlp__hidden_layer_sizes': [(24,), (24, 24), (24, 12), (24, 48)],
    'mlp__learning_rate_init': [0.01, 0.1]
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# Run grid search
grid_search.fit(X_scaled, y)

# Output best configuration
print("Best parameters:", grid_search.best_params_)
print("Best F1_macro score:", grid_search.best_score_)

# Get predictions from best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_scaled)

# Save all grid search results
pd.DataFrame(grid_search.cv_results_).to_excel("results/full_grid_search_results.xlsx", index=False)

print("\n=== Best Model ===")
print("Parameters:", grid_search.best_params_)
print("F1 Macro Score:", round(grid_search.best_score_, 4))
print("Results saved to: results/full_grid_search_results.xlsx")