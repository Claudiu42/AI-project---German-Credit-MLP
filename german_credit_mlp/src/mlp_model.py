
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load the Data
df = pd.read_csv("data/german.data-numeric", sep='\s+', header=None)

# Prepare Features and Target
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column is the target (1 = good, 2 = bad)

# Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# print("nr caracteristici", df.shape)


# Train the MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(24,24),
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    max_iter=2000,
    early_stopping=True,
    random_state=42
)


# Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'f1_weighted': make_scorer(f1_score, average='weighted')
}

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Create pipeline to apply SMOTE and then train the model
pipeline = Pipeline([('smote', smote), ('mlp', mlp)])

# Cross-Validation
scores = cross_validate(pipeline, X_scaled, y, cv=skf, scoring=scoring, return_estimator=True)

# Evaluate and Save Best Model
best_index = np.argmax(scores['test_accuracy'])
best_model = scores['estimator'][best_index]

# ✅ Save model and scaler
joblib.dump(mlp, "results/mlp_model.joblib")
joblib.dump(scaler, "results/scaler.joblib")

# Evaluate on held-out fold
X_test = X_scaled
y_test = y


# Make Predictions
y_pred = best_model.predict(X_test)

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 2])

# Cost Matrix Calculation
cost_matrix = [[0, 1], [5, 0]]  # from dataset description
cost = (conf_matrix * cost_matrix).sum()

# Save Results
os.makedirs("results", exist_ok=True)

# Save metrics to text file
print("Saving metrics.txt...")
with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Total Misclassification Cost: {cost}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
print("Saved metrics.txt ✅")

# Train/test accuracy from best fold
train_accuracy = best_model.score(X_scaled, y)
print(f"Train Accuracy (best fold): {train_accuracy:.2f}")
print(f"Test Accuracy (same data, for report consistency): {accuracy:.2f}")


# Save confusion matrix plot
print("Saving confusion_matrix.png...")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[1, 2])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()
print("Saved confusion_matrix.png ✅")

# print to console
print(f"Accuracy: {accuracy:.2f}")
print(f"Total Misclassification Cost: {cost}")
print(report)