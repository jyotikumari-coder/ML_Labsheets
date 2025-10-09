# Q4: Logistic Regression with L1 and L2 Regularization
# Author: Your Name
# MCA III Semester - Lab Sheet 03

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Logistic Regression with L1 and L2
models = {
    "L1": LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=42),
    "L2": LogisticRegression(penalty="l2", solver="saga", C=1.0, max_iter=5000, random_state=42)
}

results = {}

for reg, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    non_zero_coeffs = np.sum(model.coef_ != 0)
    
    results[reg] = {
        "Accuracy": accuracy,
        "Non-zero Coefficients": non_zero_coeffs,
        "Coefficients": model.coef_[0]
    }

# 5. Print results
print("=== Comparison of L1 vs L2 Logistic Regression ===")
for reg in results:
    print(f"\n{reg} Regularization:")
    print(f"Accuracy: {results[reg]['Accuracy']:.4f}")
    print(f"Non-zero Coefficients: {results[reg]['Non-zero Coefficients']} / {len(feature_names)}")

# 6. Plot coefficient comparison
plt.figure(figsize=(12,6))
plt.plot(results["L1"]["Coefficients"], 'ro-', label="L1 (Sparse)")
plt.plot(results["L2"]["Coefficients"], 'bo-', label="L2 (Dense)")
plt.axhline(0, color="black", linewidth=0.7)
plt.title("Logistic Regression Coefficients: L1 vs L2")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()
