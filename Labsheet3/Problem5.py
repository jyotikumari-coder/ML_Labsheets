# Q5: Feature Selection using Backward Elimination vs Lasso Regression
# Author: Your Name
# MCA III Semester - Lab Sheet 03

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================================================
# Backward Elimination using Statsmodels
# =======================================================
X_train_sm = sm.add_constant(X_train)  # add intercept
model = sm.Logit(y_train, X_train_sm)
result = model.fit(disp=False)

p_max = 1
selected_features = list(range(X_train.shape[1]))  # all features initially

while p_max > 0.05:  # keep removing until all features are significant
    model = sm.Logit(y_train, sm.add_constant(X_train[:, selected_features]))
    result = model.fit(disp=False)
    p_values = result.pvalues[1:]  # skip intercept
    p_max = max(p_values)
    if p_max > 0.05:
        worst_feature = selected_features[p_values.argmax()]
        selected_features.remove(worst_feature)

print("\nBackward Elimination selected features:")
print([feature_names[i] for i in selected_features])

# =======================================================
# Lasso Regression for Feature Selection
# =======================================================
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

lasso_selected = np.where(lasso.coef_ != 0)[0]

print("\nLasso Regression selected features:")
print([feature_names[i] for i in lasso_selected])

# =======================================================
# Compare Results
# =======================================================
print(f"\nNumber of features selected by Backward Elimination: {len(selected_features)}")
print(f"Number of features selected by Lasso Regression: {len(lasso_selected)}")

print("\nOverlap of selected features:")
print(set([feature_names[i] for i in selected_features]) & set([feature_names[i] for i in lasso_selected]))
