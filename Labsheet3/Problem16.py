"""
Problem 19: Fraud Detection with Imbalanced Dataset
- Dataset: Synthetic
- Handle class imbalance using SMOTE
- Evaluate with precision, recall, F1-score
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Create synthetic imbalanced dataset
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10, n_redundant=5,
    n_clusters_per_class=1, weights=[0.98, 0.02], flip_y=0, random_state=42
)

print("Original class distribution:\n", pd.Series(y).value_counts())

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("Resampled class distribution:\n", pd.Series(y_train_res).value_counts())

# 5. Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

# 6. Evaluate on test set
y_pred = rf.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
