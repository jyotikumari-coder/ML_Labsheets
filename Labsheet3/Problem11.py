"""
Problem 11: Stacking Classifier vs Bagging and Boosting
- Base learners: Decision Tree, KNN, SVM
- Meta-learner: Logistic Regression
- Dataset: Breast Cancer
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Standardize features (for LR and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define base learners
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True, random_state=42))
]

# 5. Define meta-learner
meta_learner = LogisticRegression(max_iter=5000, random_state=42)

# 6. Create Stacking Classifier
stack_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
stack_clf.fit(X_train_scaled, y_train)

# 7. Evaluate Stacking
y_pred_stack = stack_clf.predict(X_test
