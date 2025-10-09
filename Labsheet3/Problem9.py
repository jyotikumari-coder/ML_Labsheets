"""
Problem 9: Compare Bagging vs Boosting with multiple base learners
- Base learners: Decision Tree, KNN, Logistic Regression
- Dataset: Breast Cancer
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Standardize features for LR and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define base learners
base_learners = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42)
}

# 4. Evaluate Bagging and Boosting
results = {}

for name, clf in base_learners.items():
    # 4a. Bagging
    bag = BaggingClassifier(base_estimator=clf, n_estimators=50, random_state=42)
    bag.fit(X_train_scaled if name != "Decision Tree" else X_train, y_train)
    y_pred_bag = bag.predict(X_test_scaled if name != "Decision Tree" else X_test)
    acc_bag = accuracy_score(y_test, y_pred_bag)

    # 4b. Boosting (AdaBoost) - works well with DT as base learner
    # For KNN and LR, AdaBoost may fail, so we use DecisionTree(max_depth=1)
    boost_base = clf
    if name != "Decision Tree":
        boost_base = DecisionTreeClassifier(max_depth=1, random_state=42)
    boost = AdaBoostClassifier(base_estimator=boost_base, n_estimators=50, random_state=42)
    boost.fit(X_train_scaled if name != "Decision Tree" else X_train, y_train)
    y_pred_boost = boost.predict(X_test_scaled if name != "Decision Tree" else X_test)
    acc_boost = accuracy_score(y_test, y_pred_boost)

    results[name] = {"Bagging": acc_bag, "Boosting": acc_boost}

# 5. Display results
print("\n=== Bagging vs Boosting Accuracy ===")
for name in results:
    print(f"{name}: Bagging={results[name]['Bagging']:.4f}, Boosting={results[name]['Boosting']:.4f}")
