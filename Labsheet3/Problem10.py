"""
Problem 10: Analyze AdaBoost hyperparameters (n_estimators, learning_rate) using GridSearchCV
- Dataset: Breast Cancer
- Base learner: DecisionTreeClassifier (stump)
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Base estimator for AdaBoost
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)  # Decision stump

# 4. Define AdaBoost classifier
ada = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)

# 5. Define grid for hyperparameters
param_grid = {
    "n_estimators": [10, 50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1]
}

# 6. Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 7. Best hyperparameters
print("Best hyperparameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# 8. Evaluate on test set
best_ada = grid_search.best_estimator_
y_pred = best_ada.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("\nTest set accuracy:", test_acc)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Analyze effect: plot heatmap of CV scores
import numpy as np
import seaborn as sns

scores_mean = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['n_estimators']),
                                                                 len(param_grid['learning_rate']))

plt.figure(figsize=(10,6))
sns.heatmap(scores_mean, annot=True, fmt=".4f", xticklabels=param_grid['learning_rate'],
            yticklabels=param_grid['n_estimators'], cmap="YlGnBu")
plt.xlabel("Learning Rate")
plt.ylabel("Number of Estimators")
plt.title("AdaBoost Accuracy (CV) for Different Hyperparameters")
plt.show()
