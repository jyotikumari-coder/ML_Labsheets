"""
Problem 8: Ensemble Voting Classifier
- Base models: Logistic Regression, Decision Tree, SVM
- Evaluate ensemble vs individual classifiers
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Standardize features for LR and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define base models
clf1 = LogisticRegression(random_state=42, max_iter=5000)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = SVC(probability=True, random_state=42)

# 5. Individual model evaluation
for name, clf, X_in in zip(
    ["Logistic Regression", "Decision Tree", "SVM"],
    [clf1, clf2, clf3],
    [X_train_scaled, X_train, X_train_scaled]
):
    clf.fit(X_in, y_train)
    y_pred = clf.predict(X_test_scaled if name != "Decision Tree" else X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 6. Create Voting Classifier (soft voting)
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)],
    voting='soft'  # soft = probabilities averaged
)
voting_clf.fit(X_train_scaled, y_train)  # all inputs scaled for simplicity
y_pred_vote = voting_clf.predict(X_test_scaled)
acc_vote = accuracy_score(y_test, y_pred_vote)

print(f"\nEnsemble Voting Classifier Accuracy: {acc_vote:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_vote))
print("Classification Report:\n", classification_report(y_test, y_pred_vote))

# 7. Comparison summary
print("\n--- Summary ---")
print("Individual models vs Ensemble accuracy:")
print(f"Logistic Regression: {accuracy_score(y_test, clf1.predict(X_test_scaled)):.4f}")
print(f"Decision Tree: {accuracy_score(y_test, clf2.predict(X_test)):.4f}")
print(f"SVM: {accuracy_score(y_test, clf3.predict(X_test_scaled)):.4f}")
print(f"Voting Ensemble: {acc_vote:.4f}")
