"""
Problem 6: Decision Tree on a medical dataset and pruning demonstration
- Dataset: Pima Indians Diabetes (if datasets/diabetes.csv exists),
  otherwise falls back to sklearn's breast cancer dataset for quick run.
- Demonstrates:
    1) Cost-complexity pruning (ccp_alpha)
    2) Max-depth control pruning
- Outputs:
    - Accuracy, confusion matrix, classification report
    - Plots: ccp_alpha vs accuracy, max_depth vs accuracy
    - Tree plot before and after pruning
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def load_medical_dataset():
    """
    Attempt to load datasets/diabetes.csv (Pima Indians Diabetes).
    If not found, fall back to sklearn's breast cancer dataset.
    Expected diabetes.csv columns: features + target column named 'Outcome' or 'target'
    """
    csv_path = "datasets/diabetes.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # try common name 'Outcome' or 'target'
        if 'Outcome' in df.columns:
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
        elif 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
        else:
            # assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        dataset_name = "Pima Diabetes (from datasets/diabetes.csv)"
    else:
        # fallback
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        dataset_name = "Breast Cancer (sklearn fallback)"
    return X, y, dataset_name


def evaluate_model(clf, X_test, y_test, desc="Model"):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Evaluation: {desc} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return acc


def plot_tree_fig(clf, feature_names, title, figsize=(12, 8), max_depth_plot=3):
    plt.figure(figsize=figsize)
    # plot only up to a few levels to keep figure readable
    plot_tree(clf, feature_names=feature_names, class_names=True, filled=True, max_depth=max_depth_plot)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    # 1. Load dataset
    X, y, dataset_name = load_medical_dataset()
    print("Dataset used:", dataset_name)
    print("Data shape:", X.shape)
    feature_names = X.columns.tolist()

    # 2. Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Decision trees don't require scaling, but we'll show optional scaling commented:
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # 3. Train an unpruned decision tree (baseline)
    clf_base = DecisionTreeClassifier(random_state=42)
    clf_base.fit(X_train, y_train)
    acc_base = evaluate_model(clf_base, X_test, y_test, desc="Unpruned Decision Tree (baseline)")

    # Plot baseline tree (show top levels to keep readable)
    print("\nPlotting baseline tree (top levels)...")
    plot_tree_fig(clf_base, feature_names, "Baseline Decision Tree (top levels)", max_depth_plot=3)

    # ----------------------------
    # 4. Cost-Complexity Pruning (ccp_alpha path)
    # ----------------------------
    print("\n--- Cost-complexity pruning path ---")
    path = clf_base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    # We will ignore the last alpha which prunes everything
    ccp_alphas = ccp_alphas[:-1]

    # For each alpha, train a tree and evaluate with cross-validation or holdout
    clfs = []
    train_scores = []
    val_scores = []
    # use stratified kfold for stable scores
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for a in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
        # cross-validated accuracy on training set (to estimate generalization)
        scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
        val_scores.append(scores.mean())
        clfs.append(clf)

    # Plot alpha vs CV accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(ccp_alphas, val_scores, marker='o')
    plt.xscale('log')
    plt.xlabel('ccp_alpha (log scale)')
    plt.ylabel('CV Accuracy (train set)')
    plt.title('Cost-Complexity pruning: alpha vs CV accuracy')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Pick alpha with best CV accuracy (tie-breaker: smallest alpha)
    best_idx = int(np.argmax(val_scores))
    best_alpha = ccp_alphas[best_idx]
    print(f"Best ccp_alpha by CV: {best_alpha:.6f} (CV acc={val_scores[best_idx]:.4f})")

    # Fit tree with best alpha and evaluate on test set
    clf_ccp = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    clf_ccp.fit(X_train, y_train)
    acc_ccp = evaluate_model(clf_ccp, X_test, y_test, desc=f"Pruned Decision Tree (ccp_alpha={best_alpha:.6f})")

    print("\nPlotting pruned tree (top levels)...")
    plot_tree_fig(clf_ccp, feature_names, f"Pruned Tree (ccp_alpha={best_alpha:.6f})", max_depth_plot=3)

    # ----------------------------
    # 5. Depth control pruning: grid search over max_depth
    # ----------------------------
    print("\n--- Max depth control: grid search ---")
    depths = list(range(1, 21))  # try depths 1..20
    depth_scores = []
    for d in depths:
        clf_d = DecisionTreeClassifier(max_depth=d, random_state=42)
        scores = cross_val_score(clf_d, X_train, y_train, cv=skf, scoring='accuracy')
        depth_scores.append(scores.mean())

    # Plot depth vs CV accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(depths, depth_scores, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('CV Accuracy (train set)')
    plt.title('Depth control: max_depth vs CV accuracy')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    best_depth = depths[int(np.argmax(depth_scores))]
    print(f"Best max_depth by CV: {best_depth} (CV acc={max(depth_scores):.4f})")

    # Fit tree with best depth
    clf_depth = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
    clf_depth.fit(X_train, y_train)
    acc_depth = evaluate_model(clf_depth, X_test, y_test, desc=f"Pruned Decision Tree (max_depth={best_depth})")

    print("\nPlotting depth-limited tree (top levels)...")
    plot_tree_fig(clf_depth, feature_names, f"Depth-limited Tree (max_depth={best_depth})", max_depth_plot=3)

    # ----------------------------
    # 6. Compare results summary
    # ----------------------------
    print("\n=== Summary of models ===")
    print(f"Unpruned accuracy: {acc_base:.4f}")
    print(f"Cost-complexity pruned accuracy (ccp_alpha={best_alpha:.6f}): {acc_ccp:.4f}")
    print(f"Depth-limited pruned accuracy (max_depth={best_depth}): {acc_depth:.4f}")

    print("\nNotes:")
    print("- Cost-complexity pruning uses ccp_alpha to trade complexity vs training fit.")
    print("- Depth control forces a maximum tree size; useful when interpretability and small trees are desired.")
    print("- Use cross-validation when choosing ccp_alpha or max_depth to avoid overfitting to the test set.")
    print("- Inspect the trees visually to ensure they remain interpretable (few levels).")


if __name__ == "__main__":
    main()
