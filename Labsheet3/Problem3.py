"""
Problem 3: Ridge and Lasso regression on a multicollinear dataset
- Create multicollinear features
- Fit Ridge and Lasso across alphas
- Plot coefficient paths (how coefficients shrink with alpha)
- Compare test MSE and show Lasso sparsity
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def make_multicollinear_data(n_samples=200, n_features=20, n_informative=5, corr=0.95, random_state=42):
    """
    Create synthetic linear regression data where many features are highly correlated.
    Approach:
      - Create n_informative base features.
      - For other features, make them linear combinations of informative features + small noise.
    """
    rng = np.random.RandomState(random_state)
    # informative base features
    X_base = rng.normal(size=(n_samples, n_informative))
    # true coefficients for base features
    beta_base = rng.uniform(-5, 5, size=n_informative)
    # generate dependent features (multicollinear) by mixing base features
    X = np.zeros((n_samples, n_features))
    for j in range(n_features):
        # choose random combination weights for base features
        w = rng.normal(size=n_informative)
        # scale to control correlation strength
        X[:, j] = X_base.dot(w) * corr + rng.normal(scale=(1 - corr), size=n_samples) * 0.1

    # Construct target using only first few base features (sparse true model)
    y = X_base.dot(beta_base) + rng.normal(scale=1.0, size=n_samples)
    return X, y, beta_base

def plot_coefficient_paths(alphas, coefs, labels, title):
    """
    Plot coefficient paths: each line is a coefficient value across alphas (log scale).
    - alphas: array-like (descending or ascending)
    - coefs: shape (n_alphas, n_features)
    - labels: optional list of labels for legend (we won't label every line)
    """
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], linewidth=1)
    plt.xscale("log")
    plt.gca().invert_xaxis()  # so small alpha (less regularization) on left
    plt.xlabel("alpha (log scale) — smaller = less regularization")
    plt.ylabel("Coefficient value")
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def main():
    # 1) Generate multicollinear data
    X, y, beta_base = make_multicollinear_data(n_samples=400, n_features=30, n_informative=5, corr=0.98, random_state=0)
    print("Data shape:", X.shape)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 3) Standardize features (important for regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Prepare alpha grid (from large regularization to tiny)
    alphas = np.logspace(3, -4, 80)  # large -> small

    # 5) Compute Ridge coefficient paths manually for alphas
    ridge_coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=True, max_iter=10000)
        ridge.fit(X_train_scaled, y_train)
        ridge_coefs.append(ridge.coef_)
    ridge_coefs = np.array(ridge_coefs)

    # 6) Compute Lasso coefficient paths
    lasso_coefs = []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        lasso_coefs.append(lasso.coef_)
    lasso_coefs = np.array(lasso_coefs)

    # 7) Plot coefficient paths
    plot_coefficient_paths(alphas, ridge_coefs, None, "Ridge Coefficient Paths (coeff vs alpha)")
    plot_coefficient_paths(alphas, lasso_coefs, None, "Lasso Coefficient Paths (coeff vs alpha)")

    # 8) Cross-validated best alpha for Ridge and Lasso (RidgeCV uses built-in)
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(X_train_scaled, y_train)
    best_ridge_alpha = ridge_cv.alpha_
    print("Best Ridge alpha (CV):", best_ridge_alpha)

    lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
    lasso_cv.fit(X_train_scaled, y_train)
    best_lasso_alpha = lasso_cv.alpha_
    print("Best Lasso alpha (CV):", best_lasso_alpha)

    # 9) Fit final models with best alpha and evaluate
    ridge_best = Ridge(alpha=best_ridge_alpha).fit(X_train_scaled, y_train)
    lasso_best = Lasso(alpha=best_lasso_alpha, max_iter=10000).fit(X_train_scaled, y_train)

    y_pred_ridge = ridge_best.predict(X_test_scaled)
    y_pred_lasso = lasso_best.predict(X_test_scaled)

    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    print(f"Test MSE - Ridge: {mse_ridge:.4f}, Lasso: {mse_lasso:.4f}")

    # 10) Show coefficient sparsity for Lasso
    n_nonzero_lasso = np.sum(lasso_best.coef_ != 0)
    print(f"Lasso non-zero coefficients: {n_nonzero_lasso} / {len(lasso_best.coef_)}")

    # 11) Bar plot of final coefficients (Ridge vs Lasso)
    idx = np.arange(len(ridge_best.coef_))
    width = 0.4
    plt.figure(figsize=(12, 5))
    plt.bar(idx - width/2, ridge_best.coef_, width=width, label="Ridge Coefs")
    plt.bar(idx + width/2, lasso_best.coef_, width=width, label="Lasso Coefs")
    plt.xlabel("Feature index")
    plt.ylabel("Coefficient value")
    plt.title("Final Coefficients: Ridge vs Lasso (best alpha by CV)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 12) Print top absolute coefficients (for interpretability)
    abs_coef_idx_ridge = np.argsort(np.abs(ridge_best.coef_))[::-1][:10]
    abs_coef_idx_lasso = np.argsort(np.abs(lasso_best.coef_))[::-1][:10]
    print("Top Ridge coefficient indices (by abs value):", abs_coef_idx_ridge)
    print("Top Lasso coefficient indices (by abs value):", abs_coef_idx_lasso)

    # 13) Optional: show sample coefficients values for first 10 features
    for i in range(10):
        print(f"Feature {i}: Ridge coef={ridge_best.coef_[i]:.4f}, Lasso coef={lasso_best.coef_[i]:.4f}")

if __name__ == "__main__":
    main()
