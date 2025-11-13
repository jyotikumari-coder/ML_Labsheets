# Q19: Lasso regression (L1 regularization)
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3*X**2 + 2*X + np.random.randn(100,1)*10
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train, y_train)
print("Lasso Score:", lasso.score(X_test, y_test))
