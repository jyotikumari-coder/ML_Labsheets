# Q14: Train polynomial regression and compare with linear
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

X = np.arange(1, 10).reshape(-1, 1)
y = [2*x + np.random.randint(1, 5) for x in X]

# Linear Regression
lin_reg = LinearRegression().fit(X, y)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression().fit(X_poly, y)

plt.scatter(X, y, color='blue')
plt.plot(X, lin_reg.predict(X), color='red', label='Linear')
plt.plot(X, poly_reg.predict(X_poly), color='green', label='Polynomial')
plt.legend()
plt.show()
