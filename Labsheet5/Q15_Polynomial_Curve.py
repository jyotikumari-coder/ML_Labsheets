# Q15: Visualize polynomial regression curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.arange(1, 10).reshape(-1, 1)
y = [2*x**2 + np.random.randint(1, 10) for x in X]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X_poly), color='red')
plt.title("Polynomial Regression Curve")
plt.show()
