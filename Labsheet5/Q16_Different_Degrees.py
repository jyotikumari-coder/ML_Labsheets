# Q16: Compare polynomial regression degrees
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.linspace(0, 10, 20).reshape(-1, 1)
y = 3*X**2 + 2*X + np.random.randn(20,1)*10

plt.scatter(X, y, color='black')
for d in [2, 3, 4]:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    plt.plot(X, model.predict(X_poly), label=f"Degree {d}")
plt.legend()
plt.show()
