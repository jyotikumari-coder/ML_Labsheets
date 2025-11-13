# Q20: Plot effect of alpha on coefficients
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3*X**2 + 2*X + np.random.randn(100,1)*10
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

alphas = [0.01, 0.1, 1, 10, 100]
coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_poly, y)
    coefs.append(ridge.coef_[0])

plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient values')
plt.title('Effect of Alpha on Ridge Coefficients')
plt.show()
