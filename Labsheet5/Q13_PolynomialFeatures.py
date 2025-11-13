# Q13: Transform feature into polynomial terms
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[1], [2], [3], [4]])
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print(X_poly)
