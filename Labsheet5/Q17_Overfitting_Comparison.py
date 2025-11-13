# Q17: Compare training and test scores for overfitting
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3*X**2 + 2*X + np.random.randn(100,1)*10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for d in [1, 2, 3, 5]:
    poly = PolynomialFeatures(degree=d)
    X_train_poly, X_test_poly = poly.fit_transform(X_train), poly.transform(X_test)
    model = LinearRegression().fit(X_train_poly, y_train)
    print(f"Degree {d}: Train Score={model.score(X_train_poly, y_train):.3f}, Test Score={model.score(X_test_poly, y_test):.3f}")
