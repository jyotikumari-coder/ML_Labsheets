# Q7: Plot regression line for simple regression
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = load_boston(return_X_y=True)
X = X[:, [5]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, model.predict(X_test), color='red', label='Regression Line')
plt.xlabel('RM (Average rooms per dwelling)')
plt.ylabel('Price')
plt.legend()
plt.show()
