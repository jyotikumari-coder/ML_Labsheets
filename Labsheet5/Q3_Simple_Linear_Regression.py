# Q3: Simple linear regression using one feature
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = load_boston(return_X_y=True)
X = X[:, [5]]  # RM feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("Trained Simple Linear Regression Model")
