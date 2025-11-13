# Q20: Feature selection using SelectKBest
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, f_regression

# Load example dataset
X, y = load_boston(return_X_y=True)
selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)
print("Top 5 Features Indices:", selector.get_support(indices=True))
