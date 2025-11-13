# Q9: Linear regression using statsmodels
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

X = sm.add_constant(df[['RM', 'LSTAT']])
y = df['PRICE']
model = sm.OLS(y, X).fit()
print(model.summary())
