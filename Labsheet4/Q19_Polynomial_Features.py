# Q19: Create polynomial features
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
poly = PolynomialFeatures(degree=2, include_bias=False)
transformed = poly.fit_transform(df)
print(pd.DataFrame(transformed, columns=poly.get_feature_names_out(['x1','x2'])))
