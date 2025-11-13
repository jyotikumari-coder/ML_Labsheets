# Q13: Power transformation using PowerTransformer
import pandas as pd
from sklearn.preprocessing import PowerTransformer

data = pd.DataFrame({'Values': [1, 2, 3, 5, 10, 20, 50]})
pt = PowerTransformer()
data['Transformed'] = pt.fit_transform(data[['Values']])
print(data)
