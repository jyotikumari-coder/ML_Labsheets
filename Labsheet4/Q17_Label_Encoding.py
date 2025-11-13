# Q17: Label encoding for ordinal data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({'Size': ['Small','Medium','Large','Medium']})
encoder = LabelEncoder()
df['Size_Code'] = encoder.fit_transform(df['Size'])
print(df)
