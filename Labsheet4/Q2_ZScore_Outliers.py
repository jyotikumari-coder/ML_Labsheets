# Q2: Compute Z-scores and highlight extreme values
import pandas as pd
from scipy import stats

data = pd.DataFrame({'Values': [10, 12, 11, 13, 12, 15, 200, 13, 11, 12]})
data['Z_Score'] = stats.zscore(data['Values'])
data['Outlier'] = data['Z_Score'].apply(lambda x: 'Yes' if abs(x) > 3 else 'No')
print(data)
