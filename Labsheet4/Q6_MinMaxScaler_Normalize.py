# Q6: Normalize data using MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.DataFrame({'Values': [10, 12, 15, 20, 25]})
scaler = MinMaxScaler()
data['Normalized'] = scaler.fit_transform(data[['Values']])
print(data)
