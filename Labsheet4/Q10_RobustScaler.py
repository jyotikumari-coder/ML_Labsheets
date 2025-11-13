# Q10: Scale data using RobustScaler
import pandas as pd
from sklearn.preprocessing import RobustScaler

data = pd.DataFrame({'Values': [10, 12, 15, 20, 200]})
scaler = RobustScaler()
data['Scaled'] = scaler.fit_transform(data[['Values']])
print(data)
