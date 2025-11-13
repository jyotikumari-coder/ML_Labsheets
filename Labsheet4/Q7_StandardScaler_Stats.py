# Q7: Standardize dataset and show mean, std
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({'Values': [10, 12, 15, 20, 25]})
scaler = StandardScaler()
scaled = scaler.fit_transform(data[['Values']])
print("After Standardization:\nMean:", scaled.mean(), " Std:", scaled.std())
