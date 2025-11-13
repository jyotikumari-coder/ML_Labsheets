# Q8: Compare MinMaxScaler vs StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.DataFrame({'Values': [10, 12, 15, 20, 25]})
minmax = MinMaxScaler().fit_transform(data[['Values']])
standard = StandardScaler().fit_transform(data[['Values']])

plt.figure(figsize=(8,4))
plt.hist(minmax, alpha=0.5, label='MinMaxScaler')
plt.hist(standard, alpha=0.5, label='StandardScaler')
plt.legend()
plt.title("Scaler Comparison")
plt.show()
