# Q4: Compare summary statistics before and after outlier removal
import pandas as pd

data = pd.DataFrame({'Values': [10, 12, 11, 13, 12, 15, 200, 13, 11, 12]})
before = data.describe()

Q1, Q3 = data['Values'].quantile(0.25), data['Values'].quantile(0.75)
IQR = Q3 - Q1
filtered = data[(data['Values'] >= Q1 - 1.5 * IQR) & (data['Values'] <= Q3 + 1.5 * IQR)]
after = filtered.describe()

print("Before Outlier Removal:\n", before)
print("\nAfter Outlier Removal:\n", after)
