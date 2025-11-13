# Q1: Visualize data distribution before and after outlier removal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({'Values': [10, 12, 11, 13, 12, 15, 200, 13, 11, 12]})

plt.figure(figsize=(8,4))
plt.hist(data['Values'], bins=10, color='skyblue', edgecolor='black')
plt.title("Before Outlier Removal")
plt.show()

Q1 = data['Values'].quantile(0.25)
Q3 = data['Values'].quantile(0.75)
IQR = Q3 - Q1
filtered = data[(data['Values'] >= Q1 - 1.5 * IQR) & (data['Values'] <= Q3 + 1.5 * IQR)]

plt.hist(filtered['Values'], bins=10, color='lightgreen', edgecolor='black')
plt.title("After Outlier Removal")
plt.show()
