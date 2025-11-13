# Q12: Log transformation on skewed data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({'Values': [1, 2, 3, 5, 10, 20, 50, 100, 200]})
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist(data['Values'], bins=10)
plt.title("Original")
plt.subplot(1,2,2)
plt.hist(np.log1p(data['Values']), bins=10)
plt.title("After Log Transformation")
plt.show()
