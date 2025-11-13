# Q5: Apply log and sqrt transformations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({'Values': [1, 2, 3, 5, 10, 20, 50, 100, 200]})
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(data['Values'], bins=10)
plt.title("Original")
plt.subplot(1,3,2)
plt.hist(np.log1p(data['Values']), bins=10)
plt.title("Log Transform")
plt.subplot(1,3,3)
plt.hist(np.sqrt(data['Values']), bins=10)
plt.title("Square Root Transform")
plt.show()
