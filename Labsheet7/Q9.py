"""
Labsheet7 - Introduction to Machine Learning
Problem 9: Apply K-Means on a dataset with two highly correlated features and interpret results.
"""

# Package Name: Labsheet7

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Your implementation here
print("Problem 9: Apply K-Means on a dataset with two highly correlated features and interpret results.")

np.random.seed(42)
x = np.random.rand(100)
y = x * 0.8 + np.random.rand(100) * 0.1
data = pd.DataFrame({'X': x, 'Y': y})

kmeans = KMeans(n_clusters=2).fit(data)
data['Cluster'] = kmeans.labels_

print("Silhouette Score:", silhouette_score(data[['X', 'Y']], data['Cluster']))
