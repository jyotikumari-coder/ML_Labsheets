"""
Labsheet7 - Introduction to Machine Learning
Problem 16: Apply Agglomerative Clustering on the Iris dataset.
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
print("Problem 16: Apply Agglomerative Clustering on the Iris dataset.")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
agg = AgglomerativeClustering(n_clusters=3)
df['Cluster'] = agg.fit_predict(df)
print(df.head())
