"""
Labsheet7 - Introduction to Machine Learning
Problem 15: Save the K-Means model using joblib and use it to predict on new samples.
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
print("Problem 15: Save the K-Means model using joblib and use it to predict on new samples.")
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
model = KMeans(n_clusters=3).fit(X)
joblib.dump(model, 'kmeans_model.pkl')

loaded = joblib.load('kmeans_model.pkl')
print("Predictions:", loaded.predict(X[:5]))

