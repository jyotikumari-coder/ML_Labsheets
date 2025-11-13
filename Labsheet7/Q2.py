from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris().data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

print("Cluster Centers:\n", kmeans.cluster_centers_)
